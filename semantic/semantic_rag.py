import numpy as np
import torch
import open_clip
import time
from pathlib import Path
from PIL import Image
import pickle
import sys
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
sys.path.append(str(Path(__file__).parent.parent))

from baseline.video_utils import extract_frames_at_fps, create_chunks
from semantic.description_generator import DescriptionGenerator
from semantic.semantic_chunker import SemanticChunker

class SemanticRAG:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-2B-Instruct", 
                 similarity_threshold=0.65, load_model=True):
        self.similarity_threshold = similarity_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Always set cache_dir regardless of load_model
        self.cache_dir = Path("data/semantic_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if load_model:
            print(f"Loading Semantic VLM: {model_name}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            print(f"Semantic VLM loaded on {self.device}")
        else:
            print("Semantic RAG initialized without model (cache-only mode)")
            self.model = None
            self.processor = None
        
    def embed_frames(self, frames):
        """Embed frames using CLIP"""
        if not frames:
            return np.array([])
        
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                pil_frames.append(Image.fromarray(frame))
            else:
                pil_frames.append(frame)
        
        processed_frames = torch.stack([
            self.preprocess(frame) for frame in pil_frames
        ]).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_image(processed_frames)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    def embed_text(self, text):
        """Embed text using CLIP"""
        text_tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        return text_embedding.cpu().numpy()
    
    def retrieve_top_k_chunks(self, chunk_embeddings, query_embedding, k=10):
        """Retrieve top-K most similar chunks"""
        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices, similarities[top_k_indices]
    
    def process_video(self, video_path, query, k=10):
        """Main pipeline with caching support"""
        timing = {}
        
        print(f"Processing video: {video_path}")
        print(f"Query: {query}")
        
        # Try to load from cache
        cached = self._load_from_cache(video_path)
        
        if cached is not None:
            semantic_chunks, semantic_descriptions, chunk_embeddings = cached
            timing['description_generation'] = 0
            timing['semantic_merging'] = 0
            timing['embedding'] = 0
            timing['frame_extraction'] = 0
            timing['uniform_chunking'] = 0
            print(f"Using cached semantic chunks: {len(semantic_chunks)} chunks")
        else:
            # Full processing pipeline
            # 1. Extract frames
            start = time.time()
            frames = extract_frames_at_fps(video_path, fps=1)
            if not frames:
                return None, None, None
            timing['frame_extraction'] = time.time() - start
            
            # 2. Create uniform chunks
            start = time.time()
            uniform_chunks = create_chunks(frames, chunk_size=3)
            print(f"Created {len(uniform_chunks)} uniform chunks")
            timing['uniform_chunking'] = time.time() - start
            
            # 3. Generate descriptions
            start = time.time()
            descriptions = self.desc_generator.generate_batch_descriptions(uniform_chunks)
            timing['description_generation'] = time.time() - start
            print(f"Description generation took {timing['description_generation']:.2f}s")
            
            # 4. Merge into semantic chunks
            start = time.time()
            semantic_chunks, semantic_descriptions = self.chunker.merge_chunks(
                uniform_chunks, descriptions
            )
            timing['semantic_merging'] = time.time() - start

            # 5. Extract frame indices for each semantic chunk
            semantic_chunks_indices = []
            for chunk in semantic_chunks:
                chunk_indices = []
                for frame in chunk:
                    # Find this frame's index in original frames list
                    for idx, orig_frame in enumerate(frames):
                        if np.array_equal(frame, orig_frame):
                            chunk_indices.append(idx)
                            break
                semantic_chunks_indices.append(chunk_indices)

            # 6. Embed semantic chunks
            start = time.time()
            chunk_embeddings = []
            for chunk in semantic_chunks:
                if len(chunk) > 0:
                    mid_idx = len(chunk) // 2
                    mid_frame_emb = self.embed_frames([chunk[mid_idx]])
                    if len(mid_frame_emb) > 0:
                        chunk_embeddings.append(mid_frame_emb[0])

            if not chunk_embeddings:
                return None, None, None

            chunk_embeddings = np.array(chunk_embeddings)
            timing['embedding'] = time.time() - start

            # Save indices to cache, not raw frames
            self._save_to_cache(video_path, semantic_chunks_indices, semantic_descriptions, chunk_embeddings)
        
        # 6. Embed query and retrieve
        query_embedding = self.embed_text(query)
        
        start = time.time()
        top_k_indices, similarities = self.retrieve_top_k_chunks(
            chunk_embeddings, query_embedding, k=k
        )
        timing['retrieval'] = time.time() - start
        
        relevant_chunks = [semantic_chunks[i] for i in top_k_indices]
        
        print(f"Top similarities: {similarities}")
        print(f"Timing breakdown: {timing}")
        
        return relevant_chunks, similarities, timing
    
    def _get_cache_path(self, video_path):
        """Get cache file path for a video"""
        video_id = Path(video_path).stem
        return self.cache_dir / f"{video_id}.pkl"
    
    def _save_to_cache(self, video_path, semantic_chunks_indices, semantic_descriptions, chunk_embeddings):
        """Save only indices and embeddings, not raw frames"""
        cache_path = self._get_cache_path(video_path)
        
        cache_data = {
            'video_id': Path(video_path).stem,
            'semantic_chunks_indices': semantic_chunks_indices,  # Just frame numbers
            'semantic_descriptions': semantic_descriptions,
            'chunk_embeddings': chunk_embeddings.tolist(),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def _load_from_cache(self, video_path):
        """Load metadata and reconstruct chunks from video"""
        cache_path = self._get_cache_path(video_path)
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Reload frames from video using cached indices
        frames = extract_frames_at_fps(video_path, fps=1)
        
        semantic_chunks = []
        for indices in cache_data['semantic_chunks_indices']:
            chunk = [frames[i] for i in indices]
            semantic_chunks.append(chunk)
        
        chunk_embeddings = np.array(cache_data['chunk_embeddings'])
        
        return semantic_chunks, cache_data['semantic_descriptions'], chunk_embeddings