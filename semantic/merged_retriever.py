import numpy as np
import torch
import open_clip
from pathlib import Path
import pickle
import cv2
from PIL import Image

class MergedChunkRetriever:
    """Retrieve merged semantic chunks from Level 2 cache using CLIP similarity"""
    
    def __init__(self, model_name='ViT-B-32', pretrained='openai'):
        """Initialize CLIP model for retrieval"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP for retrieval on {self.device}...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.current_video_path = None  # Track current video for frame extraction
        
        print("MergedChunkRetriever initialized")
    
    def retrieve_chunks(self, video_path, query, k=10, cache_dir='data/semantic_cache_level2'):
        """
        Main retrieval pipeline: load cache → retrieve top-K chunks
        
        Args:
            video_path: Path to source video file
            query: Text query string
            k: Number of chunks to retrieve
            cache_dir: Directory containing merged chunk caches
        
        Returns:
            chunks: List of retrieved chunks (each chunk is list of frames)
            descriptions: List of merged descriptions for retrieved chunks
            similarities: CLIP similarity scores for retrieved chunks
        """
        print(f"Retrieving from: {video_path}")
        print(f"Query: {query}")
        
        self.current_video_path = video_path
        
        # 1. Load merged cache
        cache_data = self._load_cache(video_path, cache_dir)
        if cache_data is None:
            print(f"No merged cache found for {Path(video_path).stem}")
            return None, None, None
        
        # 2. Extract all frames from video once
        video_frames = self._extract_all_video_frames(video_path)
        
        # 3. Build chunks from frame indices in cache
        chunks = self._build_chunks_from_indices(video_frames, cache_data['merged_chunks'])
        descriptions = cache_data['descriptions']
        
        print(f"Loaded {len(chunks)} merged chunks from cache")
        
        # 4. Embed middle frame of each chunk with CLIP
        chunk_embeddings = self._embed_chunks(chunks)
        
        # 5. Embed query with CLIP
        query_embedding = self._embed_text(query)
        
        # 6. Retrieve top-K most similar chunks
        top_k_indices, similarities = self._retrieve_top_k(
            chunk_embeddings, query_embedding, k
        )
        
        # 7. Return retrieved chunks, descriptions, and similarities
        retrieved_chunks = [chunks[i] for i in top_k_indices]
        retrieved_descriptions = [descriptions[i] for i in top_k_indices]
        
        print(f"Retrieved {len(retrieved_chunks)} chunks")
        print(f"Top similarities: {similarities}")
        
        return retrieved_chunks, retrieved_descriptions, similarities
    
    def _load_cache(self, video_path, cache_dir):
        """Load merged chunk cache for a video"""
        cache_dir = Path(cache_dir)
        video_id = Path(video_path).stem
        cache_path = cache_dir / f"{video_id}.pkl"
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def _extract_all_video_frames(self, video_path):
        """
        Extract all frames from video at 1 FPS (matching base cache)
        
        This replicates the extraction logic from precompute to ensure
        frame indices in merged cache align with actual video frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / 1)  # 1 FPS
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _build_chunks_from_indices(self, video_frames, merged_chunks_metadata):
        """
        Build frame chunks using frame indices from merged cache
        
        The cache stores original video frame indices (e.g., [0, 24, 48, 72, 96])
        We need to map these to our 1 FPS extracted frame array indices (e.g., [0, 1, 2, 3, 4])
        """
        chunks = []
        
        # Get video FPS to calculate the mapping
        cap = cv2.VideoCapture(str(self.current_video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        frame_interval = int(original_fps / 1)  # Frames between each 1 FPS sample
        
        for chunk_meta in merged_chunks_metadata:
            original_frame_indices = chunk_meta['frame_indices']
            
            # Convert original indices to 1 FPS array positions
            fps_indices = [idx // frame_interval for idx in original_frame_indices]
            
            # Build chunk, skipping any indices beyond array bounds
            chunk_frames = []
            for i in fps_indices:
                if i < len(video_frames):
                    chunk_frames.append(video_frames[i])
            
            chunks.append(chunk_frames)
        
        return chunks
    
    def _embed_chunks(self, chunks):
        """
        Embed each chunk using CLIP (average of 5 evenly-spaced frames)
        
        Args:
            chunks: List of chunks (each chunk is list of frames)
        
        Returns:
            Array of embeddings (shape: [num_chunks, embedding_dim])
        """
        embeddings = []
        
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            
            # Sample up to 5 evenly-spaced frames
            if len(chunk) <= 5:
                sampled_frames = chunk
            else:
                indices = np.linspace(0, len(chunk)-1, 5, dtype=int)
                sampled_frames = [chunk[i] for i in indices]
            
            # Embed each sampled frame
            frame_embeddings = []
            for frame in sampled_frames:
                # Convert to PIL
                if isinstance(frame, np.ndarray):
                    pil_frame = Image.fromarray(frame)
                else:
                    pil_frame = frame
                
                processed = self.preprocess(pil_frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.model.encode_image(processed)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                
                frame_embeddings.append(embedding.cpu().numpy()[0])
            
            # Average the frame embeddings
            chunk_embedding = np.mean(frame_embeddings, axis=0)
            
            # Re-normalize after averaging
            chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding)
            
            embeddings.append(chunk_embedding)
        
        return np.array(embeddings)
    
    def _embed_text(self, text):
        """Embed text query using CLIP"""
        text_tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        return text_embedding.cpu().numpy()
    
    def _retrieve_top_k(self, chunk_embeddings, query_embedding, k):
        """
        Retrieve top-K most similar chunks via cosine similarity
        
        Args:
            chunk_embeddings: Array of chunk embeddings
            query_embedding: Query embedding
            k: Number of chunks to retrieve
        
        Returns:
            top_k_indices: Indices of top-K chunks
            similarities: Similarity scores for top-K chunks
        """
        # Compute cosine similarities
        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
        
        # Get top-K indices (highest similarity first)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_k_indices, similarities[top_k_indices]