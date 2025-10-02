import numpy as np
import torch
import open_clip
from pathlib import Path
import json
from PIL import Image
from .video_utils import extract_frames_at_fps, create_chunks

class VideoRAGBaseline:
    def __init__(self, model_name='ViT-B-32', pretrained='openai'):
        """Initialize CLIP model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
    def embed_frames(self, frames):
        """Embed a list of frames using CLIP"""
        if not frames:
            return np.array([])
            
        # Convert numpy arrays to PIL Images
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # Convert from numpy array to PIL Image
                pil_frame = Image.fromarray(frame)
                pil_frames.append(pil_frame)
            else:
                pil_frames.append(frame)
        
        # Preprocess frames
        processed_frames = torch.stack([
            self.preprocess(frame) for frame in pil_frames
        ]).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_image(processed_frames)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
        return embeddings.cpu().numpy()
    
    # Rest of the methods stay the same...
    def embed_text(self, text):
        """Embed text query using CLIP"""
        text_tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            
        return text_embedding.cpu().numpy()
    
    def retrieve_top_k_chunks(self, chunk_embeddings, query_embedding, k=5):
        """Retrieve top-K most similar chunks to query"""
        # Compute cosine similarities
        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
        
        # Get top-K indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_k_indices, similarities[top_k_indices]
    
    def process_video(self, video_path, query, k=10):
        """Main pipeline: video -> frames -> chunks -> embeddings -> retrieval"""
        print(f"Processing video: {video_path}")
        print(f"Query: {query}")
        
        # Extract frames
        frames = extract_frames_at_fps(video_path, fps=1)
        if not frames:
            return None, None
            
        # Create chunks
        chunks = create_chunks(frames, chunk_size=3)
        print(f"Created {len(chunks)} chunks")
        
        # Embed chunks - only middle frame
        chunk_embeddings = []
        for chunk in chunks:
            if len(chunk) > 0:
                mid_idx = len(chunk) // 2
                mid_frame_emb = self.embed_frames([chunk[mid_idx]])  # Single frame
                if len(mid_frame_emb) > 0:
                    chunk_embeddings.append(mid_frame_emb[0])  # Take first (only) embedding
        
        if not chunk_embeddings:
            return None, None
            
        chunk_embeddings = np.array(chunk_embeddings)
        
        # Embed query
        query_embedding = self.embed_text(query)
        
        # Retrieve top-K
        top_k_indices, similarities = self.retrieve_top_k_chunks(
            chunk_embeddings, query_embedding, k=k
        )
        
        # Return relevant chunks and their similarities
        relevant_chunks = [chunks[i] for i in top_k_indices]
        
        print(f"Top similarities: {similarities}")
        return relevant_chunks, similarities