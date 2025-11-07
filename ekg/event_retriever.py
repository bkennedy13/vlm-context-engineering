"""
Hybrid event retrieval using both frame similarity (CLIP) and description similarity.
CLIP embeddings computed on-the-fly from video frames.
"""

import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import open_clip
import torch
import cv2
from PIL import Image


class EventRetriever:
    """Retrieve events using hybrid CLIP + text similarity."""
    
    def __init__(self, ekg_cache_dir='data/ekg_cache'):
        """Initialize retriever with text and visual encoders."""
        self.ekg_cache_dir = Path(ekg_cache_dir)
        
        # Text encoder
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # CLIP encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai', device=self.device
        )
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_model.eval()
        
        print("EventRetriever initialized (hybrid CLIP + text)")
    
    def retrieve_events(self, youtube_id, question, video_path, k=5,
                       clip_weight=0.6, text_weight=0.4,
                       gap_threshold=0.10, min_events=1,
                       frames_per_event=5):
        """
        Retrieve events using hybrid similarity.
        
        Args:
            youtube_id: Video identifier
            question: Question text
            video_path: Path to video file (for frame extraction)
            k: Maximum number of events to retrieve
            clip_weight: Weight for visual similarity (default: 0.6)
            text_weight: Weight for text similarity (default: 0.4)
            gap_threshold: Similarity gap for adaptive selection
            min_events: Minimum events to retrieve
            frames_per_event: Number of frames to sample per event for CLIP
        
        Returns:
            events: List of retrieved event dicts
            similarities: Combined similarity scores
        """
        # Load EKG
        ekg = self._load_ekg(youtube_id)
        if ekg is None:
            return None, None
        
        events = ekg['events']
        print(f"Loaded EKG: {len(events)} events")
        
        # Extract video frames at 1 FPS
        video_frames = self._extract_video_frames(video_path)
        
        # Get video FPS for frame index mapping
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / 1)
        cap.release()
        
        # 1. CLIP similarity - compute event embeddings on-the-fly
        event_clip_embeddings = []
        for event in events:
            clip_emb = self._embed_event_frames(
                event, video_frames, frame_interval, frames_per_event
            )
            event_clip_embeddings.append(clip_emb)
        
        event_clip_embeddings = np.array(event_clip_embeddings)
        
        # Encode question with CLIP
        text_tokens = self.clip_tokenizer([question]).to(self.device)
        with torch.no_grad():
            query_clip = self.clip_model.encode_text(text_tokens)
            query_clip = query_clip / query_clip.norm(dim=-1, keepdim=True)
        query_clip = query_clip.cpu().numpy()
        
        clip_sims = np.dot(event_clip_embeddings, query_clip.T).flatten()
        
        # 2. Text similarity
        descriptions = [e['description'] for e in events]
        desc_embeddings = self.text_encoder.encode(descriptions)
        query_text = self.text_encoder.encode([question])[0]
        text_sims = cosine_similarity([query_text], desc_embeddings)[0]
        
        # 3. Combine scores
        combined_sims = clip_weight * clip_sims + text_weight * text_sims
        
        # print(f"  CLIP sims: {clip_sims[:3]} ...")
        # print(f"  Text sims: {text_sims[:3]} ...")
        # print(f"  Combined: {combined_sims[:3]} ...")
        
        # 4. Adaptive selection (vs top)
        sorted_indices = np.argsort(combined_sims)[::-1]
        sorted_sims = combined_sims[sorted_indices]
        
        top_similarity = sorted_sims[0]
        n_selected = min_events
        
        for i in range(min_events, min(k, len(sorted_sims))):
            gap_from_top = top_similarity - sorted_sims[i]
            if gap_from_top > gap_threshold:
                n_selected = i
                break
        else:
            n_selected = min(k, len(sorted_sims))
        
        # Get top-n events
        top_indices = sorted_indices[:n_selected]
        retrieved_events = [events[i] for i in top_indices]
        scores = sorted_sims[:n_selected]
        
        print(f"Retrieved {n_selected} events")
        # print(f"Similarities: {scores}")
        
        return retrieved_events, scores
    
    def _extract_video_frames(self, video_path):
        """Extract all frames at 1 FPS."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / 1)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _embed_event_frames(self, event, video_frames, frame_interval, n_frames=5):
        """
        Embed an event by sampling and averaging frames.
        
        Args:
            event: Event dict with frame_indices
            video_frames: All video frames at 1 FPS
            frame_interval: Frames between each 1 FPS sample
            n_frames: Number of frames to sample from event
        
        Returns:
            Averaged CLIP embedding for event
        """
        # Get frame indices for this event
        frame_indices = event['frame_indices']
        fps_indices = [idx // frame_interval for idx in frame_indices]
        
        # Sample n_frames evenly from event
        if len(fps_indices) <= n_frames:
            sampled_indices = fps_indices
        else:
            step = len(fps_indices) / n_frames
            sampled_indices = [fps_indices[int(i * step)] for i in range(n_frames)]
        
        # Embed sampled frames
        frame_embeddings = []
        for idx in sampled_indices:
            if idx < len(video_frames):
                pil_frame = Image.fromarray(video_frames[idx])
                processed = self.clip_preprocess(pil_frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    emb = self.clip_model.encode_image(processed)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                
                frame_embeddings.append(emb.cpu().numpy()[0])
        
        if not frame_embeddings:
            # Fallback: zero embedding
            return np.zeros(512)
        
        # Average and normalize
        avg_embedding = np.mean(frame_embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding
    
    def _load_ekg(self, youtube_id):
        """Load cached EKG for video."""
        ekg_path = self.ekg_cache_dir / f"{youtube_id}_ekg.pkl"
        if not ekg_path.exists():
            print(f"EKG not found: {ekg_path}")
            return None
        
        with open(ekg_path, 'rb') as f:
            return pickle.load(f)