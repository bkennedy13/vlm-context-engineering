"""
Simple event-based retrieval using description similarity.
"""

import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EventRetriever:
    """Retrieve events from EKG based on description similarity."""
    
    def __init__(self, ekg_cache_dir='data/ekg_cache'):
        """Initialize retriever with text embedding model."""
        self.ekg_cache_dir = Path(ekg_cache_dir)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("EventRetriever initialized")
    
    def retrieve_events(self, youtube_id, question, k=5, gap_threshold=0.08, min_events=2):
        """
        Retrieve events with adaptive selection based on similarity drop-off.
        
        Args:
            youtube_id: Video identifier
            question: Question text
            k: Maximum number of events to retrieve
            gap_threshold: Minimum similarity gap to stop retrieval (default: 0.08)
            min_events: Minimum number of events to retrieve (default: 2)
        
        Returns:
            events: List of retrieved event dicts
            similarities: Similarity scores for events
        """
        # Load EKG
        ekg = self._load_ekg(youtube_id)
        if ekg is None:
            return None, None
        
        events = ekg['events']
        print(f"Loaded EKG: {len(events)} events")
        
        # Embed descriptions and question
        descriptions = [e['description'] for e in events]
        desc_embeddings = self.encoder.encode(descriptions)
        question_embedding = self.encoder.encode([question])[0]
        
        # Compute similarities
        similarities = cosine_similarity([question_embedding], desc_embeddings)[0]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_sims = similarities[sorted_indices]
        
        # Adaptive selection: stop when similarity drops significantly
        n_selected = min_events  # Start with minimum
        
        for i in range(min_events - 1, min(k, len(sorted_sims)) - 1):
            gap = sorted_sims[i] - sorted_sims[i + 1]
            if gap > gap_threshold:
                n_selected = i + 1
                break
        else:
            # No significant gap found, use all k events
            n_selected = min(k, len(sorted_sims))
        
        # Get top-n events
        top_indices = sorted_indices[:n_selected]
        retrieved_events = [events[i] for i in top_indices]
        scores = sorted_sims[:n_selected]
        
        print(f"Retrieved {len(retrieved_events)} events (adaptive selection)")
        print(f"Similarities: {scores}")
        
        return retrieved_events, scores
    
    def _load_ekg(self, youtube_id):
        """Load cached EKG for video."""
        ekg_path = self.ekg_cache_dir / f"{youtube_id}_ekg.pkl"
        if not ekg_path.exists():
            print(f"EKG not found: {ekg_path}")
            return None
        
        with open(ekg_path, 'rb') as f:
            return pickle.load(f)