"""
Tri-view event retrieval using pre-computed visual, semantic, and entity embeddings.
Uses Reciprocal Rank Fusion (RRF) to combine rankings from three modalities.
"""

import pickle
import numpy as np
from pathlib import Path
import torch
import open_clip
from sentence_transformers import SentenceTransformer


class TriViewEventRetriever:
    """Retrieve events using three modalities: visual, semantic, and entity."""
    
    def __init__(self, triview_cache_dir='data/triview_cache', 
                 ekg_cache_dir='data/ekg_cache'):
        """Initialize encoders for query embedding."""
        self.triview_cache_dir = Path(triview_cache_dir)
        self.ekg_cache_dir = Path(ekg_cache_dir)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai', device=self.device
        )
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_model.eval()
        
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve_events(self, youtube_id, question, k=10, 
                       frame_pooling='mean',
                       rrf_k=60,
                       use_visual=True,
                       use_semantic=True,
                       use_entity=True):
        """
        Retrieve events using tri-view RRF.
        
        Args:
            youtube_id: Video identifier
            question: Question text
            k: Number of events to retrieve
            frame_pooling: How to pool frame embeddings ('mean', 'max', 'first', 'last')
            rrf_k: RRF constant (default 60 is standard)
            use_visual: Whether to use visual modality
            use_semantic: Whether to use semantic modality
            use_entity: Whether to use entity modality
        
        Returns:
            events: List of retrieved event dicts (from EKG)
            scores: RRF scores for retrieved events
            modality_sims: Dict with per-modality similarities for analysis
        """
        triview_data = self._load_triview_cache(youtube_id)
        ekg_data = self._load_ekg_cache(youtube_id)
        
        if triview_data is None or ekg_data is None:
            return None, None, None
        
        events = ekg_data['events']
        triview_events = triview_data['events']
        
        print(f"Loaded {len(events)} events")
        
        # Embed query
        query_visual = self._embed_query_visual(question) if use_visual else None
        query_text = self._embed_query_text(question) if (use_semantic or use_entity) else None
        
        # Compute similarities for each modality
        modality_similarities = {}
        modality_rankings = {}
        
        if use_visual and query_visual is not None:
            visual_sims = self._compute_visual_similarity(
                triview_events, query_visual, frame_pooling
            )
            modality_similarities['visual'] = visual_sims
            modality_rankings['visual'] = np.argsort(visual_sims)[::-1]
        
        if use_semantic and query_text is not None:
            semantic_sims = self._compute_semantic_similarity(
                triview_events, query_text
            )
            modality_similarities['semantic'] = semantic_sims
            modality_rankings['semantic'] = np.argsort(semantic_sims)[::-1]
        
        if use_entity and query_text is not None:
            entity_sims = self._compute_entity_similarity(
                triview_events, query_text
            )
            modality_similarities['entity'] = entity_sims
            modality_rankings['entity'] = np.argsort(entity_sims)[::-1]
        
        # Reciprocal Rank Fusion
        rrf_scores = self._reciprocal_rank_fusion(
            modality_rankings, 
            n_events=len(events),
            k=rrf_k
        )
        
        top_indices = np.argsort(rrf_scores)[::-1][:k]
        retrieved_events = [events[i] for i in top_indices]
        retrieved_scores = rrf_scores[top_indices]
        
        modality_sims_topk = {}
        for modality, sims in modality_similarities.items():
            modality_sims_topk[modality] = sims[top_indices]
        
        print(f"Retrieved {len(retrieved_events)} events")
        print(f"Top RRF scores: {retrieved_scores[:3]}")
        
        return retrieved_events, retrieved_scores, modality_sims_topk
    
    def _compute_visual_similarity(self, triview_events, query_visual, pooling='mean'):
        """
        Compute visual similarity between query and events.
        
        Args:
            triview_events: List of event dicts with frame_embeddings
            query_visual: Query CLIP embedding (512,)
            pooling: How to pool frame embeddings
        
        Returns:
            Array of similarities (n_events,)
        """
        similarities = []
        
        for event in triview_events:
            frame_embs = event['frame_embeddings']  # (n_frames, 512)
            
            # Pool frame embeddings
            if pooling == 'mean':
                event_emb = np.mean(frame_embs, axis=0)
            elif pooling == 'max':
                frame_sims = np.dot(frame_embs, query_visual)
                similarities.append(np.max(frame_sims))
                continue
            elif pooling == 'first':
                event_emb = frame_embs[0]
            elif pooling == 'last':
                event_emb = frame_embs[-1]
            else:
                event_emb = np.mean(frame_embs, axis=0)
            
            event_emb = event_emb / np.linalg.norm(event_emb)
            
            sim = np.dot(event_emb, query_visual)
            similarities.append(sim)
        
        return np.array(similarities)
    
    def _compute_semantic_similarity(self, triview_events, query_text):
        """
        Compute semantic similarity between query and event descriptions.
        
        Returns:
            Array of similarities (n_events,)
        """
        similarities = []
        
        for event in triview_events:
            desc_emb = event['description_embedding']
            sim = np.dot(desc_emb, query_text)
            similarities.append(sim)
        
        return np.array(similarities)
    
    def _compute_entity_similarity(self, triview_events, query_text):
        """
        Compute entity similarity between query and event entity signatures.
        
        Returns:
            Array of similarities (n_events,)
        """
        similarities = []
        
        for event in triview_events:
            entity_emb = event['entity_embedding']
            
            if entity_emb is None:
                # No entities in this event - assign low similarity
                similarities.append(0.0)
            else:
                sim = np.dot(entity_emb, query_text)
                similarities.append(sim)
        
        return np.array(similarities)
    
    def _reciprocal_rank_fusion(self, modality_rankings, n_events, k=60):
        """
        Combine rankings from multiple modalities using RRF.
        
        RRF formula: score(event) = Σ_modality 1 / (k + rank_modality(event))
        
        Args:
            modality_rankings: Dict of {modality_name: ranking_array}
                ranking_array[i] = event_id with rank i (0 = best)
            n_events: Total number of events
            k: RRF constant (typically 60)
        
        Returns:
            Array of RRF scores (n_events,)
        """
        rrf_scores = np.zeros(n_events)
        
        for modality, ranking in modality_rankings.items():
            for rank, event_idx in enumerate(ranking):
                rrf_scores[event_idx] += 1.0 / (k + rank)
        
        return rrf_scores
    
    def _embed_query_visual(self, question):
        """Embed question using CLIP (for visual similarity)."""
        text_tokens = self.clip_tokenizer([question]).to(self.device)
        
        with torch.no_grad():
            query_emb = self.clip_model.encode_text(text_tokens)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        
        return query_emb.cpu().numpy()[0]
    
    def _embed_query_text(self, question):
        """Embed question using text encoder (for semantic and entity similarity)."""
        query_emb = self.text_encoder.encode([question], show_progress_bar=False)[0]
        return query_emb
    
    def _load_triview_cache(self, youtube_id):
        """Load tri-view cache for video."""
        cache_path = self.triview_cache_dir / f"{youtube_id}_triview.pkl"
        
        if not cache_path.exists():
            print(f"Tri-view cache not found: {cache_path}")
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_ekg_cache(self, youtube_id):
        """Load EKG cache for video (to get full event data)."""
        cache_path = self.ekg_cache_dir / f"{youtube_id}_ekg.pkl"
        
        if not cache_path.exists():
            print(f"EKG cache not found: {cache_path}")
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)