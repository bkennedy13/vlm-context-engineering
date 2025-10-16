"""Merge semantic chunks into high-level events based on description similarity."""

import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EventMerger:
    """Merge semantic chunks into events using description embeddings."""
    
    def __init__(self, similarity_threshold=0.75, model_name='all-MiniLM-L6-v2'):
        """
        Args:
            similarity_threshold: Minimum similarity to merge chunks into same event
            model_name: Sentence transformer model for text embeddings
        """
        self.threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")
    
    def merge_chunks_to_events(self, chunks, descriptions):
        """
        Merge semantic chunks into events.
        
        Args:
            chunks: List of chunk dicts from semantic_cache_level2
            descriptions: List of descriptions (one per merged chunk)
            
        Returns:
            List of event dicts with keys: [event_id, chunk_ids, description, frame_indices, ...]
        """
        if not chunks or not descriptions:
            return []
        
        # Embed descriptions
        embeddings = self.model.encode(descriptions, show_progress_bar=False)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Greedy clustering: assign each chunk to an event
        events = []
        chunk_to_event = {}  # chunk merged_id -> event_id
        
        MAX_EVENT_DURATION = 120  # 2 minutes in seconds
        HIGH_SIMILARITY_THRESHOLD = 0.9  # Allow longer events if very similar
        
        for i, (chunk, description) in enumerate(zip(chunks, descriptions)):
            merged_id = chunk['merged_id']
            
            # Check if should merge with existing event
            best_event_idx = None
            best_similarity = 0
            
            for event_idx, event in enumerate(events):
                # Compare to all chunks in this event
                event_chunk_indices = event['chunk_indices']
                event_sims = [similarities[i][j] for j in event_chunk_indices]
                avg_sim = np.mean(event_sims)
                
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_event_idx = event_idx
            
            # Merge if similarity above threshold AND duration constraint satisfied
            if best_event_idx is not None and best_similarity >= self.threshold:
                # Check duration constraint
                new_duration = events[best_event_idx]['duration_seconds'] + chunk['length_seconds']
                
                # Allow merge if:
                # 1. Duration stays under limit, OR
                # 2. Similarity is very high (>0.9)
                if new_duration <= MAX_EVENT_DURATION or best_similarity >= HIGH_SIMILARITY_THRESHOLD:
                    events[best_event_idx]['chunk_indices'].append(i)
                    events[best_event_idx]['chunk_ids'].append(merged_id)
                    events[best_event_idx]['frame_indices'].extend(chunk['frame_indices'])
                    events[best_event_idx]['start_time'] = min(
                        events[best_event_idx]['start_time'], 
                        chunk['start_time']
                    )
                    events[best_event_idx]['end_time'] = max(
                        events[best_event_idx]['end_time'], 
                        chunk['end_time']
                    )
                    events[best_event_idx]['duration_seconds'] += chunk['length_seconds']
                    chunk_to_event[merged_id] = best_event_idx
                else:
                    # Duration would exceed limit and similarity not high enough
                    # Create new event instead
                    event_id = len(events)
                    events.append({
                        'event_id': event_id,
                        'chunk_indices': [i],
                        'chunk_ids': [merged_id],
                        'frame_indices': chunk['frame_indices'].copy(),
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'duration_seconds': chunk['length_seconds'],
                        'description': description
                    })
                    chunk_to_event[merged_id] = event_id
            else:
                # Similarity below threshold, create new event
                event_id = len(events)
                events.append({
                    'event_id': event_id,
                    'chunk_indices': [i],
                    'chunk_ids': [merged_id],
                    'frame_indices': chunk['frame_indices'].copy(),
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'duration_seconds': chunk['length_seconds'],
                    'description': description
                })
                chunk_to_event[merged_id] = event_id
        
        # Generate merged descriptions for multi-chunk events
        for event in events:
            if len(event['chunk_indices']) > 1:
                # Concatenate descriptions from all chunks
                chunk_descs = [descriptions[i] for i in event['chunk_indices']]
                event['description'] = ' '.join(chunk_descs)
                # Remove duplicate frame indices
                event['frame_indices'] = sorted(list(set(event['frame_indices'])))
        
        return events
    
    def process_video(self, video_id, semantic_cache_path):
        """
        Process one video: load semantic chunks and merge into events.
        
        Args:
            video_id: Video identifier
            semantic_cache_path: Path to Level 2 semantic cache pickle file
            
        Returns:
            List of events for this video
        """
        # Load semantic chunks from pickle
        with open(semantic_cache_path, 'rb') as f:
            data = pickle.load(f)
        
        chunks = data.get('merged_chunks', [])
        descriptions = data.get('descriptions', [])
        
        if not chunks or not descriptions:
            print(f"Warning: No chunks/descriptions found for {video_id}")
            return []
        
        if len(chunks) != len(descriptions):
            print(f"Warning: Mismatch in {video_id}: {len(chunks)} chunks vs {len(descriptions)} descriptions")
            return []
        
        # Merge into events
        events = self.merge_chunks_to_events(chunks, descriptions)
        
        # Add video ID to each event
        for event in events:
            event['video_id'] = video_id
            # duration_seconds is already correctly calculated in merge_chunks_to_events!
            # Don't overwrite it here
        
        print(f"  {video_id}: {len(chunks)} chunks → {len(events)} events")
        return events


def merge_all_videos(semantic_dir='data/semantic_cache_level2', 
                    output_path='data/ekg/events.json'):
    """Process all videos and save merged events."""
    import json
    
    semantic_dir = Path(semantic_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merger = EventMerger(similarity_threshold=0.75)
    
    all_events = []
    cache_files = sorted(semantic_dir.glob('*.pkl'))
    
    print(f"Processing {len(cache_files)} videos...")
    for cache_file in cache_files:
        video_id = cache_file.stem
        events = merger.process_video(video_id, cache_file)
        all_events.extend(events)
    
    # Save all events (as JSON for easier inspection)
    with open(output_path, 'w') as f:
        json.dump({
            'total_events': len(all_events),
            'events': all_events
        }, f, indent=2)
    
    print(f"\nSaved {len(all_events)} events to {output_path}")
    return all_events


if __name__ == '__main__':
    merge_all_videos()