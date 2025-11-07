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
            List of event dicts
        """
        if not chunks or not descriptions:
            return []
        
        # Embed descriptions
        embeddings = self.model.encode(descriptions, show_progress_bar=False)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # SEQUENTIAL MERGING: only compare adjacent chunks
        events = []
        current_event = None
        
        MAX_EVENT_DURATION = 120  # 2 minutes
        HIGH_SIMILARITY_THRESHOLD = 0.9
        
        for i, (chunk, description) in enumerate(zip(chunks, descriptions)):
            merged_id = chunk['merged_id']
            
            # Start first event
            if current_event is None:
                current_event = {
                    'event_id': len(events),
                    'chunk_indices': [i],
                    'chunk_ids': [merged_id],
                    'frame_indices': chunk['frame_indices'].copy(),
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'duration_seconds': chunk['length_seconds'],
                    'descriptions': [description]  # Track all descriptions
                }
                continue
            
            # Check if should merge with current event
            # Compare to ALL chunks in current event (for semantic consistency)
            event_chunk_indices = current_event['chunk_indices']
            event_sims = [similarities[i][j] for j in event_chunk_indices]
            avg_sim = np.mean(event_sims)
            
            # Check duration constraint
            new_duration = current_event['duration_seconds'] + chunk['length_seconds']
            
            # Merge conditions:
            # 1. Similar to current event
            # 2. Duration within limit OR very high similarity
            if avg_sim >= self.threshold and \
            (new_duration <= MAX_EVENT_DURATION or avg_sim >= HIGH_SIMILARITY_THRESHOLD):
                # Merge into current event
                current_event['chunk_indices'].append(i)
                current_event['chunk_ids'].append(merged_id)
                current_event['frame_indices'].extend(chunk['frame_indices'])
                current_event['end_time'] = chunk['end_time']  # Update end (sequential)
                current_event['duration_seconds'] = current_event['end_time'] - current_event['start_time']
                current_event['descriptions'].append(description)
            else:
                # Finalize current event and start new one
                current_event['description'] = ' '.join(current_event['descriptions'])
                current_event['frame_indices'] = sorted(list(set(current_event['frame_indices'])))
                del current_event['descriptions']  # Clean up
                events.append(current_event)
                
                # Start new event
                current_event = {
                    'event_id': len(events),
                    'chunk_indices': [i],
                    'chunk_ids': [merged_id],
                    'frame_indices': chunk['frame_indices'].copy(),
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'duration_seconds': chunk['length_seconds'],
                    'descriptions': [description]
                }
        
        # Don't forget the last event!
        if current_event is not None:
            current_event['description'] = ' '.join(current_event['descriptions'])
            current_event['frame_indices'] = sorted(list(set(current_event['frame_indices'])))
            del current_event['descriptions']
            events.append(current_event)
        
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