"""
Build tri-view embeddings cache for Level 4 event retrieval.
Pre-computes visual (CLIP frames), semantic (description), and entity embeddings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import pickle
import time
from tqdm import tqdm
import torch
import numpy as np
import cv2
from PIL import Image

import open_clip
from sentence_transformers import SentenceTransformer


class TriViewCacheBuilder:
    """Build tri-view embeddings for efficient multi-modal event retrieval."""
    
    def __init__(self):
        """Initialize CLIP and text embedding models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai', device=self.device
        )
        self.clip_model.eval()
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def build_triview_cache(self, youtube_id, ekg_cache_dir, video_path):
        """
        Build tri-view embeddings for one video.
        
        Args:
            youtube_id: Video identifier
            ekg_cache_dir: Path to EKG cache directory
            video_path: Path to video file (for frame extraction)
        
        Returns:
            triview_data: Dict with embeddings for each event
            preprocessing_time: Time taken to build cache (seconds)
        """
        start_time = time.time()
        
        ekg_path = Path(ekg_cache_dir) / f"{youtube_id}_ekg.pkl"
        if not ekg_path.exists():
            print(f"No EKG found for {youtube_id}")
            return None, 0.0
        
        with open(ekg_path, 'rb') as f:
            ekg_data = pickle.load(f)
        
        events = ekg_data['events']
        
        video_frames = self._extract_video_frames(video_path)
        

        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / 1)
        cap.release()
        
        triview_events = []
        
        for event in events:
            event_embeddings = {
                'event_id': event['event_id'],
                'frame_embeddings': self._embed_event_frames(
                    event, video_frames, frame_interval, n_frames=10
                ),
                'description_embedding': self._embed_description(
                    event['description']
                ),
                'entity_embedding': self._embed_entities(
                    event.get('entities', {})
                )
            }
            triview_events.append(event_embeddings)
        
        preprocessing_time = time.time() - start_time
        
        # Package cache data
        triview_data = {
            'youtube_id': youtube_id,
            'ekg_cache_path': str(ekg_path),
            'num_events': len(events),
            'events': triview_events,
            'preprocessing_time': preprocessing_time,
            'config': {
                'clip_model': 'ViT-B-32',
                'text_model': 'all-MiniLM-L6-v2',
                'frames_per_event': 10,
                'embedding_dim': 512
            },
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return triview_data, preprocessing_time
    
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
    
    def _embed_event_frames(self, event, video_frames, frame_interval, n_frames=10):
        """
        Embed N sampled frames from an event using CLIP.
        
        Returns individual embeddings (not averaged) for flexibility at retrieval time.
        Shape: (n_frames, 512)
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
        
        return np.array(frame_embeddings, dtype=np.float32)
    
    def _embed_description(self, description):
        """
        Embed event description using sentence transformer.
        
        Returns: (512,) array
        """
        embedding = self.text_model.encode([description], show_progress_bar=False)[0]
        return embedding.astype(np.float32)
    
    def _embed_entities(self, entities_dict):
        """
        Create entity signature by averaging embeddings of all entities in event.
        
        Args:
            entities_dict: Dict with keys like 'objects', 'actions', 'locations', 'attributes'
        
        Returns: (512,) array, or None if no entities
        """
        # Collect all entity names
        entity_names = []
        for entity_type, entity_list in entities_dict.items():
            entity_names.extend(entity_list)
        
        # Embed all entities
        entity_embeddings = self.text_model.encode(
            entity_names, 
            show_progress_bar=False
        )
        
        # Average to create entity signature
        entity_signature = np.mean(entity_embeddings, axis=0)
        entity_signature = entity_signature / np.linalg.norm(entity_signature)
        
        return entity_signature.astype(np.float32)


def build_all_triview_caches(eval_subset_path='data/eval_subset.json',
                              ekg_cache_dir='data/ekg_cache',
                              triview_output_dir='data/triview_cache',
                              data_dir='data'):
    """Build tri-view caches for all videos in eval subset."""
    triview_output_dir = Path(triview_output_dir)
    triview_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load eval subset to get video IDs and paths
    with open(eval_subset_path, 'r') as f:
        eval_samples = json.load(f)
    
    unique_videos = {}
    for sample in eval_samples:
        youtube_id = sample['youtube_id']
        if youtube_id not in unique_videos:
            unique_videos[youtube_id] = sample
    
    video_cache_path = Path(data_dir) / 'video_cache.json'
    with open(video_cache_path, 'r') as f:
        video_cache = json.load(f)
    r
    builder = TriViewCacheBuilder()
    
    results = []
    failed = []
    total_preprocessing_time = 0
    
    for i, youtube_id in enumerate(tqdm(sorted(unique_videos.keys()), desc="Building tri-view caches")):
        try:
            # Check if already exists
            output_file = triview_output_dir / f"{youtube_id}_triview.pkl"
            if output_file.exists():
                print(f"\n[{i+1}/{len(unique_videos)}] {youtube_id}: Already cached, skipping")
                with open(output_file, 'rb') as f:
                    cached = pickle.load(f)
                    total_preprocessing_time += cached['preprocessing_time']
                continue
            
            if youtube_id not in video_cache:
                failed.append(youtube_id)
                continue
            
            video_path = video_cache[youtube_id]['path']
            if not Path(video_path).exists():
                failed.append(youtube_id)
                continue
            

            triview_data, preprocessing_time = builder.build_triview_cache(
                youtube_id, ekg_cache_dir, video_path
            )
            
            if triview_data:
                with open(output_file, 'wb') as f:
                    pickle.dump(triview_data, f)
                
                results.append({
                    'youtube_id': youtube_id,
                    'num_events': triview_data['num_events'],
                    'preprocessing_time': preprocessing_time
                })
                total_preprocessing_time += preprocessing_time
            else:
                failed.append(youtube_id)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append(youtube_id)
            continue
    
    if failed:
        print(f"\nFailed videos: {failed}")
    
    summary = {
        'total_videos': len(unique_videos),
        'successful': len(results),
        'failed': len(failed),
        'failed_videos': failed,
        'total_preprocessing_time': total_preprocessing_time,
        'avg_preprocessing_time': total_preprocessing_time / len(results) if results else 0,
        'video_stats': results
    }
    
    summary_file = triview_output_dir / 'build_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build tri-view cache for Level 4 retrieval')
    parser.add_argument('--eval-subset', type=str, default='data/eval_subset.json',
                       help='Path to eval subset JSON')
    parser.add_argument('--ekg-cache', type=str, default='data/ekg_cache',
                       help='Directory containing EKG cache files')
    parser.add_argument('--output', type=str, default='data/triview_cache',
                       help='Output directory for tri-view cache files')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory containing video cache')
    
    args = parser.parse_args()
    
    build_all_triview_caches(
        eval_subset_path=args.eval_subset,
        ekg_cache_dir=args.ekg_cache,
        triview_output_dir=args.output,
        data_dir=args.data_dir
    )