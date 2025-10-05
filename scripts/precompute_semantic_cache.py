import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append('.')

from semantic.semantic_rag import SemanticRAG
from shared.video_manager import VideoManager

def precompute_all_caches():
    """Generate and cache semantic chunks for all evaluation videos"""
    
    # Load eval subset
    print("Loading evaluation subset...")
    with open('data/eval_subset.json', 'r') as f:
        eval_samples = json.load(f)
    
    # Get unique videos
    unique_videos = {}
    for sample in eval_samples:
        vid_id = sample['video_id']
        if vid_id not in unique_videos:
            unique_videos[vid_id] = {
                'youtube_id': sample['youtube_id'],
                'duration': sample['duration']
            }
    
    print(f"\nFound {len(unique_videos)} unique videos")
    
    # Group by duration for progress tracking
    by_duration = {'short': [], 'medium': [], 'long': []}
    for vid_id, info in unique_videos.items():
        by_duration[info['duration']].append((vid_id, info['youtube_id']))
    
    print(f"  Short: {len(by_duration['short'])} videos")
    print(f"  Medium: {len(by_duration['medium'])} videos")
    print(f"  Long: {len(by_duration['long'])} videos")
    
    # Initialize
    video_manager = VideoManager()
    semantic_rag = SemanticRAG(similarity_threshold=0.65)
    
    # Track progress
    total_start = time.time()
    completed = 0
    failed = []
    
    # Process each duration group
    for duration in ['short', 'medium', 'long']:
        videos = by_duration[duration]
        if not videos:
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {len(videos)} {duration.upper()} videos")
        print(f"{'='*80}")
        
        for vid_id, youtube_id in tqdm(videos, desc=f"{duration.capitalize()} videos"):
            try:
                # Check if already cached
                video_path = video_manager.get_video(youtube_id)
                if video_path is None:
                    print(f"\nFailed to download: {vid_id}")
                    failed.append(vid_id)
                    continue
                
                cache_path = semantic_rag._get_cache_path(video_path)
                if cache_path.exists():
                    print(f"\nSkipping {vid_id} (already cached)")
                    completed += 1
                    continue
                
                # Process video (will generate and cache)
                print(f"\nProcessing {vid_id}...")
                chunks, sims, timing = semantic_rag.process_video(
                    video_path,
                    query="dummy",  # Dummy query just to build cache
                    k=10
                )
                
                if chunks is None:
                    print(f"Failed to process {vid_id}")
                    failed.append(vid_id)
                else:
                    completed += 1
                    elapsed = time.time() - total_start
                    remaining = len(unique_videos) - completed - len(failed)
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{len(unique_videos)} | ETA: {eta/3600:.1f}h")
                
            except Exception as e:
                print(f"\nError processing {vid_id}: {e}")
                failed.append(vid_id)
                continue
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print("CACHE GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Completed: {completed}/{len(unique_videos)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("Failed videos:", failed)
    
    print(f"\nCache location: data/semantic_cache/")
    print(f"Total cache files: {len(list(Path('data/semantic_cache').glob('*.pkl')))}")

if __name__ == "__main__":
    precompute_all_caches()