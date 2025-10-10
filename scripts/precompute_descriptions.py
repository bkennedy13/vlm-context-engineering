import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append('.')

from semantic.description_generator import DescriptionGenerator
from semantic.chunk_processor import create_5s_chunks, save_chunk_cache, load_chunk_cache
from shared.video_manager import VideoManager

def precompute_all_descriptions():
    """Generate and cache 5-second chunk descriptions for all evaluation videos"""
    
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
    desc_generator = DescriptionGenerator()
    
    # Track progress
    total_start = time.time()
    completed = 0
    failed = []
    skipped = 0
    
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
                # Get video path
                video_path = video_manager.get_video(youtube_id)
                if video_path is None:
                    print(f"\nFailed to download: {vid_id}")
                    failed.append(vid_id)
                    continue
                
                # Check if already cached
                cached = load_chunk_cache(video_path)
                if cached is not None:
                    print(f"\nSkipping {vid_id} (already cached with {cached['num_chunks']} chunks)")
                    skipped += 1
                    completed += 1
                    continue
                
                # Create 5-second chunks
                print(f"\nProcessing {vid_id}...")
                chunk_start = time.time()
                chunks, chunk_metadata = create_5s_chunks(video_path, fps=1)
                print(f"  Chunking took {time.time() - chunk_start:.2f}s")
                
                # Generate descriptions
                desc_start = time.time()
                descriptions = desc_generator.generate_batch_descriptions(chunks, verbose=True)
                desc_time = time.time() - desc_start
                print(f"  Description generation took {desc_time:.2f}s ({desc_time/len(chunks):.2f}s per chunk)")
                
                # Save to cache
                save_chunk_cache(video_path, chunk_metadata, descriptions)
                
                completed += 1
                elapsed = time.time() - total_start
                remaining = len(unique_videos) - completed - len(failed)
                rate = completed / elapsed if elapsed > 0 else 0
                eta = remaining / rate if rate > 0 else 0
                print(f"Progress: {completed}/{len(unique_videos)} | ETA: {eta/3600:.1f}h")
                
            except Exception as e:
                print(f"\nError processing {vid_id}: {e}")
                import traceback
                traceback.print_exc()
                failed.append(vid_id)
                continue
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print("DESCRIPTION GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Completed: {completed}/{len(unique_videos)}")
    print(f"Skipped (already cached): {skipped}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("Failed videos:", failed)
    
    print(f"\nCache location: data/semantic_cache/")
    print(f"Total cache files: {len(list(Path('data/semantic_cache').glob('*.pkl')))}")

if __name__ == "__main__":
    precompute_all_descriptions()