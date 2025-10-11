# scripts/merge_semantic_chunks.py

import sys
sys.path.append('.')

import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from bert_score import score as bertscore_fn
import time

def load_base_cache(video_path, cache_dir="data/semantic_cache"):
    """Load unmerged chunk cache"""
    cache_dir = Path(cache_dir)
    video_id = Path(video_path).stem
    cache_path = cache_dir / f"{video_id}.pkl"
    
    if not cache_path.exists():
        return None
    
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def compute_adjacent_bertscores(descriptions, batch_size=50):
    """
    Compute BERTScore F1 between adjacent description pairs
    
    Args:
        descriptions: List of description strings
        batch_size: Process in batches to avoid memory issues
    
    Returns:
        Array of F1 scores (length = len(descriptions) - 1)
    """
    if len(descriptions) < 2:
        return np.array([])
    
    candidates = descriptions[1:]
    references = descriptions[:-1]
    
    all_f1_scores = []
    
    for i in range(0, len(candidates), batch_size):
        batch_cand = candidates[i:i+batch_size]
        batch_ref = references[i:i+batch_size]
        
        P, R, F1 = bertscore_fn(
            batch_cand,
            batch_ref,
            lang='en',
            model_type='microsoft/deberta-xlarge-mnli',
            device='cuda',
            verbose=False
        )
        
        all_f1_scores.extend(F1.numpy().tolist())
    
    return np.array(all_f1_scores)

def merge_chunks_with_cap(chunk_metadata, descriptions, bertscores, threshold, max_length_seconds):
    """
    Merge chunks based on BERTScore threshold with length cap
    
    Args:
        chunk_metadata: List of chunk metadata dicts from base cache
        descriptions: List of description strings
        bertscores: Array of BERTScore F1 scores between adjacent chunks
        threshold: Minimum BERTScore to merge
        max_length_seconds: Maximum allowed chunk length
    
    Returns:
        merged_metadata: List of merged chunk metadata
        merged_descriptions: List of merged descriptions
    """
    if len(chunk_metadata) <= 1:
        return chunk_metadata, descriptions
    
    merged_metadata = []
    merged_descriptions = []
    
    # Start with first chunk
    current_indices = [0]
    current_frame_indices = chunk_metadata[0]['frame_indices'].copy()
    current_desc = descriptions[0]
    current_length = 5  # Each base chunk is 5 seconds
    
    merge_count = 0
    
    for i in range(len(bertscores)):
        should_merge = bertscores[i] > threshold
        
        # Apply length cap
        if should_merge and (current_length + 5 > max_length_seconds):
            should_merge = False
        
        if should_merge:
            # Merge: extend current chunk
            current_indices.append(i + 1)
            current_frame_indices.extend(chunk_metadata[i + 1]['frame_indices'])
            current_desc = f"{current_desc} {descriptions[i + 1]}"
            current_length += 5
            merge_count += 1
        else:
            # Save current chunk and start new
            merged_metadata.append({
                'merged_id': len(merged_metadata),
                'source_chunk_ids': current_indices,
                'start_time': chunk_metadata[current_indices[0]]['start_time'],
                'end_time': chunk_metadata[current_indices[-1]]['end_time'],
                'frame_indices': current_frame_indices,
                'length_seconds': current_length
            })
            merged_descriptions.append(current_desc)
            
            # Start new chunk
            current_indices = [i + 1]
            current_frame_indices = chunk_metadata[i + 1]['frame_indices'].copy()
            current_desc = descriptions[i + 1]
            current_length = 5
    
    # Add last chunk
    merged_metadata.append({
        'merged_id': len(merged_metadata),
        'source_chunk_ids': current_indices,
        'start_time': chunk_metadata[current_indices[0]]['start_time'],
        'end_time': chunk_metadata[current_indices[-1]]['end_time'],
        'frame_indices': current_frame_indices,
        'length_seconds': current_length
    })
    merged_descriptions.append(current_desc)
    
    return merged_metadata, merged_descriptions, merge_count

def save_merged_cache(base_cache_path, merged_metadata, merged_descriptions, 
                      threshold, max_length_seconds, merge_stats,
                      output_dir="data/semantic_cache_level2"):
    """Save merged chunks to level2 cache"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_id = base_cache_path.stem
    output_path = output_dir / f"{video_id}.pkl"
    
    cache_data = {
        'video_id': video_id,
        'source_cache': str(base_cache_path),
        'merge_threshold': threshold,
        'max_length_seconds': max_length_seconds,
        'merged_chunks': merged_metadata,
        'descriptions': merged_descriptions,
        'num_merged_chunks': len(merged_metadata),
        'merge_stats': merge_stats,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return output_path

def merge_all_videos():
    """Main function: merge all videos from base cache"""
    
    # Load eval subset to get video durations
    print("Loading evaluation subset...")
    with open('data/eval_subset.json', 'r') as f:
        eval_samples = json.load(f)
    
    video_durations = {}
    for sample in eval_samples:
        video_durations[sample['youtube_id']] = sample['duration']
    
    print(f"Loaded {len(video_durations)} video duration mappings")
    
    # Settings based on notebook exploration
    threshold = 0.70
    length_caps = {
        'short': 30,   # 30 seconds for short videos
        'medium': 30,  # 30 seconds for medium videos
        'long': 60     # 60 seconds for long videos
    }
    
    print(f"\nMerge settings:")
    print(f"  BERTScore threshold: {threshold}")
    print(f"  Length caps: {length_caps}")
    
    # Get all base cache files
    base_cache_dir = Path('data/semantic_cache')
    cache_files = list(base_cache_dir.glob('*.pkl'))
    print(f"\nFound {len(cache_files)} videos in base cache")
    
    # Track progress
    total_start = time.time()
    completed = 0
    failed = []
    skipped = []
    
    # Group by duration for reporting
    by_duration = {'short': 0, 'medium': 0, 'long': 0}
    
    for cache_file in tqdm(cache_files, desc="Merging chunks"):
        try:
            youtube_id = cache_file.stem
            
            # Get duration for this video
            if youtube_id not in video_durations:
                print(f"\nSkipping {youtube_id}: not in eval subset")
                skipped.append(youtube_id)
                continue
            
            duration = video_durations[youtube_id]
            max_length = length_caps[duration]
            
            # Load base cache
            with open(cache_file, 'rb') as f:
                base_cache = pickle.load(f)
            
            chunk_metadata = base_cache['chunk_metadata']
            descriptions = base_cache['descriptions']
            
            # Compute BERTScores
            bertscores = compute_adjacent_bertscores(descriptions)
            
            # Merge chunks
            merged_metadata, merged_descriptions, merge_count = merge_chunks_with_cap(
                chunk_metadata, 
                descriptions, 
                bertscores, 
                threshold, 
                max_length
            )
            
            # Compute stats
            original_count = len(chunk_metadata)
            merged_count = len(merged_metadata)
            reduction_pct = (1 - merged_count / original_count) * 100
            
            chunk_lengths = [meta['length_seconds'] for meta in merged_metadata]
            
            merge_stats = {
                'original_chunks': original_count,
                'merged_chunks': merged_count,
                'merges_performed': merge_count,
                'reduction_percent': reduction_pct,
                'mean_length': float(np.mean(chunk_lengths)),
                'median_length': float(np.median(chunk_lengths)),
                'max_length': float(np.max(chunk_lengths)),
                'min_length': float(np.min(chunk_lengths))
            }
            
            # Save merged cache
            output_path = save_merged_cache(
                cache_file, 
                merged_metadata, 
                merged_descriptions,
                threshold,
                max_length,
                merge_stats
            )
            
            completed += 1
            by_duration[duration] += 1
            
        except Exception as e:
            print(f"\nError processing {cache_file.stem}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(cache_file.stem)
            continue
    
    # Summary
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print("MERGE COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Completed: {completed}/{len(cache_files)}")
    print(f"  Short: {by_duration['short']}")
    print(f"  Medium: {by_duration['medium']}")
    print(f"  Long: {by_duration['long']}")
    
    if skipped:
        print(f"Skipped (not in eval subset): {len(skipped)}")
    
    if failed:
        print(f"Failed: {len(failed)}")
        print("Failed videos:", failed)
    
    print(f"\nMerged cache location: data/semantic_cache_level2/")
    print(f"Total merged cache files: {len(list(Path('data/semantic_cache_level2').glob('*.pkl')))}")
    
    # Quick stats across all videos
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")
    
    all_stats = {'short': [], 'medium': [], 'long': []}
    
    for cache_file in Path('data/semantic_cache_level2').glob('*.pkl'):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        youtube_id = cache_file.stem
        if youtube_id in video_durations:
            duration = video_durations[youtube_id]
            all_stats[duration].append(data['merge_stats'])
    
    for duration in ['short', 'medium', 'long']:
        if all_stats[duration]:
            stats = all_stats[duration]
            print(f"\n{duration.upper()} videos ({len(stats)} total):")
            print(f"  Avg reduction: {np.mean([s['reduction_percent'] for s in stats]):.1f}%")
            print(f"  Avg chunks: {np.mean([s['original_chunks'] for s in stats]):.0f} → {np.mean([s['merged_chunks'] for s in stats]):.0f}")
            print(f"  Avg chunk length: {np.mean([s['mean_length'] for s in stats]):.1f}s")
            print(f"  Max chunk length: {np.max([s['max_length'] for s in stats]):.1f}s")

if __name__ == "__main__":
    merge_all_videos()