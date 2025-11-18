"""
Measure preprocessing times for Level 4 (tri-view) to get accurate end-to-end latency.
This measures the time to build tri-view embeddings offline.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from shared.video_manager import VideoManager
from scripts.build_triview_cache import TriViewCacheBuilder

def main():
    """Measure preprocessing for a few videos at each duration."""
    
    # Load eval subset
    with open('data/eval_subset.json', 'r') as f:
        eval_samples = json.load(f)
    
    # Get 2-3 videos per duration
    video_manager = VideoManager()
    
    videos_by_duration = {
        'short': [],
        'medium': [],
        'long': []
    }
    
    # Group videos by duration
    for sample in eval_samples:
        if len(videos_by_duration[sample['duration']]) < 3:
            videos_by_duration[sample['duration']].append({
                'youtube_id': sample['youtube_id'],
                'video_id': sample['video_id'],
                'duration': sample['duration']
            })
    
    print("Measuring preprocessing times for tri-view...")
    print(f"Short: {len(videos_by_duration['short'])} videos")
    print(f"Medium: {len(videos_by_duration['medium'])} videos")
    print(f"Long: {len(videos_by_duration['long'])} videos")
    print()
    
    # Initialize builder
    builder = TriViewCacheBuilder()
    
    results = []
    
    for dur in ['short', 'medium', 'long']:
        print(f"\n{dur.upper()} VIDEOS:")
        for video in videos_by_duration[dur]:
            youtube_id = video['youtube_id']
            video_path = video_manager.get_video(youtube_id)
            
            if not video_path:
                print(f"  {youtube_id}: video unavailable")
                continue
            
            print(f"  {youtube_id}... ", end='', flush=True)
            
            try:
                # Use existing build_triview_cache method
                triview_data, preprocessing_time = builder.build_triview_cache(
                    youtube_id, 
                    'data/ekg_cache',
                    video_path
                )
                
                if triview_data:
                    print(f"{preprocessing_time:.2f}s")
                    results.append({
                        'youtube_id': youtube_id,
                        'duration': dur,
                        'preprocessing_time': preprocessing_time
                    })
                else:
                    print("no EKG")
                    
            except Exception as e:
                print(f"error: {e}")
    
    # Save results
    output = {
        'level': 'level4_triview',
        'preprocessing_times': results,
        'summary': {
            'short': {
                'count': len([r for r in results if r['duration'] == 'short']),
                'avg_time': float(np.mean([r['preprocessing_time'] for r in results if r['duration'] == 'short'])) if any(r['duration'] == 'short' for r in results) else 0
            },
            'medium': {
                'count': len([r for r in results if r['duration'] == 'medium']),
                'avg_time': float(np.mean([r['preprocessing_time'] for r in results if r['duration'] == 'medium'])) if any(r['duration'] == 'medium' for r in results) else 0
            },
            'long': {
                'count': len([r for r in results if r['duration'] == 'long']),
                'avg_time': float(np.mean([r['preprocessing_time'] for r in results if r['duration'] == 'long'])) if any(r['duration'] == 'long' for r in results) else 0
            }
        }
    }
    
    output_path = Path('data/triview_cache/preprocessing_times.json')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING TIME MEASUREMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nAverage preprocessing times:")
    for dur in ['short', 'medium', 'long']:
        stats = output['summary'][dur]
        if stats['count'] > 0:
            print(f"  {dur}: {stats['avg_time']:.2f}s (n={stats['count']})")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()