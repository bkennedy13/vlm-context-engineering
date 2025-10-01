import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.video_manager import VideoManager
from tqdm import tqdm

def download_eval_videos():
    """Download all videos in the evaluation subset"""
    
    # Load eval subset
    with open('data/eval_subset.json', 'r') as f:
        eval_samples = json.load(f)
    
    # Get unique video IDs
    unique_videos = {}
    for sample in eval_samples:
        vid_id = sample['video_id']
        if vid_id not in unique_videos:
            unique_videos[vid_id] = sample['youtube_id']
    
    print(f"Downloading {len(unique_videos)} unique videos...")
    
    # Download each video
    video_manager = VideoManager()
    failed = []
    
    for vid_id, youtube_id in tqdm(unique_videos.items(), desc="Downloading videos"):
        video_path = video_manager.get_video(youtube_id)
        if video_path is None:
            failed.append((vid_id, youtube_id))
            print(f"\nFailed to download: {vid_id} ({youtube_id})")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Downloaded: {len(unique_videos) - len(failed)}/{len(unique_videos)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("Failed videos:")
        for vid_id, yt_id in failed:
            print(f"  {vid_id}: {yt_id}")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = download_eval_videos()
    sys.exit(0 if success else 1)