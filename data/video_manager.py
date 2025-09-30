import json
from pathlib import Path
import sys
import os

# Add the baseline directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
from video_utils import download_youtube_video, get_video_info

class VideoManager:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_file = self.data_dir / "video_cache.json"
        self.cache = self.load_cache()
    
    def load_cache(self):
        """Load video cache to track what's downloaded"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        """Save video cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_video(self, youtube_id, force_download=False):
        """Get video, download if not cached"""
        if youtube_id in self.cache and not force_download:
            video_path = self.cache[youtube_id]['path']
            if Path(video_path).exists():
                print(f"Using cached video: {video_path}")
                return video_path
        
        # Download video
        print(f"Downloading video: {youtube_id}")
        video_path = download_youtube_video(youtube_id, self.data_dir)
        if video_path:
            # Cache the video info
            info = get_video_info(video_path)
            self.cache[youtube_id] = {
                'path': video_path,
                'info': info,
                'youtube_id': youtube_id
            }
            self.save_cache()
            return video_path
        
        return None
    
    def cleanup_large_videos(self, max_size_mb=100):
        """Remove videos larger than max_size_mb to save space"""
        removed = []
        for youtube_id, data in list(self.cache.items()):
            if data['info']['size_mb'] > max_size_mb:
                video_path = Path(data['path'])
                if video_path.exists():
                    video_path.unlink()
                    print(f"Removed large video: {youtube_id} ({data['info']['size_mb']:.1f}MB)")
                removed.append(youtube_id)
                del self.cache[youtube_id]
        
        if removed:
            self.save_cache()
        
        return removed
    
    def cleanup_all(self):
        """Remove all cached videos"""
        for youtube_id, data in self.cache.items():
            video_path = Path(data['path'])
            if video_path.exists():
                video_path.unlink()
        
        self.cache.clear()
        self.save_cache()
        print("Cleared all cached videos")