import yt_dlp
import json
import requests
from pathlib import Path
import cv2
import numpy as np
from datasets import load_dataset

def load_video_mme_dataset(split="test"):
    """Load Video-MME dataset from HuggingFace"""
    try:
        ds = load_dataset("lmms-lab/Video-MME", split=split)
        return ds
    except Exception as e:
        print(f"Error loading Video-MME dataset: {e}")
        return None

def get_short_visual_samples(dataset, n_samples=5):
    """Get short duration videos with visual-only questions"""
    if not dataset:
        return []
    
    # Filter for short videos and visual task types
    visual_tasks = [
        "Counting Problem", 
        "Information Synopsis", 
        "Spatial Relationship",
        "Visual Recognition",
        "Action Recognition"
    ]
    
    samples = []
    for item in dataset:
        if (item['duration'] == 'short' and 
            item['task_type'] in visual_tasks and 
            len(samples) < n_samples):
            
            samples.append({
                'video_id': item['video_id'],
                'youtube_id': item['videoID'],
                'url': item['url'],
                'question': item['question'],
                'options': item['options'],
                'answer': item['answer'],
                'task_type': item['task_type'],
                'domain': item['domain'],
                'sub_category': item['sub_category']
            })
    
    return samples

def extract_frames_at_fps(video_path, fps=1):
    """Extract frames from video at specified FPS"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB for consistency with CLIP
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def create_chunks(frames, chunk_size=3):
    """Group frames into chunks of specified size"""
    chunks = []
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def download_youtube_video(youtube_id, output_dir="data"):
    """Download YouTube video with fallback formats"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Try formats in order: 360p video-only, then any video-only, then best overall
    format_options = [
        '134',              # 640x360 mp4 video-only (preferred)
        '243',              # 640x360 webm video-only (alternative)
        '160',              # 256x144 mp4 video-only (lower quality fallback)
        'bestvideo[ext=mp4][height<=720]',  # Best mp4 video-only up to 720p
        'best[height<=720]'  # Last resort with audio
    ]
    
    for fmt in format_options:
        ydl_opts = {
            'format': fmt,
            'outtmpl': str(output_path / f'{youtube_id}.%(ext)s'),
            'no_warnings': True,
            'quiet': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                url = f"https://www.youtube.com/watch?v={youtube_id}"
                ydl.download([url])
                
            # Find downloaded file
            video_files = list(output_path.glob(f"{youtube_id}.*"))
            for f in video_files:
                if f.suffix in ['.mp4', '.webm', '.mkv']:
                    print(f"Downloaded: {f} (format: {fmt})")
                    return str(f)
        except:
            continue
    
    print(f"All formats failed for {youtube_id}")
    return None

def get_video_info(video_path):
    """Get basic video information"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'duration': duration,
        'fps': fps,
        'frame_count': frame_count,
        'resolution': (width, height),
        'size_mb': Path(video_path).stat().st_size / (1024 * 1024)
    }

def create_evaluation_subset(dataset, n_short=20, n_medium=20, n_long=10, seed=42):
    """Create balanced eval subset across durations"""
    import random
    random.seed(seed)
    
    eval_samples = []
    duration_targets = {
        'short': n_short,
        'medium': n_medium,
        'long': n_long
    }
    
    for duration, n_videos in duration_targets.items():
        # Get all videos of this duration
        duration_videos = {}
        for item in dataset:
            if item['duration'] == duration:
                vid_id = item['video_id']
                if vid_id not in duration_videos:
                    duration_videos[vid_id] = []
                duration_videos[vid_id].append({
                    'video_id': item['video_id'],
                    'youtube_id': item['videoID'],
                    'url': item['url'],
                    'question': item['question'],
                    'options': item['options'],
                    'answer': item['answer'],
                    'task_type': item['task_type'],
                    'question_id': item['question_id'],
                    'duration': item['duration']
                })
        
        # Sample n videos
        video_ids = list(duration_videos.keys())
        selected_ids = random.sample(video_ids, min(n_videos, len(video_ids)))
        
        # Add all questions for selected videos
        for vid_id in selected_ids:
            eval_samples.extend(duration_videos[vid_id])
        
        print(f"{duration}: {len(selected_ids)} videos")
    
    print(f"\nTotal: {len(set(s['video_id'] for s in eval_samples))} videos, {len(eval_samples)} questions")
    
    return eval_samples