import cv2
import numpy as np
from pathlib import Path
import pickle
import time

def extract_frames_at_indices(video_path, frame_indices):
    """
    Extract specific frames from video by their indices
    
    Args:
        video_path: Path to video file
        frame_indices: List of frame numbers to extract
    
    Returns:
        List of numpy arrays (RGB frames)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_indices_set = set(frame_indices)
    current_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_idx in frame_indices_set:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        current_idx += 1
        
        # Early exit if we've collected all needed frames
        if len(frames) == len(frame_indices):
            break
    
    cap.release()
    return frames

def create_5s_chunks(video_path, fps=1):
    """
    Create 5-second chunks with 5 frames each (1 FPS)
    
    Args:
        video_path: Path to video file
        fps: Frames per second to sample (default: 1)
    
    Returns:
        chunks: List of chunks (each chunk is list of 5 frames)
        chunk_metadata: List of dicts with start_time, end_time, frame_indices
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    cap.release()
    
    # Calculate frame interval
    frame_interval = int(original_fps / fps)
    chunk_size = 5  # 5 frames = 5 seconds at 1 FPS
    
    # Generate all frame indices we need
    all_frame_indices = list(range(0, total_frames, frame_interval))
    
    # Group into chunks of 5
    chunks = []
    chunk_metadata = []
    
    for i in range(0, len(all_frame_indices) - chunk_size + 1, chunk_size):
        chunk_frame_indices = all_frame_indices[i:i + chunk_size]
        
        # Calculate time boundaries
        start_time = chunk_frame_indices[0] / original_fps
        end_time = chunk_frame_indices[-1] / original_fps
        
        chunk_metadata.append({
            'chunk_id': len(chunk_metadata),
            'start_time': start_time,
            'end_time': end_time,
            'frame_indices': chunk_frame_indices
        })
    
    # Extract all frames in one pass
    all_frames = extract_frames_at_indices(video_path, 
                                           [idx for meta in chunk_metadata 
                                            for idx in meta['frame_indices']])
    
    # Group frames into chunks
    for i, meta in enumerate(chunk_metadata):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunks.append(all_frames[start_idx:end_idx])
    
    print(f"Created {len(chunks)} 5-second chunks from {duration:.1f}s video")
    return chunks, chunk_metadata

def save_chunk_cache(video_path, chunk_metadata, descriptions, cache_dir="data/semantic_cache"):
    """
    Save chunk descriptions and metadata to cache
    
    Args:
        video_path: Path to source video
        chunk_metadata: List of chunk metadata dicts
        descriptions: List of description strings
        cache_dir: Directory to store cache files
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    video_id = Path(video_path).stem
    cache_path = cache_dir / f"{video_id}.pkl"
    
    cache_data = {
        'video_id': video_id,
        'video_path': str(video_path),
        'chunk_metadata': chunk_metadata,
        'descriptions': descriptions,
        'chunk_size': 5,  # seconds
        'fps': 1,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_chunks': len(chunk_metadata)
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Saved cache: {cache_path}")
    return cache_path

def load_chunk_cache(video_path, cache_dir="data/semantic_cache"):
    """
    Load chunk descriptions and metadata from cache
    
    Args:
        video_path: Path to source video
        cache_dir: Directory where cache files are stored
    
    Returns:
        dict with chunk_metadata and descriptions, or None if not cached
    """
    cache_dir = Path(cache_dir)
    video_id = Path(video_path).stem
    cache_path = cache_dir / f"{video_id}.pkl"
    
    if not cache_path.exists():
        return None
    
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    return cache_data