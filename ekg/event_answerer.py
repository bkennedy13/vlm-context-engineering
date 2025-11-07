"""
VLM answerer using retrieved events.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
#from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class EventAnswerer:
    """Generate answers from retrieved events."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"): #"Qwen/Qwen3-VL-8B-Instruct"
        """Initialize VLM."""
        print(f"Loading VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"EventAnswerer loaded on {self.device}")
    
    def answer_question(self, events, video_path, question, options, frame_budget=25):
        """Generate answer from retrieved events."""
        # Extract all video frames at 1 FPS
        all_frames = self._extract_video_frames(video_path)
        
        # Get video FPS for frame index mapping
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / 1)
        cap.release()
        
        # Sort events by temporal order (start_time)
        events = sorted(events, key=lambda e: e['start_time'])
        
        # Build VLM prompt
        content = []
        
        # Add temporal context header
        content.append({
            "type": "text",
            "text": "The following events are shown in chronological order:\n"
        })
        
        # Allocate frames proportionally based on event scores
        total_available = sum(len(e['frame_indices']) for e in events)
        frames_allocated = []
        
        for i, event in enumerate(events):
            # Allocate frames proportional to event's share of total available frames
            event_frames = len(event['frame_indices'])
            if i == len(events) - 1:
                # Last event gets remaining frames
                frames_for_event = frame_budget - sum(frames_allocated)
            else:
                # Allocate proportionally, minimum 1 frame
                proportion = event_frames / total_available
                frames_for_event = max(1, int(proportion * frame_budget))
            
            frames_allocated.append(frames_for_event)
        
        # Ensure we allocate exactly frame_budget frames
        total_allocated = sum(frames_allocated)
        if total_allocated > frame_budget:
            # Reduce from largest allocation
            diff = total_allocated - frame_budget
            max_idx = frames_allocated.index(max(frames_allocated))
            frames_allocated[max_idx] -= diff
        elif total_allocated < frame_budget:
            # Add to largest allocation
            diff = frame_budget - total_allocated
            max_idx = frames_allocated.index(max(frames_allocated))
            frames_allocated[max_idx] += diff
        
        # Now extract frames for each event
        frames_collected = 0
        for i, event in enumerate(events):
            frames_for_event = frames_allocated[i]
            
            if frames_for_event <= 0:
                continue
            
            # Add event description with timestamp
            content.append({
                "type": "text",
                "text": f"Event {i+1} ({event['start_time']:.1f}s - {event['end_time']:.1f}s): {event['description'][:3000]}"
            })
            
            # debugging
            print(f"Event {i+1} ({event['start_time']:.1f}s - {event['end_time']:.1f}s)")
            
            # Sample frames from event
            frames = self._sample_frames_from_event(
                all_frames, event, frames_for_event, frame_interval
            )
            
            for frame in frames:
                pil_frame = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
                content.append({"type": "image", "image": pil_frame})
                frames_collected += 1
        
        print(f"Total frames to VLM: {frames_collected}")
        
        # Add question
        options_text = "\n".join(options)
        content.append({
            "type": "text",
            "text": f"\n{question}\n\n{options_text}\n\nAnswer with only the letter (A, B, C, or D):"
        })
        
        messages = [{"role": "user", "content": content}]
        
        # Generate answer
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Print token count
        num_tokens = inputs.input_ids.shape[1]
        num_images = len(image_inputs) if image_inputs else 0
        print(f"Tokens: {num_tokens} (text + {num_images} images)")
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"VLM output: {output_text}")
        
        # Extract answer letter
        output_text = output_text.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in output_text:
                return letter
        
        return output_text[0] if output_text else 'A'
    
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
    
    def _sample_frames_from_event(self, all_frames, event, budget, frame_interval):
        """Sample frames evenly from an event."""
        # Event stores original video frame indices
        # Convert to 1 FPS array indices
        frame_indices = event['frame_indices']
        fps_indices = [idx // frame_interval for idx in frame_indices]
        
        # Sample evenly
        if len(fps_indices) <= budget:
            sampled_indices = fps_indices
        elif budget == 1:
            sampled_indices = [fps_indices[len(fps_indices) // 2]]
        elif budget == 2:
            sampled_indices = [fps_indices[0], fps_indices[-1]]
        else:
            step = (len(fps_indices) - 1) / (budget - 1)
            sampled_indices = [fps_indices[int(round(i * step))] for i in range(budget)]
        
        # Extract frames
        frames = []
        for idx in sampled_indices:
            if idx < len(all_frames):
                frames.append(all_frames[idx])
        
        return frames