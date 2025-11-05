"""
VLM answerer using retrieved events.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class EventAnswerer:
    """Generate answers from retrieved events."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize VLM."""
        print(f"Loading VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"EventAnswerer loaded on {self.device}")
    
    def answer_question(self, events, video_path, question, options, frame_budget=25):
        """
        Generate answer from retrieved events.
        
        Args:
            events: List of event dicts
            video_path: Path to video file
            question: Question string
            options: List of option strings
            frame_budget: Total frames to use
        
        Returns:
            Single letter answer (A, B, C, or D)
        """
        # Extract all video frames at 1 FPS
        all_frames = self._extract_video_frames(video_path)
        
        # Get video FPS for frame index mapping
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / 1)
        cap.release()
        
        # Build VLM prompt
        content = []
        frames_per_event = max(2, frame_budget // len(events)) if events else 0
        total_frames = 0
        
        for i, event in enumerate(events):
            # Add description
            content.append({
                "type": "text",
                "text": f"Event {i+1}: {event['description'][:1000]}"
            })
            
            # Sample frames from event
            frames = self._sample_frames_from_event(
                all_frames, event, frames_per_event, frame_interval
            )
            
            for frame in frames:
                pil_frame = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
                content.append({"type": "image", "image": pil_frame})
                total_frames += 1
                
                if total_frames >= frame_budget:
                    break
            
            if total_frames >= frame_budget:
                break
        
        print(f"Total frames to VLM: {total_frames}")
        
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