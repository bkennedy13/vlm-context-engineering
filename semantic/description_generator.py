import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
from pathlib import Path

class DescriptionGenerator:
    """Generate text descriptions for video chunks using Qwen2-VL-2B"""
    
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        print(f"Loading Description Generator: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"Description Generator loaded on {self.device}")
        
    def generate_description(self, frames):
        """
        Generate text description for a chunk of frames
        
        Args:
            frames: List of 5 PIL Images or numpy arrays
        
        Returns:
            str: Description of what's happening in the frames
        """
        # Convert to PIL if needed
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                pil_frames.append(Image.fromarray(frame))
            else:
                pil_frames.append(frame)
        
        # Create prompt
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": frame} for frame in pil_frames],
                    {
                        "type": "text",
                        "text": (
                            "You are viewing 5 consecutive frames from a video, captured 1 second "
                            "apart (5 seconds total). Describe what is happening, focusing on:\n"
                            "- Key objects and their attributes\n"
                            "- Actions and motion (what changed between frames)\n"
                            "- Scene context (location, setting)\n"
                            "- Any notable events or transitions\n\n"
                            "Keep your description concise but informative (2-3 sentences)."
                        )
                    }
                ]
            }
        ]
        
        # Generate description
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
        
        description = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return description.strip()
    
    def generate_batch_descriptions(self, chunks, verbose=True):
        """
        Generate descriptions for multiple chunks with progress tracking
        
        Args:
            chunks: List of chunks (each chunk is list of 5 frames)
            verbose: Print progress every 10 chunks
        
        Returns:
            List of description strings
        """
        descriptions = []
        for i, chunk in enumerate(chunks):
            if len(chunk) != 5:
                print(f"Warning: Chunk {i} has {len(chunk)} frames, expected 5")
            
            desc = self.generate_description(chunk)
            descriptions.append(desc)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(chunks)} descriptions")
        
        return descriptions