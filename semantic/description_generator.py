import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np

class DescriptionGenerator:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"): #7b too slow
        """Initialize VLM for generating chunk descriptions"""
        print(f"Loading Description Generator: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Reuse same model as VLMAnswerer to avoid loading twice
        if model_name == "Qwen/Qwen2-VL-2B-Instruct":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def generate_description(self, frames):
        """
        Generate text description for a chunk of frames
        
        Args:
            frames: List of PIL Images or numpy arrays
        
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
                        "text": "Describe what is happening in these video frames in 1-2 sentences. Focus on actions, objects, and events."
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
                max_new_tokens=64,  # Keep descriptions concise
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
    
    def generate_batch_descriptions(self, chunks):
        """Generate descriptions for multiple chunks"""
        descriptions = []
        for i, chunk in enumerate(chunks):
            desc = self.generate_description(chunk)
            descriptions.append(desc)
            if (i + 1) % 5 == 0:
                print(f"Generated {i + 1}/{len(chunks)} descriptions")
        return descriptions