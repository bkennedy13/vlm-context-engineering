import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np

class VLMAnswerer:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize VLM for answer generation"""
        print(f"Loading VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and processor
        # Note: 7B model will use most of your T4's 16GB VRAM
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"VLM loaded on {self.device}")
        
    def answer_question(self, chunks, question, options):
        """
        Generate answer from retrieved chunks and question
        
        Args:
            chunks: List of frame chunks (each chunk is a list of numpy arrays)
            question: The question string
            options: List of option strings like ['A. ...', 'B. ...', etc]
        
        Returns:
            Single letter answer (A, B, C, or D)
        """
        # Select middle frame from each chunk as representative
        frames = []
        for chunk in chunks:
            if len(chunk) > 0:
                mid_idx = len(chunk) // 2
                frame = chunk[mid_idx]
                # Convert numpy array to PIL Image
                if isinstance(frame, np.ndarray):
                    frames.append(Image.fromarray(frame))
                else:
                    frames.append(frame)
        
        # Format options
        options_text = "\n".join(options)
        
        # Create multi-modal message
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": frame} for frame in frames],
                    {
                        "type": "text", 
                        "text": f"{question}\n\n{options_text}\n\nAnswer with only the letter (A, B, C, or D):"
                    }
                ]
            }
        ]
        
        # Prepare inputs
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
        
        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False  # Deterministic for consistency
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"VLM raw output: {output_text}")
        
        # Extract just the letter (A, B, C, or D)
        output_text = output_text.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in output_text:
                return letter
        
        # Fallback if no clear answer
        return output_text[0] if output_text else 'A'