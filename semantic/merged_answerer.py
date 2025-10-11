import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np

class MergedVLMAnswerer:
    """Generate answers using Qwen2.5-VL-7B from retrieved merged chunks"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"Loading VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"✓ MergedVLMAnswerer loaded on {self.device}")
    
    def answer_question(self, chunks, descriptions, similarities, question, options):
        """
        Generate answer from retrieved merged chunks
        
        Args:
            chunks: List of retrieved chunks (each is list of frames)
            descriptions: List of merged descriptions for chunks
            similarities: CLIP similarity scores for chunks
            question: Question string
            options: List of option strings
        
        Returns:
            Single letter answer (A, B, C, or D)
        """
        # Adaptive chunk selection based on similarity drop-off
        n_selected = self._select_chunks_adaptively(similarities)
        selected_indices = np.argsort(similarities)[::-1][:n_selected]
        selected_sims = similarities[selected_indices]
        
        print(f"Selected {n_selected} chunks (similarities: {selected_sims})")
        
        # Allocate frame budget across selected chunks
        frame_budgets = self._allocate_frame_budget(n_selected, selected_sims)
        print(f"Frame budget: {frame_budgets}")
        
        # Build VLM prompt: description + sampled frames per chunk
        content = []
        total_frames = 0
        
        for i, chunk_idx in enumerate(selected_indices):
            chunk = chunks[chunk_idx]
            description = descriptions[chunk_idx]
            budget = frame_budgets[i]
            
            # Add description
            content.append({
                "type": "text",
                "text": f"Event {i+1}: {description[:500]}"
            })
            
            # Sample frames from chunk based on budget
            sampled_frames = self._sample_frames_from_chunk(chunk, budget)
            
            for frame in sampled_frames:
                pil_frame = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
                content.append({"type": "image", "image": pil_frame})
                total_frames += 1
        
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
    
    def _select_chunks_adaptively(self, similarities, min_chunks=1, max_chunks=3, gap_threshold=0.05):
        """Select chunks based on similarity drop-off"""
        if len(similarities) <= min_chunks:
            return len(similarities)
        
        sorted_sims = np.sort(similarities)[::-1]
        
        for i in range(min_chunks - 1, min(max_chunks, len(sorted_sims)) - 1):
            gap = sorted_sims[i] - sorted_sims[i + 1]
            if gap > gap_threshold:
                return i + 1
        
        return min(max_chunks, len(similarities))
    
    def _allocate_frame_budget(self, n_selected_chunks, similarities):
        """Distribute 25 frames across chunks based on similarity"""
        total_budget = 25
        
        selected_sims = similarities[:n_selected_chunks]
        sim_sum = np.sum(selected_sims)
        
        if sim_sum == 0:
            proportions = np.ones(n_selected_chunks) / n_selected_chunks
        else:
            proportions = selected_sims / sim_sum
        
        min_frames = 2
        remaining_budget = total_budget - (min_frames * n_selected_chunks)
        
        if remaining_budget < 0:
            budgets = np.full(n_selected_chunks, total_budget // n_selected_chunks)
        else:
            extra_frames = (proportions * remaining_budget).astype(int)
            budgets = min_frames + extra_frames
            
            while budgets.sum() < total_budget:
                for i in range(n_selected_chunks):
                    if budgets.sum() >= total_budget:
                        break
                    budgets[i] += 1
        
        return budgets
    
    def _sample_frames_from_chunk(self, chunk, budget):
        """Sample frames evenly from chunk based on budget"""
        chunk_length = len(chunk)
        
        if chunk_length == 0:
            return []
        
        if chunk_length <= budget:
            return list(chunk)
        
        if budget == 1:
            return [chunk[chunk_length // 2]]
        elif budget == 2:
            return [chunk[0], chunk[-1]]
        else:
            indices = [0]
            step = (chunk_length - 1) / (budget - 1)
            for i in range(1, budget - 1):
                idx = int(round(i * step))
                indices.append(idx)
            indices.append(chunk_length - 1)
            
            return [chunk[i] for i in indices]