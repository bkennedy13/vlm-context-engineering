import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import sys
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

class SemanticVLMAnswerer:  
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize 7B VLM for answering"""
        print(f"Loading Semantic VLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load 7B answering model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"Semantic VLM loaded on {self.device}")
    
    def answer_from_video(self, video_path, question, options, k=10, semantic_rag=None):
        """Main entry point: retrieve from cache and answer"""
        if semantic_rag is None:
            raise ValueError("Must provide SemanticRAG instance for retrieval")
            
        # Use SemanticRAG's retrieval
        chunks, similarities, timing = semantic_rag.process_video(video_path, question, k)
        
        # Load descriptions from cache
        cache_path = semantic_rag._get_cache_path(video_path)
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        descriptions = cache_data['semantic_descriptions']
        
        # Answer using retrieved chunks
        return self.answer_question(chunks, similarities, descriptions, question, options)
    
    def _select_chunks_adaptively(self, similarities, min_chunks=1, max_chunks=3, gap_threshold=0.05):
        """
        Select chunks based on similarity drop-off
        
        Args:
            similarities: Array of similarity scores
            min_chunks: Minimum chunks to use even if gap is large
            max_chunks: Maximum chunks to consider
            gap_threshold: Minimum gap to trigger cutoff
        
        Returns:
            Number of chunks to use
        """
        if len(similarities) <= min_chunks:
            return len(similarities)
        
        # Sort descending
        sorted_sims = np.sort(similarities)[::-1]
        
        # Look for significant drop-off
        for i in range(min_chunks - 1, min(max_chunks, len(sorted_sims)) - 1):
            gap = sorted_sims[i] - sorted_sims[i + 1]
            if gap > gap_threshold:
                return i + 1
        
        # No significant gap found, use max_chunks
        return min(max_chunks, len(similarities))
    
    def _allocate_frame_budget(self, n_selected_chunks, similarities):
        """
        Distribute frame budget across selected chunks based on similarity
        
        Returns:
            List of frame counts per chunk
        """
        total_budget = 25
        
        # Normalize similarities to get budget proportions
        selected_sims = similarities[:n_selected_chunks]
        sim_sum = np.sum(selected_sims)
        
        if sim_sum == 0:
            # Equal distribution if all similarities are zero
            proportions = np.ones(n_selected_chunks) / n_selected_chunks
        else:
            proportions = selected_sims / sim_sum
        
        # Allocate budget proportionally, with minimum 2 frames per chunk
        min_frames = 2
        remaining_budget = total_budget - (min_frames * n_selected_chunks)
        
        if remaining_budget < 0:
            # If we can't give min_frames to all, just distribute evenly
            budgets = np.full(n_selected_chunks, total_budget // n_selected_chunks)
        else:
            # Distribute remaining budget proportionally
            extra_frames = (proportions * remaining_budget).astype(int)
            budgets = min_frames + extra_frames
            
            # Handle rounding by giving extra frames to top chunks
            while budgets.sum() < total_budget:
                for i in range(n_selected_chunks):
                    if budgets.sum() >= total_budget:
                        break
                    budgets[i] += 1
        
        return budgets
    
    def _sample_frames_from_chunk(self, chunk, budget):
        """
        Sample frames from a chunk based on allocated budget
        
        Uses adaptive sampling density based on chunk length
        """
        chunk_length = len(chunk)
        
        if chunk_length == 0:
            return []
        
        if chunk_length <= budget:
            # Use all frames if chunk is small
            return list(chunk)
        
        # Sample evenly across the chunk, ensuring start and end are included
        if budget == 1:
            return [chunk[chunk_length // 2]]
        elif budget == 2:
            return [chunk[0], chunk[-1]]
        else:
            # Include start, end, and evenly spaced middle frames
            indices = [0]  # Start
            
            # Middle frames
            step = (chunk_length - 1) / (budget - 1)
            for i in range(1, budget - 1):
                idx = int(round(i * step))
                indices.append(idx)
            
            indices.append(chunk_length - 1)  # End
            
            return [chunk[i] for i in indices]
    
    def answer_question(self, chunks, similarities, descriptions, question, options):
        """
        Generate answer using semantic chunks with adaptive selection
        
        Args:
            chunks: List of semantic chunks (variable-length frame lists)
            similarities: Similarity scores for each chunk
            descriptions: Text descriptions for each semantic chunk
            question: The question string
            options: List of option strings
        
        Returns:
            Single letter answer (A, B, C, or D)
        """
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_sims = similarities[sorted_indices]
        
        # Adaptively select number of chunks
        n_selected = self._select_chunks_adaptively(sorted_sims)
        selected_indices = sorted_indices[:n_selected]
        selected_sims = sorted_sims[:n_selected]
        
        print(f"Selected {n_selected} chunks based on similarity drop-off")
        print(f"Similarity scores: {selected_sims}")
        
        # Allocate frame budget
        frame_budgets = self._allocate_frame_budget(n_selected, selected_sims)
        print(f"Frame allocation: {frame_budgets}")
        
        # Build content: description + frames for each selected chunk
        content = []
        total_frames = 0
        
        for i, chunk_idx in enumerate(selected_indices):
            chunk = chunks[chunk_idx]
            description = descriptions[chunk_idx]
            budget = frame_budgets[i]
            
            # Add description
            content.append({
                "type": "text",
                "text": f"Event {i+1}: {description[:500]}"  # Truncate very long descriptions
            })
            
            # Sample and add frames
            sampled_frames = self._sample_frames_from_chunk(chunk, budget)
            
            for frame in sampled_frames:
                # Convert to PIL
                if isinstance(frame, np.ndarray):
                    pil_frame = Image.fromarray(frame)
                else:
                    pil_frame = frame
                
                content.append({
                    "type": "image",
                    "image": pil_frame
                })
                total_frames += 1
        
        print(f"Total frames sent to VLM: {total_frames}")
        
        # Add question
        options_text = "\n".join(options)
        content.append({
            "type": "text",
            "text": f"\nBased on the events shown above, answer the following question:\n\n{question}\n\n{options_text}\n\nAnswer with only the letter (A, B, C, or D):"
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
        
        print(f"Semantic VLM raw output: {output_text}")
        
        output_text = output_text.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in output_text:
                return letter
        
        return output_text[0] if output_text else 'A'