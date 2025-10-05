import numpy as np
from bert_score import score

class SemanticChunker:
    def __init__(self, similarity_threshold=0.65):
        """
        Initialize semantic chunker with BERTScore
        
        Args:
            similarity_threshold: Merge chunks if BERTScore > this value
        """
        self.threshold = similarity_threshold
    
    def compute_pairwise_bertscore(self, descriptions):
        """Compute BERTScore on CPU to avoid OOM"""
        if len(descriptions) < 2:
            return []
        
        candidates = descriptions[1:]
        references = descriptions[:-1]
        
        # Force CPU for BERTScore
        P, R, F1 = score(
            candidates, 
            references, 
            lang='en',
            model_type='microsoft/deberta-xlarge-mnli',
            device='cpu',  # ADD THIS
            verbose=False
        )
        
        return F1.numpy()
    
    def merge_chunks(self, chunks, descriptions):
        """
        Merge adjacent chunks with high semantic similarity
        
        Args:
            chunks: List of frame chunks
            descriptions: List of text descriptions for each chunk
        
        Returns:
            merged_chunks: List of merged frame chunks
            merged_descriptions: List of descriptions for merged chunks
        """
        if len(chunks) != len(descriptions):
            raise ValueError("Chunks and descriptions must have same length")
        
        if len(chunks) <= 1:
            return chunks, descriptions
        
        # Compute similarity between adjacent chunks
        similarities = self.compute_pairwise_bertscore(descriptions)
        
        # Merge chunks greedily
        merged_chunks = []
        merged_descriptions = []
        
        current_chunk = chunks[0]
        current_desc = descriptions[0]
        
        for i in range(1, len(chunks)):
            # Check if should merge with next chunk
            if similarities[i-1] > self.threshold:
                # Merge: concatenate frames
                current_chunk = current_chunk + chunks[i]
                # Update description (could also regenerate, but expensive)
                current_desc = f"{current_desc} {descriptions[i]}"
            else:
                # Don't merge: save current and start new
                merged_chunks.append(current_chunk)
                merged_descriptions.append(current_desc)
                current_chunk = chunks[i]
                current_desc = descriptions[i]
        
        # Add last chunk
        merged_chunks.append(current_chunk)
        merged_descriptions.append(current_desc)
        
        print(f"Merged {len(chunks)} uniform chunks into {len(merged_chunks)} semantic chunks")
        return merged_chunks, merged_descriptions