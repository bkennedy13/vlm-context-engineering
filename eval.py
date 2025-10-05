import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from shared.vlm_answerer import VLMAnswerer
from shared.video_manager import VideoManager
from baseline.baseline_rag import VideoRAGBaseline
from semantic.semantic_rag import SemanticRAG
from semantic.semantic_answerer import SemanticVLMAnswerer
import open_clip

def precompute_question_embeddings(eval_subset_path='data/eval_subset.json',
                                 embeddings_path='data/question_embeddings.json'):
    """Precompute CLIP embeddings for all questions"""
    
    # Load eval subset
    print("Loading evaluation subset...")
    with open(eval_subset_path, 'r') as f:
        eval_samples = json.load(f)
    
    print(f"Precomputing embeddings for {len(eval_samples)} questions...")
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model = clip_model.to("cuda")
    clip_model.eval()
    
    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory with CLIP: {allocated:.2f}GB")
    
    # Compute embeddings for all questions
    question_embeddings = {}
    embedding_times = {}
    
    for sample in tqdm(eval_samples, desc="Computing question embeddings"):
        question_id = sample['question_id']
        
        # Time the embedding
        embed_start = time.time()
        
        # Tokenize and embed question
        text_tokens = tokenizer([sample['question']]).to("cuda")
        with torch.no_grad():
            question_embedding = clip_model.encode_text(text_tokens)
            question_embedding = question_embedding / question_embedding.norm(dim=-1, keepdim=True)
        
        embed_time = time.time() - embed_start
        
        # Store embedding and timing
        question_embeddings[question_id] = question_embedding.cpu().numpy().tolist()
        embedding_times[question_id] = embed_time
    
    # Save embeddings
    embeddings_data = {
        'embeddings': question_embeddings,
        'embedding_times': embedding_times,
        'total_questions': len(eval_samples),
        'avg_embedding_time': float(np.mean(list(embedding_times.values())))
    }
    
    with open(embeddings_path, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"Question embeddings saved to {embeddings_path}")
    print(f"Average embedding time: {embeddings_data['avg_embedding_time']:.4f}s")
    
    # Clear CLIP from GPU
    del clip_model, preprocess, tokenizer
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory after clearing CLIP: {allocated:.2f}GB")
    
    return embeddings_data

def run_semantic_evaluation(eval_subset_path='data/eval_subset.json', 
                           output_path='results/level2_semantic_rag.json',
                           embeddings_path='data/question_embeddings.json',
                           checkpoint_interval=10):
    """Run semantic evaluation using precomputed question embeddings"""
    
    # Load eval subset
    print("Loading evaluation subset...")
    with open(eval_subset_path, 'r') as f:
        eval_samples = json.load(f)
    
    # Load precomputed question embeddings
    print("Loading precomputed question embeddings...")
    with open(embeddings_path, 'r') as f:
        embeddings_data = json.load(f)
    
    question_embeddings = embeddings_data['embeddings']
    embedding_times = embeddings_data['embedding_times']
    
    n_videos = len(set(s['video_id'] for s in eval_samples))
    print(f"Loaded {len(eval_samples)} questions from {n_videos} videos")
    print(f"Loaded {len(question_embeddings)} precomputed embeddings")
    
    # Initialize models (no CLIP needed now)
    print("\nInitializing models...")
    video_manager = VideoManager()
    semantic_rag = SemanticRAG(similarity_threshold=0.65, load_model=False)  # Cache-only mode
    semantic_vlm = SemanticVLMAnswerer()  # 7B VLM for answering
    
    # Check GPU memory allocation after model loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU Memory after model loading (without CLIP):")
        print(f"  Allocated: {allocated:.2f}GB")
        print(f"  Reserved: {reserved:.2f}GB") 
        print(f"  Total: {total:.2f}GB")
        print(f"  Free: {total - reserved:.2f}GB")
    
    # Run evaluation
    results = []
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
        try:
            # Get video
            video_path = video_manager.get_video(sample['youtube_id'])
            if not video_path:
                print(f"\nSkipping {sample['video_id']}: video unavailable")
                continue
            
            # Get precomputed question embedding
            if sample['question_id'] not in question_embeddings:
                print(f"\nSkipping {sample['question_id']}: no precomputed embedding")
                continue

            question_embedding = np.array(question_embeddings[sample['question_id']])

            # Retrieve relevant chunks using precomputed embedding
            retrieval_start = time.time()
            chunks, similarities, timing = semantic_rag.process_video_with_embedding(
                video_path, 
                question_embedding,
                k=10
            )
            retrieval_time = time.time() - retrieval_start
            
            if chunks is None:
                print(f"\nSkipping {sample['question_id']}: retrieval failed")
                continue
            
            # Load descriptions from cache
            cache_path = semantic_rag._get_cache_path(video_path)
            with open(cache_path, 'rb') as f:
                import pickle
                cache_data = pickle.load(f)
            descriptions = cache_data['semantic_descriptions']
            
            # Get semantic VLM answer with adaptive selection
            inference_start = time.time()
            predicted = semantic_vlm.answer_question(
                chunks, similarities, descriptions,
                sample['question_id'], 
                sample['options']
            )
            inference_time = time.time() - inference_start
            
            # Clear GPU cache to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Store result with detailed timing
            is_correct = predicted == sample['answer']
            results.append({
                'video_id': sample['video_id'],
                'question_id': sample['question_id'],
                'task_type': sample['task_type'],
                'duration': sample['duration'],
                'question': sample['question'],
                'predicted': predicted,
                'correct': sample['answer'],
                'is_correct': is_correct,
                'retrieval_time': retrieval_time,
                'inference_time': inference_time,
                'total_chunks': len(chunks),
                'top_similarity': float(similarities[0]) if len(similarities) > 0 else 0.0,
                'mean_similarity': float(np.mean(similarities)) if len(similarities) > 0 else 0.0,
                'similarity_std': float(np.std(similarities)) if len(similarities) > 0 else 0.0,
                # Semantic-specific metrics
                'chunk_lengths': [len(chunk) for chunk in chunks],
                'max_chunk_length': max(len(chunk) for chunk in chunks) if chunks else 0,
                'avg_chunk_length': float(np.mean([len(chunk) for chunk in chunks])) if chunks else 0.0,
                'total_frames_used': sum(len(chunk) for chunk in chunks)
            })
            
            # Periodic checkpoint
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_path = output_path.replace('.json', f'_checkpoint_{i+1}.json')
                save_semantic_results(results, checkpoint_path, partial=True)
                print(f"\nCheckpoint saved: {len(results)} results so far")
                
        except Exception as e:
            print(f"\nError processing {sample['question_id']}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Save final results
    save_semantic_results(results, output_path, total_time=total_time)
    
    print(f"\n{'='*60}")
    print("SEMANTIC EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_path}")
    
    return results

def save_semantic_results(results, output_path, partial=False, total_time=None):
    """Save semantic evaluation results with summary statistics"""
    
    if not results:
        print("No results to save")
        return
    
    total = len(results)
    correct = sum(r['is_correct'] for r in results)
    accuracy = correct / total if total > 0 else 0
    
    # Task type breakdown
    task_stats = {}
    for task in set(r['task_type'] for r in results):
        task_results = [r for r in results if r['task_type'] == task]
        task_correct = sum(r['is_correct'] for r in task_results)
        task_stats[task] = {
            'total': len(task_results),
            'correct': task_correct,
            'accuracy': task_correct / len(task_results) if task_results else 0
        }
    
    # Duration breakdown
    duration_stats = {}
    for duration in ['short', 'medium', 'long']:
        dur_results = [r for r in results if r['duration'] == duration]
        dur_correct = sum(r['is_correct'] for r in dur_results)
        duration_stats[duration] = {
            'total': len(dur_results),
            'correct': dur_correct,
            'accuracy': dur_correct / len(dur_results) if dur_results else 0,
            'avg_retrieval_time': float(np.mean([r['retrieval_time'] for r in dur_results])) if dur_results else 0,
            'avg_inference_time': float(np.mean([r['inference_time'] for r in dur_results])) if dur_results else 0,
            'avg_chunk_count': float(np.mean([r['total_chunks'] for r in dur_results])) if dur_results else 0,
            'avg_frames_used': float(np.mean([r['total_frames_used'] for r in dur_results])) if dur_results else 0
        }
    
    # Semantic-specific statistics
    semantic_stats = {
        'avg_chunks_per_question': float(np.mean([r['total_chunks'] for r in results])) if results else 0,
        'avg_frames_per_question': float(np.mean([r['total_frames_used'] for r in results])) if results else 0,
        'avg_chunk_length': float(np.mean([r['avg_chunk_length'] for r in results])) if results else 0,
        'max_chunk_length': max([r['max_chunk_length'] for r in results]) if results else 0,
        'avg_top_similarity': float(np.mean([r['top_similarity'] for r in results])) if results else 0,
        'avg_mean_similarity': float(np.mean([r['mean_similarity'] for r in results])) if results else 0
    }
    
    output = {
        'level': 'level2_semantic_rag',
        'config': {
            'k': 10,
            'fps': 1,
            'semantic_chunking': True,
            'similarity_threshold': 0.65,
            'adaptive_selection': True,
            'description_model': 'Qwen2.5-VL-2B-Instruct',
            'answering_model': 'Qwen2.5-VL-7B-Instruct'
        },
        'partial': partial,
        'total_questions': total,
        'correct': correct,
        'accuracy': accuracy,
        'avg_retrieval_time': float(np.mean([r['retrieval_time'] for r in results])) if results else 0,
        'avg_inference_time': float(np.mean([r['inference_time'] for r in results])) if results else 0,
        'total_time': total_time,
        'task_type_stats': task_stats,
        'duration_stats': duration_stats,
        'semantic_stats': semantic_stats,
        'results': results
    }
    
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    if not partial:
        print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"Avg Retrieval Time: {output['avg_retrieval_time']:.2f}s")
        print(f"Avg Inference Time: {output['avg_inference_time']:.2f}s")
        print(f"Avg Chunks per Question: {semantic_stats['avg_chunks_per_question']:.1f}")
        print(f"Avg Frames per Question: {semantic_stats['avg_frames_per_question']:.1f}")
        
        print("\nBy Task Type:")
        for task, stats in sorted(task_stats.items(), key=lambda x: -x[1]['accuracy']):
            print(f"  {task}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        print("\nBy Duration:")
        for dur in ['short', 'medium', 'long']:
            stats = duration_stats[dur]
            print(f"  {dur}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']}) | "
                  f"avg {stats['avg_frames_used']:.0f} frames")

def run_baseline_evaluation(eval_subset_path='data/eval_subset.json', 
                           output_path='results/level1_baseline_rag.json',
                           checkpoint_interval=10):
    """Run baseline evaluation on the full eval set"""
    
    # Load eval subset
    print("Loading evaluation subset...")
    with open(eval_subset_path, 'r') as f:
        eval_samples = json.load(f)
    
    n_videos = len(set(s['video_id'] for s in eval_samples))
    print(f"Loaded {len(eval_samples)} questions from {n_videos} videos")
    
    # Initialize models once
    print("\nInitializing models...")
    video_manager = VideoManager()
    rag = VideoRAGBaseline()
    vlm = VLMAnswerer()
    
    # Run evaluation
    results = []
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
        try:
            # Get video
            video_path = video_manager.get_video(sample['youtube_id'])
            if not video_path:
                print(f"\nSkipping {sample['video_id']}: video unavailable")
                continue
            
            # Retrieve relevant chunks
            retrieval_start = time.time()
            chunks, similarities = rag.process_video(
                video_path, 
                sample['question'], 
                k=10
            )
            retrieval_time = time.time() - retrieval_start
            
            if chunks is None:
                print(f"\nSkipping {sample['question_id']}: retrieval failed")
                continue
            
            # Get VLM answer
            inference_start = time.time()
            predicted = vlm.answer_question(
                chunks, 
                sample['question'], 
                sample['options']
            )
            inference_time = time.time() - inference_start
            
            # Store result
            is_correct = predicted == sample['answer']
            results.append({
                'video_id': sample['video_id'],
                'question_id': sample['question_id'],
                'task_type': sample['task_type'],
                'duration': sample['duration'],
                'question': sample['question'],
                'predicted': predicted,
                'correct': sample['answer'],
                'is_correct': is_correct,
                'retrieval_time': retrieval_time,
                'inference_time': inference_time,
                'top_similarity': float(similarities[0]) if len(similarities) > 0 else 0.0,
                'mean_similarity': float(np.mean(similarities)) if len(similarities) > 0 else 0.0
            })
            
            # Periodic checkpoint
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_path = output_path.replace('.json', f'_checkpoint_{i+1}.json')
                save_results(results, checkpoint_path, partial=True)
                print(f"\nCheckpoint saved: {len(results)} results so far")
                
        except Exception as e:
            print(f"\nError processing {sample['question_id']}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Save final results
    save_results(results, output_path, total_time=total_time)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_path}")
    
    return results

def save_results(results, output_path, partial=False, total_time=None):
    """Save results with summary statistics"""
    
    if not results:
        print("No results to save")
        return
    
    total = len(results)
    correct = sum(r['is_correct'] for r in results)
    accuracy = correct / total if total > 0 else 0
    
    # Task type breakdown
    task_stats = {}
    for task in set(r['task_type'] for r in results):
        task_results = [r for r in results if r['task_type'] == task]
        task_correct = sum(r['is_correct'] for r in task_results)
        task_stats[task] = {
            'total': len(task_results),
            'correct': task_correct,
            'accuracy': task_correct / len(task_results) if task_results else 0
        }
    
    # Duration breakdown
    duration_stats = {}
    for duration in ['short', 'medium', 'long']:
        dur_results = [r for r in results if r['duration'] == duration]
        dur_correct = sum(r['is_correct'] for r in dur_results)
        duration_stats[duration] = {
            'total': len(dur_results),
            'correct': dur_correct,
            'accuracy': dur_correct / len(dur_results) if dur_results else 0
        }
    
    output = {
        'level': 'level1_baseline_rag',
        'config': {
            'k': 10,
            'fps': 1,
            'chunk_size': 3,
            'embedding': 'middle_frame_only'
        },
        'partial': partial,
        'total_questions': total,
        'correct': correct,
        'accuracy': accuracy,
        'avg_retrieval_time': float(np.mean([r['retrieval_time'] for r in results])) if results else 0,
        'avg_inference_time': float(np.mean([r['inference_time'] for r in results])) if results else 0,
        'total_time': total_time,
        'task_type_stats': task_stats,
        'duration_stats': duration_stats,
        'results': results
    }
    
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    if not partial:
        print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"Avg Retrieval Time: {output['avg_retrieval_time']:.2f}s")
        print(f"Avg Inference Time: {output['avg_inference_time']:.2f}s")
        
        print("\nBy Task Type:")
        for task, stats in sorted(task_stats.items(), key=lambda x: -x[1]['accuracy']):
            print(f"  {task}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        print("\nBy Duration:")
        for dur in ['short', 'medium', 'long']:
            stats = duration_stats[dur]
            print(f"  {dur}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, required=True, 
                       choices=['baseline', 'semantic', 'ekg', 'triview', 'agentic', 'consistency'])
    parser.add_argument('--precompute-embeddings', action='store_true',
                       help='Precompute question embeddings for semantic evaluation')
    args = parser.parse_args()
    
    if args.level == 'baseline':
        print("Running Level 1: Baseline RAG Evaluation")
        results = run_baseline_evaluation()
    elif args.level == 'semantic':
        if args.precompute_embeddings:
            print("Precomputing question embeddings for semantic evaluation...")
            precompute_question_embeddings()
            print("Embeddings precomputed. Now run without --precompute-embeddings flag to start evaluation.")
        else:
            print("Running Level 2: Semantic RAG Evaluation")
            results = run_semantic_evaluation()
    else:
        print(f"Level {args.level} not yet implemented")
        return
        
if __name__ == "__main__":
    main()