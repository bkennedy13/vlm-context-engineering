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
    args = parser.parse_args()
    
    if args.level == 'baseline':
        print("Running Level 1: Baseline RAG Evaluation")
        from baseline.baseline_rag import VideoRAGBaseline
        # run baseline eval
        results = run_baseline_evaluation()
        
if __name__ == "__main__":
    main()