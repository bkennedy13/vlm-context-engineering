from baseline.baseline_rag import VideoRAGBaseline
from baseline.vlm_answerer import VLMAnswerer
from baseline.video_utils import load_video_mme_dataset, get_short_visual_samples
from data.video_manager import VideoManager
from pathlib import Path

def test_with_sample():
    # Load Video-MME dataset
    print("Loading Video-MME dataset...")
    dataset = load_video_mme_dataset()
    
    if not dataset:
        print("Could not load dataset")
        return
    
    # Get some short visual samples
    samples = get_short_visual_samples(dataset, n_samples=2)
    
    if not samples:
        print("No suitable samples found")
        return
    
    # Use the first sample
    sample = samples[0]
    print(f"Selected sample: {sample['video_id']} - {sample['task_type']}")
    
    # Download the video
    video_manager = VideoManager()
    video_path = video_manager.get_video(sample['youtube_id'])
    
    if not video_path:
        print("Could not download video")
        return
    
    # Initialize models
    rag = VideoRAGBaseline()
    vlm = VLMAnswerer()
    
    # Process video and retrieve relevant chunks
    question = sample['question']
    chunks, similarities = rag.process_video(video_path, question, k=3)
    
    if chunks is not None:
        print(f"\nQuestion: {question}")
        print(f"Options: {sample['options']}")
        print(f"Correct answer: {sample['answer']}")
        print(f"Retrieved {len(chunks)} relevant chunks with similarities: {similarities}")
        
        # Get VLM answer
        print("\nQuerying VLM...")
        predicted_answer = vlm.answer_question(chunks, question, sample['options'])
        
        print(f"\nVLM Predicted: {predicted_answer}")
        print(f"Correct Answer: {sample['answer']}")
        print(f"Result: {'✓ CORRECT' if predicted_answer == sample['answer'] else '✗ INCORRECT'}")
    else:
        print("No chunks retrieved")

if __name__ == "__main__":
    test_with_sample()