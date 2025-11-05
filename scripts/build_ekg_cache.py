"""
Build and cache EKGs for all videos in eval subset.
This creates event knowledge graphs with entities and relationships
for efficient retrieval during Level 3 evaluation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pickle
from tqdm import tqdm
import torch

from ekg.event_merger import EventMerger
from ekg.entity_extractor import EntityExtractor
from ekg.entity_linker import EntityLinker
from ekg.graph_builder import GraphBuilder


def build_ekg_for_video(youtube_id, semantic_cache_dir, merger, extractor, linker, builder):
    """Build complete EKG for a single video using pre-loaded models."""
    
    # Load semantic cache
    cache_file = semantic_cache_dir / f"{youtube_id}.pkl"
    if not cache_file.exists():
        print(f"  WARNING: No semantic cache found for {youtube_id}")
        return None
    
    # Step 1: Create events from semantic chunks
    events = merger.process_video(youtube_id, cache_file)
    
    if not events:
        print(f"  WARNING: No events created for {youtube_id}")
        return None
    
    # Step 2: Extract entities from events
    events_with_entities = extractor.process_events(events)
    
    # Step 3: Link entities across events
    unique_entities, entity_map = linker.link_entities(events_with_entities)
    
    # Step 4: Build relationships
    relationships = builder.build_relationships(
        events_with_entities,
        unique_entities,
        entity_map
    )
    
    # Package complete EKG
    ekg_data = {
        'youtube_id': youtube_id,
        'events': events_with_entities,
        'entities': unique_entities,
        'entity_map': entity_map,
        'relationships': relationships,
        'stats': {
            'num_events': len(events_with_entities),
            'num_entities': len(unique_entities),
            'num_relationships': len(relationships),
            'relationship_types': {
                'NEXT': len([r for r in relationships if r['type'] == 'NEXT']),
                'APPEARS_IN': len([r for r in relationships if r['type'] == 'APPEARS_IN']),
                'CO_OCCURS': len([r for r in relationships if r['type'] == 'CO_OCCURS'])
            }
        }
    }
    
    return ekg_data


def build_all_ekgs(eval_subset_path='data/eval_subset.json',
                   semantic_cache_dir='data/semantic_cache_level2',
                   ekg_output_dir='data/ekg_cache'):
    """Build EKGs for all videos in eval subset."""
    
    # Setup directories
    semantic_cache_dir = Path(semantic_cache_dir)
    ekg_output_dir = Path(ekg_output_dir)
    ekg_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load eval subset
    print("Loading evaluation subset...")
    with open(eval_subset_path, 'r') as f:
        eval_samples = json.load(f)
    
    # Get unique youtube IDs
    youtube_ids = sorted(set(s['youtube_id'] for s in eval_samples))
    print(f"Found {len(youtube_ids)} unique videos")
    print(f"Semantic cache: {semantic_cache_dir}")
    print(f"EKG output: {ekg_output_dir}")
    print()
    
    # Initialize models ONCE
    print("Initializing models...")
    merger = EventMerger(similarity_threshold=0.75)
    extractor = EntityExtractor(batch_size=8)
    linker = EntityLinker(similarity_threshold=0.85)
    builder = GraphBuilder()
    print("Models loaded")
    print()
    
    # Build EKG for each video
    results = []
    failed = []
    
    for i, youtube_id in enumerate(tqdm(youtube_ids, desc="Building EKGs")):
        try:
            # Check if already exists
            output_file = ekg_output_dir / f"{youtube_id}_ekg.pkl"
            if output_file.exists():
                print(f"\n[{i+1}/{len(youtube_ids)}] {youtube_id}: Already cached, skipping")
                continue
            
            print(f"\n[{i+1}/{len(youtube_ids)}] Processing {youtube_id}...")
            ekg_data = build_ekg_for_video(
                youtube_id, 
                semantic_cache_dir, 
                merger, 
                extractor, 
                linker, 
                builder
            )
            
            if ekg_data:
                stats = ekg_data['stats']
                print(f"  Events: {stats['num_events']}")
                print(f"  Entities: {stats['num_entities']}")
                print(f"  Relationships: {stats['num_relationships']}")
                print(f"    NEXT: {stats['relationship_types']['NEXT']}")
                print(f"    APPEARS_IN: {stats['relationship_types']['APPEARS_IN']}")
                print(f"    CO_OCCURS: {stats['relationship_types']['CO_OCCURS']}")
                
                # Save EKG cache
                with open(output_file, 'wb') as f:
                    pickle.dump(ekg_data, f)
                
                results.append({'youtube_id': youtube_id, 'stats': stats})
            else:
                failed.append(youtube_id)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(youtube_id)
            continue
    
    # Summary
    print("\n" + "="*60)
    print("EKG CACHE BUILD COMPLETE")
    print("="*60)
    print(f"Successful: {len(results)}/{len(youtube_ids)}")
    print(f"Failed: {len(failed)}/{len(youtube_ids)}")
    
    if failed:
        print(f"\nFailed videos: {failed}")
    
    # Save summary
    summary = {
        'total_videos': len(youtube_ids),
        'successful': len(results),
        'failed': len(failed),
        'failed_videos': failed,
        'video_stats': results
    }
    
    summary_file = ekg_output_dir / 'build_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Overall statistics
    if results:
        avg_events = sum(r['stats']['num_events'] for r in results) / len(results)
        avg_entities = sum(r['stats']['num_entities'] for r in results) / len(results)
        avg_relationships = sum(r['stats']['num_relationships'] for r in results) / len(results)
        
        print(f"\nAverage statistics per video:")
        print(f"  Events: {avg_events:.1f}")
        print(f"  Entities: {avg_entities:.1f}")
        print(f"  Relationships: {avg_relationships:.1f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build EKG cache for all videos')
    parser.add_argument('--eval-subset', type=str, default='data/eval_subset.json',
                       help='Path to eval subset JSON')
    parser.add_argument('--semantic-cache', type=str, default='data/semantic_cache_level2',
                       help='Directory containing semantic cache files')
    parser.add_argument('--output', type=str, default='data/ekg_cache',
                       help='Output directory for EKG cache files')
    
    args = parser.parse_args()
    
    build_all_ekgs(
        eval_subset_path=args.eval_subset,
        semantic_cache_dir=args.semantic_cache,
        ekg_output_dir=args.output
    )