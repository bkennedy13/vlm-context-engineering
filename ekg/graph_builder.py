import json
from pathlib import Path


class GraphBuilder:
    """Build entity knowledge graph with all relationships."""
    
    def build_relationships(self, events, entities, entity_map):
        """
        Build all relationships in the graph.
        
        Returns:
            List of relationship dicts: {source, target, type, metadata}
        """
        relationships = []
        
        # 1. Event-to-Event: temporal NEXT relationships
        events_by_video = {}
        for event in events:
            video_id = event['video_id']
            if video_id not in events_by_video:
                events_by_video[video_id] = []
            events_by_video[video_id].append(event)
        
        for video_id, video_events in events_by_video.items():
            # Sort by first frame
            video_events.sort(key=lambda e: min(e['frame_ids']))
            
            for i in range(len(video_events) - 1):
                relationships.append({
                    'source': video_events[i]['event_id'],
                    'target': video_events[i + 1]['event_id'],
                    'type': 'NEXT',
                    'source_type': 'event',
                    'target_type': 'event'
                })
        
        # 2. Entity-to-Event: APPEARS_IN relationships
        for event in events:
            event_id = event['event_id']
            entities_dict = event.get('entities', {})
            
            for entity_type, entity_list in entities_dict.items():
                for entity_text in entity_list:
                    entity_text = entity_text.lower().strip()
                    if entity_text in entity_map:
                        entity_id = entity_map[entity_text]
                        relationships.append({
                            'source': entity_id,
                            'target': event_id,
                            'type': 'APPEARS_IN',
                            'source_type': 'entity',
                            'target_type': 'event'
                        })
        
        # 3. Entity-to-Entity: CO_OCCURS relationships
        # Entities co-occur if they appear in the same event
        entity_cooccurrence = {}
        
        for event in events:
            event_entities = set()
            entities_dict = event.get('entities', {})
            
            for entity_type, entity_list in entities_dict.items():
                for entity_text in entity_list:
                    entity_text = entity_text.lower().strip()
                    if entity_text in entity_map:
                        event_entities.add(entity_map[entity_text])
            
            # Create co-occurrence pairs
            event_entities = sorted(list(event_entities))
            for i, e1 in enumerate(event_entities):
                for e2 in event_entities[i+1:]:
                    pair = (e1, e2)
                    entity_cooccurrence[pair] = entity_cooccurrence.get(pair, 0) + 1
        
        # Add co-occurrence relationships (threshold: appear together at least once)
        for (e1, e2), count in entity_cooccurrence.items():
            relationships.append({
                'source': e1,
                'target': e2,
                'type': 'CO_OCCURS',
                'source_type': 'entity',
                'target_type': 'entity',
                'count': count
            })
        
        return relationships
    
    def build_graph(self, events_path, entities_path):
        """Build complete EKG."""
        # Load events and entities
        with open(events_path, 'r') as f:
            events_data = json.load(f)
        events = events_data['events']
        
        with open(entities_path, 'r') as f:
            entities_data = json.load(f)
        entities = entities_data['entities']
        
        # Build entity map
        entity_map = {}
        for entity in entities:
            for variant in entity['variants']:
                entity_map[variant] = entity['entity_id']
        
        # Build relationships
        print(f"Building relationships...")
        relationships = self.build_relationships(events, entities, entity_map)
        
        print(f"  Event→Event: {sum(1 for r in relationships if r['type'] == 'NEXT')}")
        print(f"  Entity→Event: {sum(1 for r in relationships if r['type'] == 'APPEARS_IN')}")
        print(f"  Entity→Entity: {sum(1 for r in relationships if r['type'] == 'CO_OCCURS')}")
        
        return relationships


def build_ekg(events_path='data/ekg/events_with_entities.json',
             entities_path='data/ekg/entities.json',
             relationships_output='data/ekg/relationships.json'):
    """Build and save complete EKG."""
    events_path = Path(events_path)
    entities_path = Path(entities_path)
    relationships_output = Path(relationships_output)
    
    builder = GraphBuilder()
    relationships = builder.build_graph(events_path, entities_path)
    
    # Save relationships
    with open(relationships_output, 'w') as f:
        json.dump({
            'total_relationships': len(relationships),
            'relationships': relationships
        }, f, indent=2)
    
    print(f"\nSaved {len(relationships)} relationships to {relationships_output}")
    print("\nEKG construction complete!")
    print(f"  Events: {events_path}")
    print(f"  Entities: {entities_path}")
    print(f"  Relationships: {relationships_output}")


if __name__ == '__main__':
    build_ekg()