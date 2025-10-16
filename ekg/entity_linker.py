import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np


class EntityLinker:
    """Deduplicate entities via hierarchical clustering."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.85):
        """
        Args:
            similarity_threshold: Minimum similarity to link entities
        """
        self.threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model for entity linking")
    
    def link_entities(self, events):
        """
        Find all unique entities across events and deduplicate.
        
        Returns:
            entities: List of unique entity dicts
            entity_map: Dict mapping (event_id, entity_text) -> entity_id
        """
        # Collect all entities with their event IDs
        all_entities = []
        
        for event in events:
            event_id = event['event_id']
            entities_dict = event.get('entities', {})
            
            for entity_type, entity_list in entities_dict.items():
                for entity_text in entity_list:
                    all_entities.append({
                        'text': entity_text.lower().strip(),
                        'type': entity_type,
                        'event_id': event_id
                    })
        
        if not all_entities:
            return [], {}
        
        print(f"Found {len(all_entities)} total entity mentions")
        
        # Get unique entity texts
        unique_texts = list(set([e['text'] for e in all_entities]))
        print(f"  {len(unique_texts)} unique entity strings")
        
        # Embed all unique entities
        embeddings = self.model.encode(unique_texts, show_progress_bar=False)
        
        # Cluster using hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.threshold,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        n_clusters = len(set(cluster_labels))
        
        print(f"  Linked to {n_clusters} unique entities")
        
        # Build unique entities list
        unique_entities = []
        text_to_entity_id = {}
        
        for cluster_id in range(n_clusters):
            # Get all texts in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = [unique_texts[i] for i, m in enumerate(cluster_mask) if m]
            
            # Use most common text as canonical name
            canonical_name = max(set(cluster_texts), key=cluster_texts.count)
            
            # Get all entity types for this cluster
            cluster_types = set()
            cluster_events = set()
            for entity in all_entities:
                if entity['text'] in cluster_texts:
                    cluster_types.add(entity['type'])
                    cluster_events.add(entity['event_id'])
            
            unique_entities.append({
                'entity_id': cluster_id,
                'name': canonical_name,
                'types': list(cluster_types),
                'variants': cluster_texts,
                'appears_in_events': sorted(list(cluster_events)),
                'frequency': len(cluster_events)
            })
            
            # Map all variants to this entity_id
            for text in cluster_texts:
                text_to_entity_id[text] = cluster_id
        
        return unique_entities, text_to_entity_id


def link_all_entities(events_path='data/ekg/events_with_entities.json',
                     entities_output='data/ekg/entities.json'):
    """Link entities across all events."""
    events_path = Path(events_path)
    entities_output = Path(entities_output)
    
    # Load events
    with open(events_path, 'r') as f:
        data = json.load(f)
    
    events = data['events']
    
    # Link entities
    linker = EntityLinker(similarity_threshold=0.85)
    unique_entities, entity_map = linker.link_entities(events)
    
    # Save entities
    with open(entities_output, 'w') as f:
        json.dump({
            'total_entities': len(unique_entities),
            'entities': unique_entities
        }, f, indent=2)
    
    print(f"\nSaved {len(unique_entities)} unique entities to {entities_output}")
    return unique_entities, entity_map


if __name__ == '__main__':
    link_all_entities()