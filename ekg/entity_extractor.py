"""Extract entities with batched inference (faster than parallel models)."""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from tqdm import tqdm


class EntityExtractor:
    """Extract entities using batched inference."""
    
    def __init__(self, model_name='Qwen/Qwen3-4B-Instruct-2507', batch_size=8):
        #'Qwen/Qwen2.5-1.5B-Instruct'
        """
        Load LLM for entity extraction.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Number of events to process at once (default 8)
        """
        self.batch_size = batch_size
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.model.eval()
        print("Model loaded successfully")
    
    def extract_entities_batch(self, descriptions):
        """
        Extract entities from multiple descriptions at once.
        
        Args:
            descriptions: List of description strings
            
        Returns:
            List of entity dicts
        """
        # Truncate descriptions
        descriptions = [d[:800] + "..." if len(d) > 800 else d for d in descriptions]
        
        # Create prompts
        prompts = []
        for desc in descriptions:
            prompt = f"""Extract key entities from this video description. Return ONLY a JSON object.

Description: {desc}

Format (lists of strings only):
{{"objects": ["item1", "item2"], "actions": ["action1"], "locations": ["place1"], "attributes": ["attr1"]}}

JSON:"""
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # Generate batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode all outputs
        results = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[len(inputs.input_ids[i]):], 
                skip_special_tokens=True
            )
            entities = self._parse_entities(response)
            results.append(entities)
        
        return results
    
    def _parse_entities(self, response):
        """Parse entities from model response."""
        try:
            # Extract JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            if '{' in response:
                start = response.find('{')
                brace_count = 0
                end = start
                for i, char in enumerate(response[start:], start=start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                if end > start:
                    response = response[start:end]
                else:
                    response = response[start:]
                    open_braces = response.count('{') - response.count('}')
                    open_brackets = response.count('[') - response.count(']')
                    response += ']' * open_brackets + '}' * open_braces
            
            entities = json.loads(response)
            
            # Validate
            for key in ['objects', 'actions', 'locations', 'attributes']:
                if key not in entities:
                    entities[key] = []
                elif not isinstance(entities[key], list):
                    entities[key] = []
                else:
                    flattened = []
                    for item in entities[key]:
                        if isinstance(item, str):
                            flattened.append(item)
                        elif isinstance(item, dict):
                            if 'name' in item:
                                flattened.append(item['name'])
                    entities[key] = flattened
            
            return entities
            
        except:
            # Regex fallback
            entities = {"objects": [], "actions": [], "locations": [], "attributes": []}
            for key in entities.keys():
                pattern = rf'"{key}"\s*:\s*\[(.*?)\]'
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    items = re.findall(r'"([^"]*)"', match.group(1))
                    entities[key] = items
            
            return entities if any(entities.values()) else {
                "objects": [], "actions": [], "locations": [], "attributes": []
            }
    
    def process_events(self, events):
        """
        Extract entities from all events using batched inference.
        
        Args:
            events: List of event dicts
            
        Returns:
            events with added 'entities' field
        """
        print(f"Extracting entities from {len(events)} events (batch_size={self.batch_size})...")
        
        # Process in batches
        for i in range(0, len(events), self.batch_size):
            batch = events[i:i + self.batch_size]
            descriptions = [e['description'] for e in batch]
            
            # Extract entities for batch
            batch_entities = self.extract_entities_batch(descriptions)
            
            # Add to events
            for event, entities in zip(batch, batch_entities):
                event['entities'] = entities
            
            if i % (self.batch_size * 5) == 0:
                print(f"  Progress: {i}/{len(events)}")
        
        print(f"  Progress: {len(events)}/{len(events)}")
        return events


def extract_all_entities(events_path='data/ekg/events.json', 
                        output_path='data/ekg/events_with_entities.json',
                        batch_size=8):
    """
    Extract entities from all events with batched inference.
    
    Args:
        batch_size: Number of events per batch (8-16 recommended for L4)
    """
    events_path = Path(events_path)
    output_path = Path(output_path)
    
    with open(events_path, 'r') as f:
        data = json.load(f)
    
    events = data['events']
    
    extractor = EntityExtractor(batch_size=batch_size)
    events = extractor.process_events(events)
    
    with open(output_path, 'w') as f:
        json.dump({
            'total_events': len(events),
            'events': events
        }, f, indent=2)
    
    print(f"\nSaved events with entities to {output_path}")
    return events


if __name__ == '__main__':
    extract_all_entities(batch_size=16)