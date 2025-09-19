import json
import os
from pathlib import Path

# Look for the most recent results file
results_dir = Path("causal_results")
if results_dir.exists():
    json_files = list(results_dir.glob("causal_research_results_*.json"))
    if json_files:
        latest_file = max(json_files, key=os.path.getctime)

        with open(latest_file, 'r') as f:
            saved_results = json.load(f)

        # Extract successful event details
        successful_events = saved_results.get('research_results', {})
        for event_name, event_data in successful_events.items():
            print(f"\n🎯 Successful Event: {event_name}")

            causal_rels = event_data.get('causal_relationships', [])
            print(f"Causal Features ({len(causal_rels)}):")

            for rel in causal_rels:
                feature = rel.get('cause_feature')
                strength = rel.get('strength')
                print(f"  • {feature}: {strength:.4f}")


