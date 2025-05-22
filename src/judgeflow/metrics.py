import os
import re
from typing import List, Callable, Any
import yaml  # Note: Requires PyYAML to be installed (e.g., poetry add PyYAML)
from pydantic import BaseModel

class MetricSpec(BaseModel):
    name: str
    description: str
    prompt_template: str
    parser: str  # Could be a regex, function name, or configuration key
    rai_category: str
    reflection_prompt: str
    confidence_prompt: str

    def parse_score(self, text: str) -> float:
        """Parse the score from text using the specified parser."""
        if self.parser.startswith("regex:"):
            pattern = self.parser[6:]  # Remove "regex:" prefix
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    raise ValueError(f"Could not parse float from match: {match.group(0)}")
            raise ValueError(f"No match found for pattern: {pattern}")
        raise ValueError(f"Unsupported parser type: {self.parser}")

def load_registry(registry_path: str = None) -> List[MetricSpec]:
    """
    Loads metric specifications from all YAML files in the given directory.
    If no path is provided, looks in the default metrics directory.
    """
    if registry_path is None:
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        registry_path = os.path.join(current_dir, "metrics")

    metric_specs = []
    if not os.path.isdir(registry_path):
        print(f"Warning: Metrics directory '{registry_path}' not found.")
        return metric_specs

    for filename in os.listdir(registry_path):
        if filename.endswith((".yaml", ".yml")):
            filepath = os.path.join(registry_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data:  # Ensure file is not empty and data is not None
                        metric_specs.append(MetricSpec(**data))
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {filepath}: {e}")
            except Exception as e:  # Catches Pydantic validation errors and other issues
                print(f"Error loading metric from {filepath}: {e}")
    
    return metric_specs

if __name__ == "__main__":
    # This test assumes the 'metrics' directory is located at the project root,
    # and this script is run in a context where "metrics" resolves correctly
    # (e.g., run from the project root: python judgeflow/metrics.py or python -m judgeflow.metrics).
    loaded_metrics = load_registry()
    print(f"Loaded {len(loaded_metrics)} metrics:") 