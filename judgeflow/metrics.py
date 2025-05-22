import os
from typing import List
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

def load_registry(registry_path: str) -> List[MetricSpec]:
    """
    Loads metric specifications from all YAML files in the given directory.
    """
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
            except Exception as e: # Catches Pydantic validation errors and other issues
                print(f"Error loading metric from {filepath}: {e}")
    return metric_specs

if __name__ == "__main__":
    # This test assumes the 'metrics' directory is located at the project root,
    # and this script is run in a context where "metrics" resolves correctly
    # (e.g., run from the project root: python judgeflow/metrics.py or python -m judgeflow.metrics).
    metrics_directory = "metrics"
    
    # For a more robust path if this script is inside a package, e.g., judgeflow/judgeflow/metrics.py
    # and 'metrics' is at the project root (e.g., judgeflow/metrics/):
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir) # Adjust if nested deeper
    # metrics_directory = os.path.join(project_root, "metrics")
    # However, to keep it simple and aligned with the plan's test:
    
    loaded_metrics = load_registry(metrics_directory)
    # The plan states: "`print(len(load_registry()))` shows 6."
    print(len(loaded_metrics)) 