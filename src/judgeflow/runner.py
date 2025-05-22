import asyncio
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import csv
import re
import os

from .metrics import load_registry, MetricSpec
from .llm import chat

class Runner:
    def __init__(self, csv_path: str = "scores.csv"):
        self.csv_path = csv_path
        self.metrics = load_registry()
    
    async def evaluate_dataset(self, dataset_path: Path, quick: bool = False) -> List[Dict]:
        """Evaluate a dataset using all registered metrics."""
        df = pd.read_parquet(dataset_path)
        if quick:
            df = df.head(3)  # Only process 3 rows in quick mode
        
        scores = []
        for _, row in df.iterrows():
            row_scores = await self.evaluate_row(row)
            scores.extend(row_scores)
        
        # Write all scores to CSV
        self._write_scores_to_csv(scores)
        
        return scores
    
    async def evaluate_row(self, row: Dict[Any, Any]) -> List[Dict]:
        """Evaluate a single row using all metrics."""
        scores = []
        for metric in self.metrics:
            score = await self._apply_metric(row, metric)
            scores.append(score)
        return scores
    
    async def _apply_metric(self, row: Dict[Any, Any], metric: MetricSpec) -> Dict:
        """Apply a single metric to a row."""
        # Format the prompt template with row data
        prompt = metric.prompt_template.format(**row)
        
        # Get response from LLM
        response = await chat(prompt)
        
        # Parse the score using metric's parser
        try:
            score_value = metric.parse_score(response)
        except ValueError as e:
            print(f"Warning: Failed to parse score for metric {metric.name}: {e}")
            score_value = 0.0  # Default score on parsing failure

        # --- Self-reflection step ---
        # Prepare the reflection prompt (inject score and row fields)
        reflection_context = dict(row)
        reflection_context['score'] = score_value
        reflection_prompt = metric.reflection_prompt.format(**reflection_context)
        # Enforce 256-token limit (approx 1024 chars, conservative)
        max_chars = 1024
        if len(reflection_prompt) > max_chars:
            reflection_prompt = reflection_prompt[:max_chars]
        
        # Get critique and revised score from LLM
        reflection_response = await chat(reflection_prompt)
        # Try to parse revised score
        revised_score = None
        # First, look for 'Revised score: X' (case-insensitive)
        match = re.search(r"Revised score[:\s]*([0-9]+(?:\.[0-9]+)?)", reflection_response, re.IGNORECASE)
        if match:
            try:
                revised_score = float(match.group(1))
            except Exception:
                revised_score = None
        # Fallback: first number 0-10 in the response
        if revised_score is None:
            match = re.search(r"\b([0-9](?:\.[0-9]+)?|10(?:\.0+)?)\b", reflection_response)
            if match:
                try:
                    revised_score = float(match.group(1))
                except Exception:
                    revised_score = None
        # Calculate revision_delta
        revision_delta = None
        if revised_score is not None:
            revision_delta = revised_score - score_value
        # Store critique (full reflection response)
        critique = reflection_response
        
        # --- Confidence metrics ---
        # 1. Self-reported confidence
        confidence_context = dict(row)
        confidence_context['score'] = score_value
        confidence_prompt = metric.confidence_prompt.format(**confidence_context)
        self_conf = None
        try:
            conf_response = await chat(confidence_prompt)
            conf_match = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)", conf_response)
            if conf_match:
                self_conf = float(conf_match.group(1))
                if self_conf > 100:
                    self_conf = 100.0
                elif self_conf < 0:
                    self_conf = 0.0
        except Exception as e:
            print(f"Warning: Failed to get self_conf for metric {metric.name}: {e}")
            self_conf = None

        # 2. Agreement % from 3 resamples (simulate seeds 21, 42, 84)
        resample_scores = []
        for _ in [21, 42, 84]:
            try:
                resample_response = await chat(prompt)
                resample_score = metric.parse_score(resample_response)
                resample_scores.append(resample_score)
            except Exception as e:
                print(f"Warning: Resample failed for metric {metric.name}: {e}")
        agree_count = sum(1 for s in resample_scores if abs(s - score_value) <= 1.0)
        agree_conf = (agree_count / 3) * 100 if resample_scores else None

        return {
            "row_id": str(row.get("id", "unknown")),
            "metric": metric.name,
            "score": score_value,
            "revised_score": revised_score,
            "revision_delta": revision_delta,
            "critique": critique,
            "self_conf": self_conf,
            "agree_conf": agree_conf
        }

    def _write_scores_to_csv(self, scores: List[Dict]):
        # Write header if file does not exist
        write_header = not Path(self.csv_path).exists()
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "row_id", "metric", "score", "revised_score", "revision_delta", "critique", "self_conf", "agree_conf", "timestamp"
                ])
            for score in scores:
                writer.writerow([
                    score["row_id"],
                    score["metric"],
                    score["score"],
                    score["revised_score"] if score["revised_score"] is not None else "",
                    score["revision_delta"] if score["revision_delta"] is not None else "",
                    score["critique"] if score["critique"] is not None else "",
                    score["self_conf"] if score["self_conf"] is not None else "",
                    score["agree_conf"] if score["agree_conf"] is not None else "",
                    score["timestamp"].isoformat() if score["timestamp"] else ""
                ])

def run_deepeval_coherence(dataset_path: Path):
    """Run DeepEval G-Eval for the coherence metric on each row in the dataset."""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError:
        print("DeepEval is not installed. Please install it with 'pip install deepeval'.")
        return []

    deepeval_key = os.environ.get("DEEPEVAL_KEY")
    if not deepeval_key:
        print("DEEPEVAL_KEY environment variable is not set.")
        return []

    import pandas as pd
    df = pd.read_parquet(dataset_path)
    results = []

    # Find the coherence metric definition
    from .metrics import load_registry
    metrics = load_registry()
    coherence_metric = next((m for m in metrics if m.name.lower() == "coherence"), None)
    if not coherence_metric:
        print("Coherence metric not found in registry.")
        return []

    # Prepare G-Eval metric
    coherence_geval = GEval(
        name="Coherence",
        criteria="Measures how well-structured, logical, and easy to follow the model's output is.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.0  # No threshold for reporting
    )

    for _, row in df.iterrows():
        # Use the same fields as in the prompt_template
        question = row.get("question", "")
        answer = row.get("answer", "")
        row_id = row.get("id", "unknown")
        test_case = LLMTestCase(
            input=question,
            actual_output=answer
        )
        try:
            coherence_geval.measure(test_case)
            score = coherence_geval.score
        except Exception as e:
            score = None
            print(f"DeepEval error for row {row_id}: {e}")
        results.append((row_id, score))
    return results

def write_geval_scores_to_csv(csv_path: str, geval_scores):
    """Append G-Eval coherence scores to the CSV output."""
    write_header = not Path(csv_path).exists()
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "row_id", "metric", "score", "revised_score", "revision_delta", "critique", "self_conf", "agree_conf", "timestamp"
            ])
        for row_id, score in geval_scores:
            writer.writerow([
                row_id,
                "coherence_g_eval",
                score if score is not None else "",
                "", "", "", "", "", ""
            ]) 