import asyncio
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sqlmodel import Session, SQLModel, create_engine

from .models import Score
from .metrics import load_registry, MetricSpec
from .llm import chat

class Runner:
    def __init__(self, db_path: str = "scores.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        SQLModel.metadata.create_all(self.engine)
        self.metrics = load_registry()
    
    async def evaluate_dataset(self, dataset_path: Path, quick: bool = False) -> List[Score]:
        """Evaluate a dataset using all registered metrics."""
        df = pd.read_parquet(dataset_path)
        if quick:
            df = df.head(3)  # Only process 3 rows in quick mode
        
        scores = []
        for _, row in df.iterrows():
            row_scores = await self.evaluate_row(row)
            scores.extend(row_scores)
            
        with Session(self.engine) as session:
            for score in scores:
                session.add(score)
            session.commit()
        
        return scores
    
    async def evaluate_row(self, row: Dict[Any, Any]) -> List[Score]:
        """Evaluate a single row using all metrics."""
        scores = []
        for metric in self.metrics:
            score = await self._apply_metric(row, metric)
            scores.append(score)
        return scores
    
    async def _apply_metric(self, row: Dict[Any, Any], metric: MetricSpec) -> Score:
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
            
        return Score(
            row_id=str(row.get("id", "unknown")),
            metric=metric.name,
            score=score_value
        ) 