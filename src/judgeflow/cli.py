import asyncio
import typer
from pathlib import Path

from .runner import Runner

def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset parquet file"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Run quick evaluation on subset of data"),
    csv_path: str = typer.Option("scores.csv", "--csv", help="Path to output CSV file"),
    deepeval: bool = typer.Option(False, "--deepeval", help="Run DeepEval G-Eval for coherence metric")
):
    """Run evaluation on a dataset using all registered metrics."""
    runner = Runner(csv_path=csv_path)
    dataset_path = Path(dataset)
    
    if not dataset_path.exists():
        typer.echo(f"Error: Dataset file {dataset_path} not found")
        raise typer.Exit(1)
        
    try:
        scores = asyncio.run(runner.evaluate_dataset(dataset_path, quick=quick))
        typer.echo(f"Successfully evaluated {len(scores)} metric-row combinations")
        if deepeval:
            # Run DeepEval G-Eval for coherence
            from .runner import run_deepeval_coherence, write_geval_scores_to_csv
            g_eval_scores = asyncio.run(run_deepeval_coherence(dataset_path))
            typer.echo("DeepEval G-Eval scores for coherence:")
            for row_id, score in g_eval_scores:
                typer.echo(f"Row {row_id}: {score}")
            # Write G-Eval scores to CSV
            write_geval_scores_to_csv(csv_path, g_eval_scores)
    except Exception as e:
        typer.echo(f"Error during evaluation: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main) 