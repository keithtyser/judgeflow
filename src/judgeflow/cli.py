import asyncio
import typer
from pathlib import Path

from .runner import Runner

def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset parquet file"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Run quick evaluation on subset of data"),
    csv_path: str = typer.Option("scores.csv", "--csv", help="Path to output CSV file")
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
    except Exception as e:
        typer.echo(f"Error during evaluation: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main) 