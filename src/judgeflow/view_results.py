from sqlmodel import Session, select
from tabulate import tabulate
from pathlib import Path
from .models import Score
from .runner import Runner

def view_results(db_path: str = "scores.db"):
    """View evaluation results from the database."""
    runner = Runner(db_path=db_path)
    
    with Session(runner.engine) as session:
        # Query all scores ordered by row_id and metric
        statement = select(Score).order_by(Score.row_id, Score.metric)
        results = session.exec(statement).all()
        
        # Prepare data for tabulation
        table_data = []
        for score in results:
            table_data.append([
                score.row_id,
                score.metric,
                f"{score.score:.2f}",
                score.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            ])
        
        # Print results in a nice table format
        headers = ["Row ID", "Metric", "Score", "Timestamp"]
        print("\nEvaluation Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    view_results() 