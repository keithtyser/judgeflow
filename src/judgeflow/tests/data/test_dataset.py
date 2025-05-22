import pandas as pd
from pathlib import Path

# Create sample data
data = [
    {
        "id": "test1",
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
        "context": "A question about French geography."
    },
    {
        "id": "test2",
        "question": "How does photosynthesis work?",
        "answer": "Photosynthesis is the process where plants convert sunlight into energy, using water and carbon dioxide to produce glucose and oxygen.",
        "context": "A science question about plant biology."
    },
    {
        "id": "test3",
        "question": "What caused World War I?",
        "answer": "World War I was triggered by the assassination of Archduke Franz Ferdinand, but underlying causes included nationalism, militarism, and complex alliances.",
        "context": "A history question about the causes of WWI."
    }
]

def create_test_dataset():
    # Create the directory if it doesn't exist
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame and save as parquet
    df = pd.DataFrame(data)
    output_path = output_dir / "test_dataset.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Created test dataset at: {output_path}")

if __name__ == "__main__":
    create_test_dataset() 