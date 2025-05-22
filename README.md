# JudgeFlow

JudgeFlow is a framework for evaluating and analyzing AI model outputs with a focus on responsible AI practices.

## Quickstart: Setup & Mini Datasets

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download mini datasets:**
   - Run the script to fetch 20 rows each from TruthfulQA, Jigsaw, and MMLU:
     ```bash
     python download_mini_datasets.py
     ```
   - **Jigsaw Note:** You must manually download the Jigsaw dataset from Kaggle (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), extract it to a folder named `jigsaw_data` in the project root, and then run the script.
   - The datasets will be saved as Parquet files in the `datasets/` directory. Total token count is kept under 8,000 for quick testing.

3. **Set your OpenAI API key:**
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY = "your-key-here"
   # Unix/Linux
   export OPENAI_API_KEY="your-key-here"
   ```

## Core Modules

### Runner Core (`runner.py` & `cli.py`)

The runner core provides functionality to evaluate datasets using multiple metrics and store results in a CSV file.

```bash
# Run evaluation on a dataset
python -m src.judgeflow.cli --dataset path/to/dataset.parquet --quick

# View evaluation results (open scores.csv in Excel, Python, etc.)
```

**Features:**
- Parallel evaluation across multiple metrics
- CSV storage for easy inspection and portability
- Quick evaluation mode for rapid testing
- Structured results with timestamps
- **Self-reflection:** After the initial score, the LLM critiques its own answer and provides a revised score. The difference (`revision_delta`) and the critique are stored in the CSV file.
- **Confidence metrics:** For each score, the LLM self-reports its confidence (0-100), and agreement is measured by resampling the evaluation 3 times and reporting the % of resamples within ±1 point of the original score. These are stored as `self_conf` and `agree_conf` in the CSV.
- **Context-rich prompts:** Self-reflection prompts now include all relevant context (question, answer, etc.) and instruct the LLM to output a revised score as 'Revised score: X'.
- **Robust parsing:** The system first looks for 'Revised score: X' in the LLM output, then falls back to the first number 0-10.

#### Score Table Schema
- `row_id`: Which test case was evaluated
- `metric`: Which evaluation metric was applied
- `score`: The initial evaluation score (0-10 scale)
- `revised_score`: The LLM's revised score after self-reflection (optional)
- `revision_delta`: The difference between revised and initial score (optional)
- `critique`: The LLM's self-critique or explanation (optional)
- `self_conf`: The LLM's self-reported confidence in its initial score (0-100)
- `agree_conf`: The agreement percentage, i.e., the % of 3 resampled scores within ±1 point of the original score
- `timestamp`: When the evaluation was performed

### Confidence Reliability: Calibration Curve

You can generate a reliability diagram (calibration curve) to visualize how well the model's self-reported confidence aligns with actual correctness.

- The script `plot_calibration_curve.py` reads `scores.csv` and plots a calibration curve using scikit-learn and matplotlib.
- It uses the `self_conf` column (normalized to 0-1) as the model's confidence, and treats a score of 8 or higher (out of 10) as "correct".
- The output is a PNG file (default: `calibration_curve.png`).

**Usage:**
```bash
python plot_calibration_curve.py --csv scores.csv --output calibration_curve.png --bin_count 10 --threshold 8
```
- `--csv`: Path to your scores CSV (default: `scores.csv`)
- `--output`: Output PNG file (default: `calibration_curve.png`)
- `--bin_count`: Number of bins for the calibration curve (default: 10)
- `--threshold`: Score threshold for correctness (default: 8.0)

The resulting plot shows how well the model's confidence matches the actual fraction of correct answers. A perfectly calibrated model will follow the diagonal line.

### LLM Adapter (`llm.py`)

The LLM adapter provides a robust interface to GPT-4 with built-in retry functionality and error handling.

```python
from judgeflow.llm import LLMAdapter

# Initialize with default settings (uses GPT-4 Turbo)
llm = LLMAdapter()

# Or specify your own API key and model
llm = LLMAdapter(
    api_key="your-key-here",
    model="gpt-4-turbo-preview"
)

# Make an async chat call
async def example():
    response = await llm.chat([
        {"role": "user", "content": "Your prompt here"}
    ])
    print(response)
```

**Features:**
- Async operation with automatic retries
- Exponential backoff retry strategy (3 attempts)
- Comprehensive error handling and logging
- Environment variable support for API key (`OPENAI_API_KEY`)
- Windows-compatible event loop handling

### Metrics Registry (`metrics.py`)

The metrics module manages evaluation metrics through YAML-based specifications.

```python
from judgeflow.metrics import load_registry, MetricSpec

# Load all metric specifications from a directory
metrics = load_registry("metrics")

# Example metric specification YAML:
"""
name: factuality
description: Measures factual accuracy of responses
prompt_template: Rate the factual accuracy...
parser: float_0_10
rai_category: reliability
reflection_prompt: The initial score for factuality was {score}.\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\nPlease provide a revised score (number only, 0-10) as 'Revised score: X' and a brief critique.
confidence_prompt: How confident are you...
"""
```

**Features:**
- YAML-based metric definitions
- Pydantic model validation
- Structured metric specifications with:
  - Name and description
  - Prompt templates
  - Response parsing rules
  - RAI categorization
  - Reflection and confidence prompts (now context-rich and parseable)
- Robust error handling for YAML parsing and validation

**Tip:**
- For best results, ensure your metric YAMLs' `reflection_prompt` includes all relevant context and instructs the LLM to output a revised score as 'Revised score: X'.
- If you see parsing errors or generic critiques, check that your dataset includes all required fields (e.g., question, answer, context).

## RAI Helper Functions

JudgeFlow now includes standalone Responsible AI (RAI) helper functions for fairness, toxicity, and PII detection. These can be used independently or integrated into your evaluation pipeline.

### Installation

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

### Usage Example

```python
from judgeflow.rai_helpers import fairness_sp_tpr_gap, detoxify_toxicity, detect_pii_spacy_regex

# Fairness (Statistical Parity & TPR gap)
y_true = [1, 0, 1, 0, 1, 0]
y_pred = [1, 0, 0, 0, 1, 1]
sensitive_attr = ['male', 'male', 'female', 'female', 'male', 'female']
fairness_result = fairness_sp_tpr_gap(y_true, y_pred, sensitive_attr)
print("Fairness (SP & TPR gap):", fairness_result)

# Detoxify toxicity
toxic_text = "You are so stupid and ugly!"
toxicity_score = detoxify_toxicity(toxic_text)
print(f"Detoxify toxicity score for '{toxic_text}':", toxicity_score)

# spaCy + regex PII detection
pii_text = "Contact me at john.doe@example.com."
pii_score = detect_pii_spacy_regex(pii_text)
print(f"PII detection for '{pii_text}':", pii_score)
```

**Sample Output:**
```
Fairness (SP & TPR gap): {'sp_gap': 0.33, 'tpr_gap': 1.0}
Detoxify toxicity score for 'You are so stupid and ugly!': 0.99
PII detection for 'Contact me at john.doe@example.com.': 1.0
```

- `fairness_sp_tpr_gap` returns a dictionary with statistical parity and TPR gap between groups.
- `detoxify_toxicity` returns a float toxicity score (0 = not toxic, 1 = highly toxic).
- `detect_pii_spacy_regex` returns 1.0 if PII is detected, 0.0 otherwise.

See `src/judgeflow/rai_helpers.py` for more details and additional examples.

## Setup

- Python 3.11+
- Use `pip` for dependency management (no longer using Poetry)
- Async/await pattern for API interactions
- YAML for metric specifications

## Usage

### Running Evaluations

1. Prepare your dataset in parquet format (see above for mini datasets or use your own)

2. Run the evaluation:
```bash
python -m src.judgeflow.cli --dataset your_dataset.parquet --quick
```

3. View the results:
# Open scores.csv in Excel, Python (pandas), or any spreadsheet tool to inspect the evaluation results.

The results will show:
- Row ID: Which test case was evaluated
- Metric: Which evaluation metric was applied
- Score: The initial evaluation score (0-10 scale)
- Revised Score: The LLM's revised score after self-reflection (if available)
- Revision Delta: The difference between revised and initial score (if available)
- Critique: The LLM's self-critique or explanation (if available)
- Self Conf: The LLM's self-reported confidence (0-100)
- Agree Conf: The agreement % from 3 resamples (within ±1 point)
- Timestamp: When the evaluation was performed

### Available Metrics

The framework includes several pre-configured metrics:
- Factuality: Measures factual accuracy (0-10)
- Coherence: Evaluates text flow and readability (0-10)
- Reasoning: Assesses logical soundness (0-10)
- Toxicity: Checks for harmful content (0-10, lower is better)
- Fairness (Gender): Evaluates gender bias (0-10)
- Privacy (PII Leak): Detects personal information exposure (0-10)

Each metric is defined in YAML format and can be customized or extended.

## Development

- Python 3.11+
- Use `pip` for dependency management (no longer using Poetry)
- Async/await pattern for API interactions
- YAML for metric specifications

## DeepEval G-Eval Integration (Coherence)

JudgeFlow now supports running DeepEval's G-Eval for the "coherence" metric via a CLI flag. This allows you to:
- Run G-Eval for coherence on your dataset with a single flag
- Print G-Eval scores to the console
- Store G-Eval scores in your CSV output as a new metric (`coherence_g_eval`)

### Usage

1. **Install DeepEval:**
   ```bash
   pip install deepeval
   ```
2. **Set your DeepEval API key:**
   ```powershell
   $env:DEEPEVAL_KEY = "your-deepeval-key-here"
   ```
3. **Run evaluation with G-Eval:**
   ```bash
   python -m src.judgeflow.cli --dataset path/to/dataset.parquet --deepeval
   ```

- The CLI will print G-Eval coherence scores for each row.
- The scores will be appended to your CSV output with `metric` set to `coherence_g_eval`.

**Note:**
- The `DEEPEVAL_KEY` environment variable is required for G-Eval.
- G-Eval scores are stored in the same CSV as other metrics, with only `row_id`, `metric`, and `score` filled for these rows.