# JudgeFlow

JudgeFlow is a framework for evaluating and analyzing AI model outputs with a focus on responsible AI practices.

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
- **Context-rich prompts:** Self-reflection prompts now include all relevant context (question, answer, etc.) and instruct the LLM to output a revised score as 'Revised score: X'.
- **Robust parsing:** The system first looks for 'Revised score: X' in the LLM output, then falls back to the first number 0-10.

#### Score Table Schema
- `row_id`: Which test case was evaluated
- `metric`: Which evaluation metric was applied
- `score`: The initial evaluation score (0-10 scale)
- `revised_score`: The LLM's revised score after self-reflection (optional)
- `revision_delta`: The difference between revised and initial score (optional)
- `critique`: The LLM's self-critique or explanation (optional)
- `timestamp`: When the evaluation was performed

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

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Set your OpenAI API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-key-here"

# Unix/Linux
export OPENAI_API_KEY="your-key-here"
```

3. Create metric specifications in the `metrics/` directory using YAML format.

## Usage

### Running Evaluations

1. Prepare your dataset in parquet format with columns:
   - id: unique identifier for each row
   - question: the input question/prompt
   - answer: the model's response to evaluate
   - context: (optional) additional context

2. Run the evaluation:
```bash
python -m src.judgeflow.cli --dataset your_dataset.parquet --quick
```

3. View the results:
```bash
python -m src.judgeflow.view_results
```

The results will show:
- Row ID: Which test case was evaluated
- Metric: Which evaluation metric was applied
- Score: The initial evaluation score (0-10 scale)
- Revised Score: The LLM's revised score after self-reflection (if available)
- Revision Delta: The difference between revised and initial score (if available)
- Critique: The LLM's self-critique or explanation (if available)
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
- Poetry for dependency management
- Async/await pattern for API interactions
- YAML for metric specifications