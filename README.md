# JudgeFlow

JudgeFlow is a framework for evaluating and analyzing AI model outputs with a focus on responsible AI practices.

## Core Modules

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
reflection_prompt: Explain your rating...
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
  - Reflection and confidence prompts
- Robust error handling for YAML parsing and validation

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

```python
# Example combining both modules
from judgeflow.llm import LLMAdapter
from judgeflow.metrics import load_registry

async def evaluate_response():
    llm = LLMAdapter()
    metrics = load_registry("metrics")
    
    for metric in metrics:
        # Evaluate response using each metric
        result = await llm.chat([
            {"role": "user", "content": metric.prompt_template}
        ])
        print(f"{metric.name}: {result}")
```

## Development

- Python 3.11+
- Poetry for dependency management
- Async/await pattern for API interactions
- YAML for metric specifications