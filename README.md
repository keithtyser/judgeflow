
# JudgeFlow 🚦 – Modular, Responsible LLM Evaluation Framework
> Plug‑in new metrics by dropping a YAML, run one command, get a CSV‑grade‑sheet & reliability diagram.

---

## ✨ Why JudgeFlow?

| JudgeFlow | ↔️ | Existing toolkits (e.g. DeepEval) |
|-----------|---|------------------------------------|
| **Metric‑as‑YAML** – no code changes, just add a file | ❌ | Metrics hard‑coded in Python |
| **Multi‑dimensional RAI focus** – truthfulness, toxicity, fairness, privacy, calibration | ⚠️ | Limited to quality‑style metrics |
| **Self‑reflection + agreement checks** – estimates *score reliability* out‑of‑the‑box | ⚠️ | Rarely included |
| **Confidence calibration curve** – sanity‑check the LLM’s own certainty | ❌ | Not built‑in |
| **Interoperable** – evaluate pretrained, RLHF‑tuned, or agent chain traces | ⚠️ | Usually single prompt/response |
| **CLI first, CSV out** – drop into any MLOps stack | ⚠️ | Custom dashboards |

<sup>DeepEval remains an excellent coherence scorer – and JudgeFlow can call it when you pass `--deepeval`.</sup>

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt                       # 1. deps
python download_mini_datasets.py                      # 2. test on 60‑row toy data
export OPENAI_API_KEY=sk‑...                          # 3. key
python -m src.judgeflow.cli -d datasets/mmlu.parquet  # 4. go!
open scores.csv                                       # 5. inspect
```

Want a one‑liner reliability diagram?

```bash
python plot_calibration_curve.py --csv scores.csv
```

---

## 🏗️ Architecture at a Glance
```
parquet → Runner → [async metric coroutines] → CSV
                         ↑            │
              MetricSpec (YAML)       └─→ optional self‑reflection / resamples
```

* **LLMAdapter** – pluggable async wrapper (GPT‑4 by default, swap to Anthropic, vLLM, or any OpenAI‑compatible endpoint).  
* **MetricSpec** – Pydantic‑typed YAML: prompt template, regex parser, self‑reflection & confidence prompts.  
* **Runner** – reads dataset, spawns per‑row tasks, writes tidy CSV.  
* **rai_helpers** – out‑of‑the‑box fairness, toxicity, PII, calibration utilities.



---

## 🔌 Defining Your Own Metric (30 s)

```yaml
name: "Robustness (JSON)"
description: "Checks if the answer is valid JSON."
prompt_template: |
  Answer:
  {answer}

  Reply 0‑10: how well‑formed & informative is the JSON?
parser: 'regex:(\d+(?:\.\d+)?)'
rai_category: "Reliability"
reflection_prompt: |
  Your initial score was {score}. Give a revised score as 'Revised score: X' plus one‑sentence critique.
confidence_prompt: |
  How confident (0‑100) are you in that score?
```

Save as `src/judgeflow/metrics/robust_json.yaml` and re‑run the CLI – nothing else to code.

---

## 📊 First‑Class Responsible‑AI Metrics

| Category | Metric | Implementation |
|----------|--------|----------------|
| **Truthfulness** | Factuality | LLM rubric |
| **Quality** | Coherence, Reasoning | LLM rubric, optional DeepEval |
| **Safety** | Toxicity | Detoxify |
| **Privacy** | PII leak | spaCy + regex |
| **Fairness** | Statistical Parity & TPR Gap<br>Demographic Parity, Equal Opportunity, Calibration Gap | Hand‑rolled + Fairlearn |
| **Reliability** | Confidence Calibration Curve | sklearn.calibration |

Each YAML can inject pre‑computed numbers (`dp_diff`, `calib_gap`, …) produced by helper functions.

---

## 🔧 Use Cases

| Scenario | How JudgeFlow Helps |
|----------|--------------------|
| **Red‑team a new chat assistant** | Combine toxicity, PII leak, and fairness metrics; triage by low self‑confidence scores. |
| **Compare RLHF vs. base model** | Point Runner at two CSVs, analyse score deltas & calibration. |
| **Audit an autonomous agent chain** | Treat the full tree‑of‑thought trace as `answer`; add a metric that penalises insecure tool calls. |
| **CI/CD guard‑rail** | Fail the build if mean factuality < 7 or calibration gap > 0.05. |

---

## 🛣️ Roadmap – What I’d Build Next

* **Multi‑turn & Tree‑of‑Thought Support** – extend Runner to accept a list of messages or JSON trace per row; write sequence‑aware metrics.  
* **Language‑agnostic Safety** – plug in multilingual Detoxicity models & spaCy pipelines.  
* **Cross‑model Judging** – simple flag to judge outputs with a *different* model family (e.g., GPT‑4 evals Llama‑2).  
* **Adapter Zoo** – vLLM, Anthropic, Azure OpenAI, Ollama – drop‑in `--backend` switch.  
* **Batch Optimization** – smart prompt packing & caching to slash eval cost.  
* **Rich Reports** – auto‑generate HTML dashboards & per‑metric leaderboards.

I spun this prototype up in a day – imagine the velocity with a full‑time seat 🚀.

---

## 📁 Repo Tour

```
├── src/judgeflow/
│   ├── runner.py          # async orchestration
│   ├── llm.py             # pluggable LLM client
│   ├── metrics.py         # YAML registry loader
│   ├── rai_helpers.py     # fairness / safety utils
│   └── metrics/*.yaml     # plug‑and‑play metric specs
├── scores.csv             # sample output
└── plot_calibration_curve.py
```

---

## 🔖 License
MIT – free for personal & commercial use.
