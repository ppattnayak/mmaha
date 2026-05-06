# Multi-AHA Benchmark

This repository contains the data and scripts used to reproduce the results in the
**Multi-AHA** paper submitted to the NeurIPS 2026 Evaluations & Datasets track.

Multi-AHA is a benchmark for studying **accessibility-related hallucinations**
in multimodal large language models (MLLMs) when they process real UI screenshots.

The benchmark includes:

- 2,000 real UI screenshots (mobile + web)
- 6 accessibility-focused prompts (P1–P6)
- Dual-judge hallucination evaluation
- Taxonomy-based hallucination metrics
- Scripts for inference, evaluation, and analysis

---

# Repository Structure

```text
mmaha/
├── data/
│   ├── images/
│   │   ├── mobile/
│   │   └── web/
│   └── ui_manifest_2k.csv
│
├── scripts/
│   ├── build_manifest_1k.py
│   ├── run_local_model_inference.py
│   ├── run_judge_inference-vision.py
│   ├── compute_metrics.py
│   ├── compute_judge_agreement.py
│   ├── local_client.py
│   └── config.py
│
└── README.md
```

---

# Overview

Multi-AHA evaluates hallucinations in multimodal accessibility reasoning across:

- Alt-text generation
- Accessibility issue detection
- WCAG-style reasoning
- Binary compliance judgments
- Severity summaries

The benchmark measures four hallucination categories:

| Metric | Description |
|---|---|
| SH | Structural hallucination |
| SeH | Semantic hallucination |
| AH | Attributional hallucination |
| RH | Rule-based hallucination |

Additional metrics:

| Metric | Description |
|---|---|
| H_any | Any hallucination occurred |
| Comp_correct@P5 | P5 compliance consistency |
| CPI | Cross-prompt inconsistency |
| CMD | Cross-model disagreement |

---

# Dataset

The benchmark combines screenshots from:

- **RICO** mobile UI dataset
- **Silatus Web UI** dataset

## Image Layout

Place screenshots in:

```text
data/images/mobile/
data/images/web/
```

The repository expects PNG/JPG screenshots.

---

# Installation

## Python Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install pandas numpy requests tqdm
```

Optional (for local VLM inference):

```bash
pip install vllm
```

---

# Step 1 — Build Manifest

Generate a benchmark manifest from the image folders:

```bash
python scripts/build_manifest_1k.py
```

This creates:

```text
data/ui_manifest_2k.csv
```

The manifest contains:

| Column | Description |
|---|---|
| image_id | Unique image ID |
| platform | mobile/web |
| ui_type | placeholder field |
| image_path | relative image path |

The script samples:

- 1000 mobile screenshots
- 1000 web screenshots

using deterministic random sampling.

---

# Step 2 — Configure Models and Prompts

Edit:

```text
scripts/config.py
```

Key configuration fields:

## Prompts

`PROMPTS` defines the six benchmark tasks:

| Prompt | Task |
|---|---|
| P1 | Concise alt-text |
| P2 | Detailed alt-text |
| P3 | Accessibility issue listing |
| P4 | WCAG-style reasoning |
| P5 | Binary compliance judgment |
| P6 | Severity summary |

## Models

`MODEL_IDS` specifies evaluated models.

Example:

```python
MODEL_IDS = [
    "openai:gpt-4o",
]
```

## Judge Models

`JUDGE_MODELS` specifies evaluation judges.

Example:

```python
JUDGE_MODELS = [
    "xai.grok-4",
    "openai:gpt-5.2",
]
```

---

# Step 3 — Run Model Inference

## Local vLLM Inference

Start a vLLM OpenAI-compatible server first.

Example:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-11B-Vision \
    --served-model-name llama-3-2-11b-vision
```

Then run:

```bash
python scripts/run_local_model_inference.py \
    --max-images 100 \
    --num-workers 8
```

Outputs are saved to:

```text
results/<model_name>/model_outputs.jsonl
```

Each line contains:

```json
{
  "image_id": "...",
  "platform": "...",
  "prompt_id": "...",
  "response_text": "..."
}
```

---

# Step 4 — Judge Model Outputs

Run hallucination evaluation:

```bash
python scripts/run_judge_inference-vision.py
```

This script:

- Loads model outputs
- Sends them to judge models
- Produces structured hallucination labels

Output format:

```json
{
  "SH": 1,
  "SeH": 0,
  "AH": 2,
  "RH": 0,
  "H_any": 1,
  "Comp_correct": 0,
  "Issues_grounded": 1
}
```

Judge outputs are stored in:

```text
results/<model>/judged_outputs_vision.jsonl
```

---

# Step 5 — Compute Metrics

Aggregate benchmark metrics:

```bash
python scripts/compute_metrics.py
```

Generated outputs include:

| File | Description |
|---|---|
| *_by_model.csv | Overall model metrics |
| *_by_model_prompt.csv | Prompt-wise metrics |
| *_by_model_platform.csv | Platform-wise metrics |
| *_cpc_by_model.csv | Cross-prompt consistency |
| *_cmd_by_platform_prompt.csv | Cross-model disagreement |
| *_platform_gap_by_model.csv | Mobile vs web gaps |

---

# Step 6 — Judge Agreement

If two judges are used (e.g. GPT-5.2 and Grok-4):

```bash
python scripts/compute_judge_agreement.py \
    --gpt5-file metrics_gpt5_by_model.csv \
    --grok4-file metrics_grok4_by_model.csv
```

This computes:

- Kendall τ
- Spearman ρ

for all hallucination metrics.

---

# Hallucination Taxonomy

## Structural Hallucination (SH)

Invented UI regions, controls, icons, or layout structure.

Example:

> “The navigation panel contains missing buttons”

when no navigation panel exists.

---

## Semantic Hallucination (SeH)

Incorrect meaning/function assigned to visible UI elements.

Example:

> “The arrow uploads files”

when the arrow is generic navigation.

---

## Attributional Hallucination (AH)

Unsupported claims about user intent or hidden functionality.

Example:

> “The user is attempting to improve productivity”

without visible evidence.

---

## Rule-Based Hallucination (RH)

Fabricated WCAG rules, thresholds, or compliance claims.

Example:

> “Fails WCAG 1.4.3 with contrast ratio below 4.5:1”

without measurable evidence.

---

# Important Notes

## P5-specific Metric

`Comp_correct@P5` applies ONLY to prompt P5.

It measures whether:

- the binary compliance judgment
- matches the textual justification

It should NOT be averaged across all prompts.

---

## Screenshot-only Benchmark

Multi-AHA evaluates screenshots only.

The benchmark does NOT include:

- DOM trees
- ARIA labels
- keyboard navigation
- interaction traces
- accessibility trees

P5 should therefore be interpreted as:

> a stress test for visible-accessibility reasoning,
> not a full WCAG compliance audit.

---

# Reproducibility

The repository contains:

- prompt templates
- evaluation scripts
- judge prompts
- metric aggregation scripts
- dataset manifest generation
- benchmark screenshots

All benchmark prompts are frozen and model-agnostic.

---

# Compute Notes

The benchmark can be computationally expensive.

For quick experiments:

```bash
--max-images 50
```

is recommended.

---

# Citation

If you use Multi-AHA, please cite:

```bibtex
@inproceedings{multiaha2026,
  title={Multi-AHA: A Multimodal Audit of Accessibility Hallucinations in User Interfaces},
  author={Anonymous Authors},
  booktitle={NeurIPS 2026 Evaluations and Datasets Track},
  year={2026}
}
```

---

# License

This repository is released for academic research purposes only.

Please respect the licenses and terms of use of:

- RICO dataset
- Silatus dataset
- evaluated model providers

---

# Contact

For questions or issues, please open a GitHub issue.
