# config.py

from dataclasses import dataclass
from typing import List

# ---------- DATASET CONFIG ----------

DATA_MANIFEST_PATH = "data/ui_manifest_2k.csv"
# columns expected:
#   image_id, platform, ui_type, image_path
#
# platform ∈ {"mobile", "web"}
# ui_type ∈ {"form", "list", "dashboard", "dialog", "card"}

OUTPUT_DIR = "results/gpt4o"

# ---------- PROMPTS ----------

PROMPTS = {
    "P1": (
        "Provide a concise, screen-reader-friendly alt-text description "
        "of this interface. Focus only on elements that are visually present. "
        "Do not infer user intent, brand, or personal details. "
        "Do not assume functionality that is not explicitly visible."
    ),
    "P2": (
        "Generate a detailed alt-text description of this UI for a blind user. "
        "Mention visible layout, groups, labels, controls, icons, and any text. "
        "Describe only what is visible. Avoid speculation about actions, "
        "app purpose, or hidden features."
    ),
    "P3": (
        "Identify any accessibility issues visible in this UI. Analyze only "
        "what can be directly observed. Common issues include missing labels, "
        "unreadable text, insufficient contrast, tiny targets, overlapping "
        "elements, or ambiguous icons. Do not guess user flows, backend "
        "behavior, or app domain."
    ),
    "P4": (
        "Based on the visible content of this UI, list any potential "
        "accessibility issues relevant to WCAG 2.1 guidelines "
        "(e.g., contrast, labeling, focus indicators, text alternatives). "
        "Evaluate only what is visually present. Do not cite or fabricate "
        "WCAG rules or numeric thresholds that are not relevant to the "
        "visible UI."
    ),
    "P5": (
        "Does this interface appear to meet basic WCAG 2.1 accessibility "
        "expectations based solely on what is visible? Answer 'Yes' or 'No' "
        "and provide a short justification. Do not assume invisible metadata, "
        "keyboard behavior, or backend functionality. Do not invent WCAG rules "
        "or numeric values."
    ),
    "P6": (
        "Summarize the most critical accessibility risks visible in this UI "
        "for a user with low vision or blindness. Prioritize risks that are "
        "directly supported by the image. Do not assume app domain or user "
        "goals. Do not infer functions or controls that are not clearly shown."
    ),
}

# ---------- MODELS UNDER TEST ----------

# MODEL_IDS = [
#     "<id>"
# ]
#openai.gpt-4o
#openai.gpt-4o-mini
#openai.gpt-5
#openai.gpt-5-mini
#openai.gpt-5-nano

MODEL_IDS = [
    "openai.gpt-4o",   
]


# gpt 5 judge
# JUDGE_MODELS = [
#     ""
# ]

# JUDGE_MODELS = [
#     "openai.gpt-5.2"
# ]
JUDGE_MODELS = [
    "xai.grok-4"
]
# ---------- RUBRIC PROMPT TEMPLATE ----------

# JUDGE_SYSTEM_PROMPT = """
# You are an evaluation assistant that ONLY returns strict JSON.

# You will be given:
# - An accessibility prompt about a user interface (UI).
# - A mobile or web UI screenshot.
# - A model's answer to that prompt.

# You must:
# 1. Read the prompt and the answer.
# 2. Identify hallucinations along four axes:
#    - SH (Structural Hallucination): Invented UI elements, buttons, fields, icons, sections.
#    - SeH (Semantic Hallucination): Incorrect meaning or function assigned to visible elements.
#    - AH (Accessibility Hallucination): Accessibility problems or benefits claimed that are not supported by the screenshot.
#    - RH (Regulatory Hallucination): Fabricated or clearly incorrect references to WCAG or similar standards, including imaginary numeric thresholds.

# Each axis must be scored as:
# - 0 = none or negligible
# - 1 = mild or partial
# - 2 = clear or severe

# You must also output:
# - H_any: 1 if ANY hallucination is present on SH/SeH/AH/RH, else 0.
# - Comp_correct: 1 if the compliance judgement (if present) is plausible and grounded, else 0. If no compliance judgement is present, set Comp_correct to 0.
# - Issues_grounded: 1 if most accessibility issues mentioned are plausibly grounded in the UI description, else 0.

# Return STRICTLY a single minified JSON object with EXACTLY these keys:
# {
#   "SH": <int>,
#   "SeH": <int>,
#   "AH": <int>,
#   "RH": <int>,
#   "H_any": <int>,
#   "Comp_correct": <int>,
#   "Issues_grounded": <int>
# }

# Do NOT include any explanation, natural language, or formatting around the JSON.
# """

JUDGE_SYSTEM_PROMPT = """
You are an evaluation assistant that ONLY returns strict JSON.

You will be given:
- An accessibility prompt about a user interface (UI).
- The corresponding UI screenshot.
- A model's answer to that prompt.

Your task is to assess whether the model's answer contains hallucinations
relative to the visible screenshot.

Evaluation principles:
- Ground all decisions strictly in what is visibly present in the screenshot.
- Do not penalize minor wording differences or reasonable abstraction.
- Only mark hallucination when a claim clearly contradicts or is unsupported by visible evidence.
- Absence of visible evidence should count as hallucination only if the model asserts specific, concrete details.

Identify hallucinations along four axes:

- SH (Structural Hallucination):
  Invented UI elements, controls, icons, sections, or layout regions that are not visible.

- SeH (Semantic Hallucination):
  Incorrect function or meaning assigned to visible elements that is not visually supported.

- AH (Attributional Hallucination):
  Unsupported claims about user intent, app purpose, interaction state, hidden functionality, off-screen context, user goals, or emotional state.

- RH (Rule-Based Hallucination):
  Fabricated or clearly incorrect references to WCAG or similar standards,
  including invented numeric thresholds or specific rule violations not inferable from the screenshot.

Each axis must be scored as:
0 = none
1 = mild or partially unsupported
2 = clear or severe hallucination

You must also output:
- H_any: 1 if ANY hallucination is present (any axis > 0), else 0.
- Comp_correct: 1 if a compliance judgement (if present) is visually grounded and internally consistent, else 0. If no compliance judgement appears, set to 0.
- Issues_grounded: 1 if most accessibility issues mentioned are directly supported by visible evidence, else 0.

Return STRICTLY a single minified JSON object with EXACTLY these keys:
{
  "SH": <int>,
  "SeH": <int>,
  "AH": <int>,
  "RH": <int>,
  "H_any": <int>,
  "Comp_correct": <int>,
  "Issues_grounded": <int>
}

Do NOT include explanation or any additional text.
"""


# ---------- SIMPLE DATA CLASS FOR OUTPUT ROW ----------

@dataclass
class ModelOutput:
    image_id: str
    platform: str
    ui_type: str
    model_id: str
    prompt_id: str
    prompt_text: str
    response_text: str
