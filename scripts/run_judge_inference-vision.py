# run_judge_inference-vision.py

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

from config import JUDGE_MODELS, JUDGE_SYSTEM_PROMPT
from oci_genai_client import call_judge_model

# ---- CONFIG ----
OUTPUT_DIR = "results/"
MANIFEST_PATH = "data/ui_manifest_2k.csv"

# If you run from repo root, PROJECT_ROOT=os.getcwd() is fine.
# If you run from elsewhere: export MMAHA_ROOT=/path/to/repo
PROJECT_ROOT = os.getenv("~/<user>/mmaha", os.getcwd())

# Global manifest map shared across threads
manifest_map: Dict[str, str] = {}

# Thread debug counters
_active_lock = threading.Lock()
_active_jobs = 0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional max number of records to judge per folder (smoke test).",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel threads to use for judge inference.",
    )
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="grok_vision",
        help="Suffix for output file name (default: vision).",
    )
    return ap.parse_args()


def build_judge_user_content(model_rec: Dict[str, Any]) -> str:
    return (
        "ACCESSIBILITY PROMPT:\n"
        f"{model_rec.get('prompt_text','')}\n\n"
        "MODEL ANSWER:\n"
        f"{model_rec.get('response_text','')}\n"
    )


def load_manifest_map(manifest_csv: str) -> Dict[str, str]:
    """
    Returns {image_id: image_path} from manifest.
    Expected columns: image_id, image_path.
    Paths are typically repo-relative like: data/images/web/web_0431.jpg
    """
    m: Dict[str, str] = {}
    if not os.path.exists(manifest_csv):
        print(f"[WARN] manifest not found: {manifest_csv}")
        return m

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id")
            image_path = row.get("image_path")
            if image_id and image_path:
                m[str(image_id)] = str(image_path)
    return m


def resolve_image_path(model_rec: Dict[str, Any], manifest_map_: Dict[str, str]) -> str | None:
    """
    Resolve absolute path to the screenshot for vision-aware judging.

    Priority:
      1) model_rec["image_path"] if exists
      2) manifest_map[image_id]

    Handles relative paths by joining with PROJECT_ROOT.
    """
    # (1) from record
    p = model_rec.get("image_path")
    if p:
        if os.path.isabs(p) and os.path.exists(p):
            return p
        cand = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(cand):
            return cand

    # (2) from manifest
    image_id = model_rec.get("image_id")
    if image_id and image_id in manifest_map_:
        p2 = manifest_map_[image_id]
        if os.path.isabs(p2) and os.path.exists(p2):
            return p2
        cand2 = os.path.join(PROJECT_ROOT, p2)
        if os.path.exists(cand2):
            return cand2

    return None


def load_model_outputs(input_file: str, max_records: int | None = None) -> List[Dict[str, Any]]:
    """
    Loads model outputs (jsonl). Skips any records where response_text contains '__ERROR__'.
    """
    records: List[Dict[str, Any]] = []
    if not os.path.exists(input_file):
        return records

    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "__ERROR__" in rec.get("response_text", ""):
                continue
            records.append(rec)

    return records


def run_one_judge_job(model_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all judge models for a single model output record.
    Returns enriched record with 'judge_scores' attached.
    """
    global _active_jobs

    image_id = model_rec.get("image_id", "<unknown_image>")
    model_id = model_rec.get("model_id", "<unknown_model>")
    prompt_id = model_rec.get("prompt_id", "<unknown_prompt>")

    start_ts = time.time()
    with _active_lock:
        _active_jobs += 1
        active_now = _active_jobs

    print(
        f"[DEBUG] START judging image={image_id} model={model_id} prompt={prompt_id} "
        f"active_jobs={active_now} t={start_ts:.2f}",
        flush=True,
    )

    try:
        user_content = build_judge_user_content(model_rec)

        # Resolve screenshot path
        img_path = resolve_image_path(model_rec, manifest_map)
        if img_path is None:
            model_rec["judge_scores"] = {
                jm: {"parse_error": "missing_image", "raw": f"image_id={image_id}"}
                for jm in JUDGE_MODELS
            }
            return model_rec

        judge_results: Dict[str, Any] = {}
        for judge_model_id in JUDGE_MODELS:
            try:
                scores = call_judge_model(
                    judge_model_id=judge_model_id,
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_content=user_content,
                    image_path=img_path,  # vision-aware
                )
            except Exception as e:
                scores = {
                    "SH": -1,
                    "SeH": -1,
                    "AH": -1,
                    "RH": -1,
                    "H_any": -1,
                    "Comp_correct": -1,
                    "Issues_grounded": -1,
                    "error": str(e),
                }
            judge_results[judge_model_id] = scores

        model_rec["judge_scores"] = judge_results
        return model_rec

    finally:
        end_ts = time.time()
        with _active_lock:
            _active_jobs -= 1
            active_now = _active_jobs

        print(
            f"[DEBUG] DONE  judging image={image_id} model={model_id} prompt={prompt_id} "
            f"duration={end_ts - start_ts:.2f}s active_jobs={active_now} t={end_ts:.2f}",
            flush=True,
        )


def main():
    args = parse_args()

    # Load manifest once
    global manifest_map
    manifest_map = load_manifest_map(MANIFEST_PATH)
    print(f"[INFO] Loaded manifest entries: {len(manifest_map)}")
    print(f"[INFO] PROJECT_ROOT={PROJECT_ROOT}")

    print(f"Scanning result folders under: {OUTPUT_DIR}")

    subfolders = [
        d for d in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ]
    subfolders.sort()

    if not subfolders:
        print("No subfolders found under OUTPUT_DIR. Nothing to judge.")
        return

    for folder in subfolders:
        folder_path = os.path.join(OUTPUT_DIR, folder)
        input_file = os.path.join(folder_path, "model_outputs.jsonl")
        output_file = os.path.join(folder_path, f"judged_outputs_{args.output_suffix}.jsonl")

        if not os.path.exists(input_file):
            print(f"[WARN] Skipping folder '{folder}' (no model_outputs.jsonl).")
            continue

        print(f"\n=== Processing folder: {folder} ===")
        records = load_model_outputs(input_file, max_records=args.max_records)
        total_jobs = len(records)
        print(f"Total judge jobs in '{folder}': {total_jobs}")

        if total_jobs == 0:
            print(f"[WARN] No records loaded from {input_file}. Skipping.")
            continue

        with open(output_file, "w", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                futures = [ex.submit(run_one_judge_job, rec) for rec in records]

                for fut in tqdm(as_completed(futures), total=total_jobs, desc=f"Judging {folder}"):
                    rec_with_scores = fut.result()
                    f_out.write(json.dumps(rec_with_scores, ensure_ascii=False) + "\n")
                    f_out.flush()

        print(f"Saved judged outputs to {output_file}")

    print("\nAll folders processed.")


if __name__ == "__main__":
    main()
