# run_local_model_inference.py

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from config import DATA_MANIFEST_PATH, PROMPTS, MODEL_IDS, ModelOutput
from local_client import call_vision_model

# config.py

MODEL_IDS = [
    "llama-3_2-11b-vision",  # must match --served-model-name
]
OUTPUT_DIR = "results/llama32_11b/"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional max number of images to process (for smoke test).",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel threads to use for inference.",
    )
    return ap.parse_args()


def build_jobs(df):
    """
    Expand dataframe rows into (image_id, platform, ui_type, image_path, model_id, prompt_id, prompt_text).
    """
    jobs = []
    for _, row in df.iterrows():
        image_id = row["image_id"]
        platform = row["platform"]
        ui_type = row["ui_type"]
        image_path = row["image_path"]

        for model_id in MODEL_IDS:
            for prompt_id, prompt_text in PROMPTS.items():
                jobs.append(
                    (
                        image_id,
                        platform,
                        ui_type,
                        image_path,
                        model_id,
                        prompt_id,
                        prompt_text,
                    )
                )
    return jobs


def run_one_job(job):
    (
        image_id,
        platform,
        ui_type,
        image_path,
        model_id,
        prompt_id,
        prompt_text,
    ) = job

    # Optional: per-job debug print
    print(
        f"[DEBUG] image={image_id} model={model_id} prompt={prompt_id} ...",
        flush=True,
    )

    try:
        response_text = call_vision_model(
            model_id=model_id,
            image_path=image_path,
            prompt=prompt_text,
        )
    except Exception as e:
        response_text = f"__ERROR__: {type(e).__name__}: {e}"

    print(
        f"[DEBUG] ...done image={image_id} model={model_id} prompt={prompt_id}",
        flush=True,
    )

    rec = ModelOutput(
        image_id=image_id,
        platform=platform,
        ui_type=ui_type,
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        response_text=response_text,
    )
    # Return JSON-serializable dict
    return rec.__dict__


def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_MANIFEST_PATH)
    if args.max_images is not None:
        df = df.head(args.max_images)

    jobs = build_jobs(df)
    total_jobs = len(jobs)
    print(f"Total jobs (image × model × prompt): {total_jobs}")

    out_path = os.path.join(OUTPUT_DIR, "model_outputs.jsonl")

    with open(out_path, "w", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(run_one_job, job): job for job in jobs}

            for fut in tqdm(as_completed(futures), total=total_jobs, desc="Jobs"):
                rec_dict = fut.result()
                f_out.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                f_out.flush()

    print(f"Saved model outputs to {out_path}")


if __name__ == "__main__":
    main()
