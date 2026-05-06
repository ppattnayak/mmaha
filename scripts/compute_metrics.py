#!/usr/bin/env python
import os
import json
import argparse

import pandas as pd


# Judge keys coming from judge_scores[…]
JUDGE_KEYS = [
    "SH",
    "SeH",
    "AH",
    "RH",
    "H_any",
    "Comp_correct",
    # "Issues_grounded",  # groundedness is exploratory; not used in main metrics
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory containing per-model result folders.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="judged_outputs_vision.jsonl",
        help="Filename for judged outputs inside each model folder.",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="metrics_gpt5_vision",
        help="Prefix for output CSV files.",
    )
    return ap.parse_args()


def load_all_records(results_root: str, pattern: str) -> pd.DataFrame:
    rows = []

    for model_dir in sorted(os.listdir(results_root)):
        model_path = os.path.join(results_root, model_dir)
        if not os.path.isdir(model_path):
            continue

        judged_path = os.path.join(model_path, pattern)
        if not os.path.exists(judged_path):
            print(f"[WARN] Missing {pattern} in {model_dir}, skipping.")
            continue

        print(f"[INFO] Loading {judged_path}")
        with open(judged_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                judge_scores = rec.get("judge_scores", {})
                if not judge_scores:
                    continue

                # Example: {"openai.gpt-5": {...}}
                judge_id, scores = next(iter(judge_scores.items()))

                row = {
                    "model_folder": model_dir,
                    "model_id": rec.get("model_id"),
                    "image_id": rec.get("image_id"),
                    "platform": rec.get("platform"),
                    "ui_type": rec.get("ui_type"),
                    "prompt_id": rec.get("prompt_id"),
                    "judge_id": judge_id,
                    "parse_error": 0,
                }

                if isinstance(scores, dict) and "parse_error" in scores:
                    # keep row, mark parse_error; metrics remain NaN
                    row["parse_error"] = 1
                else:
                    for k in JUDGE_KEYS:
                        row[k] = scores.get(k, None)

                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[INFO] Loaded {len(df)} judged rows.")
    return df


def add_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cast metrics to numeric
    for k in JUDGE_KEYS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    # Derived “any” flags (for convenience)
    if "SH" in df.columns:
        df["SH_any"] = (df["SH"] > 0).astype("float")
    else:
        df["SH_any"] = float("nan")

    if "SeH" in df.columns:
        df["SeH_any"] = (df["SeH"] > 0).astype("float")
    else:
        df["SeH_any"] = float("nan")

    if "AH" in df.columns:
        df["AH_any"] = (df["AH"] > 0).astype("float")
    else:
        df["AH_any"] = float("nan")

    if "RH" in df.columns:
        df["RH_any"] = (df["RH"] > 0).astype("float")
    else:
        df["RH_any"] = float("nan")

    return df


def summarize(df: pd.DataFrame, group_cols, label: str, out_prefix: str):
    metric_cols = [
        "SH",
        "SeH",
        "AH",
        "RH",
        "H_any",
        "Comp_correct",
        # "Issues_grounded",
        "SH_any",
        "SeH_any",
        "AH_any",
        "RH_any",
    ]

    # Only keep those actually present
    metric_cols = [c for c in metric_cols if c in df.columns]

    agg = df.groupby(group_cols)[metric_cols].mean().reset_index()
    out_path = f"{out_prefix}_{label}.csv"
    agg.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")


# ---------- NEW METRICS BELOW ----------

def compute_cpc_per_model(df: pd.DataFrame, out_prefix: str):
    """
    Cross-Prompt Consistency (CPC):
    For each model + image, check if H_any is consistent across prompts.
    CPC = fraction of images where H_any is constant across prompts.
    """
    if "H_any" not in df.columns:
        print("[INFO] No H_any column, skipping CPC by model.")
        return

    df_h = df.dropna(subset=["H_any"]).copy()

    g = df_h.groupby(["model_folder", "image_id"])["H_any"]
    consistency = g.agg(["min", "max"]).reset_index()
    consistency["consistent"] = (consistency["min"] == consistency["max"]).astype("float")

    cpc = consistency.groupby("model_folder")["consistent"].agg(
        CPC_mean="mean",
        n_images="size",
    ).reset_index()

    out_path = f"{out_prefix}_cpc_by_model.csv"
    cpc.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")


def compute_cpc_per_model_platform(df: pd.DataFrame, out_prefix: str):
    """
    Cross-Prompt Consistency per model × platform.
    """
    if "H_any" not in df.columns:
        print("[INFO] No H_any column, skipping CPC by model+platform.")
        return

    df_h = df.dropna(subset=["H_any"]).copy()

    g = df_h.groupby(["model_folder", "platform", "image_id"])["H_any"]
    consistency = g.agg(["min", "max"]).reset_index()
    consistency["consistent"] = (consistency["min"] == consistency["max"]).astype("float")

    cpc = consistency.groupby(["model_folder", "platform"])["consistent"].agg(
        CPC_mean="mean",
        n_images="size",
    ).reset_index()

    out_path = f"{out_prefix}_cpc_by_model_platform.csv"
    cpc.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")


def compute_cross_model_disagreement(df: pd.DataFrame, out_prefix: str):
    """
    Cross-Model Disagreement (CMD):
    For each (image, platform, prompt), compute variance of H_any across models.
    Then average per (platform, prompt) to see where models disagree most.
    """
    if "H_any" not in df.columns:
        print("[INFO] No H_any column, skipping CMD.")
        return

    df_h = df.dropna(subset=["H_any"]).copy()

    g = df_h.groupby(["platform", "prompt_id", "image_id"])["H_any"]
    per_item = g.var(ddof=0).reset_index(name="H_var")

    cmd = per_item.groupby(["platform", "prompt_id"])["H_var"].mean().reset_index()
    out_path = f"{out_prefix}_cmd_by_platform_prompt.csv"
    cmd.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")


def compute_platform_gap(df: pd.DataFrame, out_prefix: str):
    """
    Platform Sensitivity Gap:
    For each model, compute mean H_any for mobile vs web,
    and their difference.
    """
    if "H_any" not in df.columns:
        print("[INFO] No H_any column, skipping platform gap.")
        return

    df_h = df.dropna(subset=["H_any"]).copy()

    agg = df_h.groupby(["model_folder", "platform"])["H_any"].mean().reset_index()
    pivot = agg.pivot(index="model_folder", columns="platform", values="H_any").reset_index()

    for col in ["mobile", "web"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")

    pivot["platform_gap_web_minus_mobile"] = pivot["web"] - pivot["mobile"]
    out_path = f"{out_prefix}_platform_gap_by_model.csv"
    pivot.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")


def compute_prompt_sensitivity(df: pd.DataFrame, out_prefix: str):
    """
    Prompt Sensitivity / Robustness:
    For each model, compute variance of mean H_any across prompts.
    Lower variance => more robust to prompt wording.
    """
    if "H_any" not in df.columns:
        print("[INFO] No H_any column, skipping prompt sensitivity.")
        return

    df_h = df.dropna(subset=["H_any"]).copy()

    mp = df_h.groupby(["model_folder", "prompt_id"])["H_any"].mean().reset_index()

    def _agg_prompt_stats(group: pd.DataFrame):
        vals = group["H_any"].values
        if len(vals) == 0:
            return pd.Series({"prompt_var": float("nan"), "prompt_range": float("nan")})
        return pd.Series(
            {
                "prompt_var": float(vals.var(ddof=0)),
                "prompt_range": float(vals.max() - vals.min()),
            }
        )

    stats = mp.groupby("model_folder").apply(_agg_prompt_stats).reset_index()
    out_path = f"{out_prefix}_prompt_sensitivity_by_model.csv"
    stats.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")


def compute_parse_error_rate(df: pd.DataFrame, out_prefix: str):
    """
    Per-model judge parse_error rate.
    """
    if "parse_error" not in df.columns:
        print("[INFO] No parse_error column, skipping parse_error_rate.")
        return

    per_model = (
        df.groupby("model_folder")["parse_error"]
        .agg(parse_error_rate="mean", n_rows="size")
        .reset_index()
    )
    out_path = f"{out_prefix}_parse_error_rate_by_model.csv"
    per_model.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")


def main():
    args = parse_args()

    df = load_all_records(args.results_root, args.pattern)
    if df.empty:
        print("[WARN] No data loaded, exiting.")
        return

    print("[INFO] parse_error count:", df["parse_error"].sum())

    df = add_rate_columns(df)

    # ---------- primary summaries ----------
    summarize(df, ["model_folder"], "by_model", args.out_prefix)
    summarize(df, ["model_folder", "prompt_id"], "by_model_prompt", args.out_prefix)
    summarize(df, ["model_folder", "platform"], "by_model_platform", args.out_prefix)
    summarize(
        df,
        ["model_folder", "platform", "prompt_id"],
        "by_model_platform_prompt",
        args.out_prefix,
    )

    # ---------- new metrics ----------
    compute_cpc_per_model(df, args.out_prefix)
    compute_cpc_per_model_platform(df, args.out_prefix)
    compute_cross_model_disagreement(df, args.out_prefix)
    compute_platform_gap(df, args.out_prefix)
    compute_prompt_sensitivity(df, args.out_prefix)
    compute_parse_error_rate(df, args.out_prefix)


if __name__ == "__main__":
    main()
