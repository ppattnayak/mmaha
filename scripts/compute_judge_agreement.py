#!/usr/bin/env python
import argparse
import pandas as pd


METRIC_COLS = [
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpt5-file",
        type=str,
        default="metrics_gpt5_by_model.csv",
        help="GPT-5 judge metrics file (by_model).",
    )
    ap.add_argument(
        "--grok4-file",
        type=str,
        default="metrics_grok4_by_model.csv",
        help="Grok-4 judge metrics file (by_model).",
    )
    ap.add_argument(
        "--out-merged",
        type=str,
        default="judge_agreement_merged_by_model.csv",
        help="Output CSV with merged GPT-5 + Grok-4 metrics per model.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    gpt5 = pd.read_csv(args.gpt5_file)
    grok4 = pd.read_csv(args.grok4_file)

    # Expect common key: model_folder
    if "model_folder" not in gpt5.columns or "model_folder" not in grok4.columns:
        raise ValueError("Both CSVs must have a 'model_folder' column.")

    # Prefix columns to distinguish judges
    gpt5_prefixed = gpt5.copy()
    grok4_prefixed = grok4.copy()

    for c in METRIC_COLS:
        if c in gpt5_prefixed.columns:
            gpt5_prefixed.rename(columns={c: f"{c}_gpt5"}, inplace=True)
        if c in grok4_prefixed.columns:
            grok4_prefixed.rename(columns={c: f"{c}_grok4"}, inplace=True)

    merged = pd.merge(
        gpt5_prefixed,
        grok4_prefixed,
        on="model_folder",
        suffixes=("_gpt5", "_grok4"),
        how="inner",
    )

    merged.to_csv(args.out_merged, index=False)
    print(f"[INFO] Wrote merged per-model metrics to {args.out_merged}")
    print()

    # Compute Kendall tau and Spearman for each metric present in both
    print("=== Judge Agreement (per metric, across models) ===")
    for base in METRIC_COLS:
        col_gpt5 = f"{base}_gpt5"
        col_grok4 = f"{base}_grok4"
        if col_gpt5 not in merged.columns or col_grok4 not in merged.columns:
            continue

        x = merged[col_gpt5]
        y = merged[col_grok4]

        # Drop rows with NaN
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            continue

        # pandas can compute Kendall and Spearman directly
        kendall_tau = x.corr(y, method="kendall")
        spearman_rho = x.corr(y, method="spearman")

        print(
            f"Metric: {base:15s} | "
            f"Kendall τ = {kendall_tau: .3f} | "
            f"Spearman ρ = {spearman_rho: .3f} | "
            f"n_models = {len(x)}"
        )


if __name__ == "__main__":
    main()
