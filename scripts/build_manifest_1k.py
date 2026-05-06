# build_manifest_1k.py

import os
import random
import pandas as pd

MOBILE_DIR = "data/images/mobile"
WEB_DIR = "data/images/web"
OUT_CSV = "data/ui_manifest_2k.csv"

N_MOBILE = 1000
N_WEB = 1000

random.seed(42)

def list_images(root):
    return sorted(
        f for f in os.listdir(root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

def main():
    mobile_files = list_images(MOBILE_DIR)
    web_files = list_images(WEB_DIR)

    if len(mobile_files) < N_MOBILE:
        raise ValueError(f"Not enough mobile images: {len(mobile_files)} < {N_MOBILE}")
    if len(web_files) < N_WEB:
        raise ValueError(f"Not enough web images: {len(web_files)} < {N_WEB}")

    mobile_sample = random.sample(mobile_files, N_MOBILE)
    web_sample = random.sample(web_files, N_WEB)

    rows = []

    for fname in mobile_sample:
        image_id = os.path.splitext(fname)[0]
        rows.append({
            "image_id": image_id,
            "platform": "mobile",
            "ui_type": "unknown",   # we'll tag later if we want
            "image_path": os.path.join(MOBILE_DIR, fname),
        })

    for fname in web_sample:
        image_id = os.path.splitext(fname)[0]
        rows.append({
            "image_id": image_id,
            "platform": "web",
            "ui_type": "unknown",
            "image_path": os.path.join(WEB_DIR, fname),
        })

    df = pd.DataFrame(rows)
    # shuffle so mobile/web are mixed
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
