from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

OUT_DIR = "data/images/web"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading HF dataset silatus/1k_Website_Screenshots_and_Metadata…")
ds = load_dataset("silatus/1k_Website_Screenshots_and_Metadata", split="train")

LIMIT = None  # or set to e.g. 1000 if you want fewer

n = len(ds) if LIMIT is None else min(LIMIT, len(ds))
print(f"Saving {n} screenshots to {OUT_DIR}…")

for i, rec in enumerate(tqdm(ds)):
    if LIMIT is not None and i >= LIMIT:
        break

    img = rec["image"]          # this is already a PIL Image
    img = img.convert("RGB")
    img.save(f"{OUT_DIR}/web_{i:04d}.jpg")
