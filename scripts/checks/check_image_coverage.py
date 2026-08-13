import os
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.config import load_config

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml"

cfg = load_config(CONFIG_PATH)
data_cfg = cfg["data"]

metadata_path = os.path.join(data_cfg["data_dir"], data_cfg["metadata_csv"])
meta = pd.read_csv(metadata_path)
image_dirs = [os.path.join(data_cfg["data_dir"], d) for d in data_cfg["image_dirs"]]

print(f"Config: {CONFIG_PATH}")
print(f"data_dir: {data_cfg['data_dir']}")
print(f"metadata rows: {len(meta)}")
for d in image_dirs:
    exists = os.path.isdir(d)
    n_files = len(os.listdir(d)) if exists else 0
    print(f"  {d}: exists={exists} file_count={n_files}")

# what extensions actually exist on disk, per dir
print("\nExtension counts on disk per dir:")
dir_files = {}
for d in image_dirs:
    if not os.path.isdir(d):
        continue
    files = os.listdir(d)
    dir_files[d] = files
    ext_counts = Counter(os.path.splitext(f)[1].lower() for f in files)
    print(f"  {d}: {dict(ext_counts)}")

# exact-match check (what the current code does: <data_dir>/<image_dir>/<image_id>.jpg)
found_jpg = 0
missing_ids = []
for image_id in meta["image_id"]:
    hit = any(os.path.exists(os.path.join(d, image_id + ".jpg")) for d in image_dirs)
    if hit:
        found_jpg += 1
    else:
        missing_ids.append(image_id)

print(f"\nExact '<image_id>.jpg' matches: {found_jpg}/{len(meta)}")
print(f"Missing: {len(missing_ids)}")

if missing_ids:
    print("\nFirst 10 missing image_ids:")
    for image_id in missing_ids[:10]:
        print(f"  {image_id}")

    # for each missing id, see if ANY file with that stem exists (any extension/case/suffix)
    print("\nFor missing ids, checking for near-matches (same stem, any extension) on disk:")
    all_stems = {}
    for d, files in dir_files.items():
        for f in files:
            stem, ext = os.path.splitext(f)
            all_stems.setdefault(stem, []).append((d, f))

    near_match_count = 0
    for image_id in missing_ids[:20]:
        candidates = [v for k, v in all_stems.items() if image_id.lower() in k.lower() or k.lower() in image_id.lower()]
        if candidates:
            near_match_count += 1
            flat = [c for group in candidates for c in group]
            print(f"  {image_id} -> near matches: {flat[:3]}")
    print(f"\n{near_match_count}/{min(20, len(missing_ids))} sampled missing ids had a near-match on disk "
          f"(different extension/case/suffix) — if this is high, it's a naming-convention mismatch, not truly missing files.")
