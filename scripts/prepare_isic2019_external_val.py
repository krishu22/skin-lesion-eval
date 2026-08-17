"""
One-time, standalone data-prep script.

Builds an external validation set from the ISIC2019 dataset
("salviohexia/isic-2019-skin-lesion-images-for-classification") for later
evaluation against an already-trained HAM10000 classifier.

Does NOT touch any training/model code or HAM10000 data. Read-only against
existing HAM10000 split CSVs and the ISIC2019 ground-truth CSV; writes a
single new output CSV.

This output is an intermediate build artifact, not the final file the eval
pipeline reads: scripts/filter_isic2019_metadata.py,
scripts/add_isic2019_metadata_features.py, and
scripts/merge_isic2019_metadata_features.py (in that order) consume it and
fold it together with the metadata feature columns into the single final
outputs/isic2019_metadata.csv that scripts/eval_isic2019_external.py uses.

Run once: python scripts/prepare_isic2019_external_val.py
"""

import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ISIC_GT_CSV = REPO_ROOT / "ISIC_metadata" / "ISIC_2019_Training_GroundTruth.csv"
HAM_SPLIT_CSVS = [
    REPO_ROOT / "outputs" / "splits" / "train.csv",
    REPO_ROOT / "outputs" / "splits" / "val.csv",
    REPO_ROOT / "outputs" / "splits" / "test.csv",
]
OUTPUT_CSV = REPO_ROOT / "outputs" / "isic2019_external_val.csv"

# ISIC2019 dataset folder/CSV column name -> HAM10000 dx name.
# Excludes SCC and UNK entirely (no HAM10000 equivalent).
ISIC_TO_DX = {
    "MEL": "mel",
    "NV": "nv",
    "BCC": "bcc",
    "AK": "akiec",
    "BKL": "bkl",
    "DF": "df",
    "VASC": "vasc",
}
EXCLUDED_ISIC_CLASSES = {"SCC", "UNK"}

# Must match configs/data/ham10000.yaml `classes:` list exactly, since that
# list order is what src/data/splits.py uses to build the integer label
# (label = index into this list).
HAM10000_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
DX_TO_LABEL = {dx: i for i, dx in enumerate(HAM10000_CLASSES)}


def load_ham10000_image_ids():
    ids = set()
    for csv_path in HAM_SPLIT_CSVS:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                ids.add(row["image_id"])
    return ids


def main():
    ham_ids = load_ham10000_image_ids()

    total_found = 0
    excluded_scc_unk = 0
    excluded_dedup = 0
    rows = []

    with open(ISIC_GT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_found += 1
            image_id = row["image"]

            # Determine class from whichever one-hot column is 1.0.
            isic_class = None
            for col in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]:
                if float(row[col]) == 1.0:
                    isic_class = col
                    break

            if isic_class in EXCLUDED_ISIC_CLASSES:
                excluded_scc_unk += 1
                continue

            if image_id in ham_ids:
                excluded_dedup += 1
                continue

            dx = ISIC_TO_DX[isic_class]
            label = DX_TO_LABEL[dx]
            filepath = f"{isic_class}/{image_id}.jpg"
            rows.append((image_id, filepath, label))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filepath", "label"])
        writer.writerows(rows)

    label_to_dx = {v: k for k, v in DX_TO_LABEL.items()}
    per_class_counts = Counter(r[2] for r in rows)

    print("=== ISIC2019 External Validation Set Prep Summary ===")
    print(f"Total images in ground-truth CSV:  {total_found}")
    print(f"Excluded (SCC/UNK):                {excluded_scc_unk}")
    print(f"Excluded (dedup vs HAM10000):       {excluded_dedup}")
    print("Final count per class:")
    for label in sorted(per_class_counts):
        dx = label_to_dx[label]
        print(f"  {dx:6s} (label {label}): {per_class_counts[label]}")
    print(f"Final total image count:            {len(rows)}")
    print(f"Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
