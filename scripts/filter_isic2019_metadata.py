"""
One-time, standalone data-prep script.

Filters the full ISIC2019 metadata CSV down to only the rows that were
actually selected for external validation (see
scripts/prepare_isic2019_external_val.py).

Read-only against both input CSVs; writes a single new output CSV.

Run once: python scripts/filter_isic2019_metadata.py
"""

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

METADATA_CSV = REPO_ROOT / "outputs" / "isic2019_metadata.csv"
EXTERNAL_VAL_CSV = REPO_ROOT / "outputs" / "isic2019_external_val.csv"
OUTPUT_CSV = REPO_ROOT / "outputs" / "isic2019_metadata_filtered.csv"


def load_external_val_ids():
    ids = set()
    with open(EXTERNAL_VAL_CSV, newline="") as f:
        for row in csv.DictReader(f):
            ids.add(row["image_id"])
    return ids


def main():
    val_ids = load_external_val_ids()

    with open(METADATA_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    total_before = len(all_rows)
    kept_rows = [row for row in all_rows if row["image"] in val_ids]
    total_after = len(kept_rows)

    matched_ids = {row["image"] for row in kept_rows}
    unmatched = val_ids - matched_ids

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print("=== ISIC2019 Metadata Filtering Summary ===")
    print(f"Rows in metadata CSV before filtering: {total_before}")
    print(f"Rows in metadata CSV after filtering:  {total_after}")
    print(f"external_val image_id values with no match in metadata: {len(unmatched)}")
    if unmatched:
        print("  WARNING: unmatched image_id values found:")
        for image_id in sorted(unmatched)[:20]:
            print(f"    {image_id}")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")
    print(f"Output written to: {OUTPUT_CSV}")
    print()
    print("First few rows of output:")
    with open(OUTPUT_CSV, newline="") as f:
        for i, line in enumerate(f):
            if i > 5:
                break
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
