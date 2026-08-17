"""One-time data-prep script: build the single, final ISIC2019 external
validation CSV, outputs/isic2019_metadata.csv, in place.

Merges three things that used to live in separate files into one:
  - image_id / filepath / label from outputs/isic2019_external_val.csv
    (see scripts/prepare_isic2019_external_val.py)
  - raw age/sex/site fields (already present in outputs/isic2019_metadata.csv)
  - the 13 engineered feature columns computed in
    outputs/isic2019_metadata_filtered_with_features.csv (see
    scripts/add_isic2019_metadata_features.py)

The result is restricted to the external validation subset (rows are
matched on image_id == image) — there is no reason to carry the other
~10k ISIC2019 training images that were excluded from external
validation (SCC/UNK classes, or dedup against HAM10000). This makes
outputs/isic2019_metadata.csv the single file the eval pipeline
(scripts/eval_isic2019_external.py) reads for both image paths/labels
and metadata, replacing the old two-CSV (isic2019_external_val.csv +
isic2019_metadata.csv) setup.

Safe to re-run: it recomputes the merge from the source files rather
than accumulating columns across runs.
"""

import os

import pandas as pd

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
METADATA_PATH = os.path.join(OUTPUTS_DIR, "isic2019_metadata.csv")
EXTERNAL_VAL_PATH = os.path.join(OUTPUTS_DIR, "isic2019_external_val.csv")
FEATURES_PATH = os.path.join(OUTPUTS_DIR, "isic2019_metadata_filtered_with_features.csv")

RAW_COLUMNS = ["age_approx", "anatom_site_general", "lesion_id", "sex"]
FEATURE_COLUMNS = [
    "age_normalized",
    "age_is_missing",
    "sex_female",
    "sex_male",
    "loc_back",
    "loc_lower_extremity",
    "loc_trunk",
    "loc_upper_extremity",
    "loc_abdomen",
    "loc_face",
    "loc_chest",
    "loc_unknown",
    "loc_other_site",
]


def main():
    external_val_df = pd.read_csv(EXTERNAL_VAL_PATH)
    features_df = pd.read_csv(FEATURES_PATH)

    n_external_val = len(external_val_df)

    merged_df = external_val_df.merge(
        features_df[["image"] + RAW_COLUMNS + FEATURE_COLUMNS],
        left_on="image_id",
        right_on="image",
        how="left",
    ).drop(columns=["image"])

    n_matched = merged_df["age_normalized"].notna().sum()
    n_unmatched = len(merged_df) - n_matched
    if n_unmatched:
        raise ValueError(
            f"{n_unmatched} external validation image_id(s) had no matching metadata row "
            f"in {FEATURES_PATH} — every external validation image must have metadata."
        )

    merged_df.to_csv(METADATA_PATH, index=False)

    print("=== ISIC2019 Final Metadata File Build Summary ===")
    print(f"External val source:    {EXTERNAL_VAL_PATH} ({n_external_val} rows)")
    print(f"Feature source file:    {FEATURES_PATH}")
    print(f"Output (single file):   {METADATA_PATH}")
    print(f"Columns ({len(merged_df.columns)}): {list(merged_df.columns)}")
    print(f"Rows matched to a feature row: {n_matched} / {len(merged_df)}")
    print(f"Final shape: {merged_df.shape}")

    print("\n--- output head ---")
    print(merged_df.head(5).to_string())


if __name__ == "__main__":
    main()
