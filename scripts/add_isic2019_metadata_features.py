"""One-time data-prep script: add the same age/sex/localization feature
columns that scripts/add_metadata_features.py adds to the HAM10000
train/val/test splits, but to the ISIC2019 external validation metadata
(outputs/isic2019_metadata_filtered.csv).

Does not modify the input file — writes a new
outputs/isic2019_metadata_filtered_with_features.csv instead, since this
file is not also the split-identity file the way the HAM10000 split CSVs
are.

ISIC2019 uses "age_approx" and "anatom_site_general" instead of
HAM10000's "age" and "localization", and its localization values use
different category names (e.g. "anterior torso" instead of "trunk").
Only exact conceptual matches ("lower extremity", "upper extremity") are
mapped to their HAM10000-equivalent columns; every other value, including
categories with no HAM10000 equivalent, falls into loc_other_site. The
age median used for imputation is computed from this file alone (there is
no train/val/test split here to avoid leakage across).
"""

import os

import pandas as pd

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
IN_PATH = os.path.join(OUTPUTS_DIR, "isic2019_metadata_filtered.csv")
OUT_PATH = os.path.join(OUTPUTS_DIR, "isic2019_metadata_filtered_with_features.csv")

AGE_MIN = 0.0
AGE_MAX = 85.0

# ISIC2019 anatom_site_general values with an exact HAM10000 localization
# equivalent. All other values (e.g. "anterior torso", "posterior torso",
# "lateral torso", "head/neck", "palms/soles", "oral/genital") fall into
# loc_other_site.
MATCHED_LOCALIZATIONS = [
    "lower extremity",
    "upper extremity",
]
LOC_COLUMNS = [
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


def add_age_features(df, median):
    is_missing = df["age_approx"].isna()
    filled = df["age_approx"].fillna(median)
    normalized = (filled - AGE_MIN) / (AGE_MAX - AGE_MIN)
    df["age_normalized"] = normalized
    df["age_is_missing"] = is_missing.astype(int)
    return is_missing.sum()


def add_sex_features(df):
    missing = df["sex"].isna()
    sex = df["sex"].where(~missing, "unknown")
    df["sex_female"] = (sex == "female").astype(int)
    df["sex_male"] = (sex == "male").astype(int)
    return missing.sum()


def add_localization_features(df):
    missing = df["anatom_site_general"].isna()
    loc = df["anatom_site_general"].where(~missing, "unknown")

    for col in LOC_COLUMNS:
        df[col] = 0
    df["loc_lower_extremity"] = (loc == "lower extremity").astype(int)
    df["loc_upper_extremity"] = (loc == "upper extremity").astype(int)
    df["loc_unknown"] = (loc == "unknown").astype(int)
    df["loc_other_site"] = (~loc.isin(MATCHED_LOCALIZATIONS + ["unknown"])).astype(int)
    return missing.sum()


def main():
    df = pd.read_csv(IN_PATH)

    median = df["age_approx"].median()
    print(f"Computed age_approx median (from this file only): {median}")

    n_age_missing = add_age_features(df, median)
    n_sex_missing = add_sex_features(df)
    n_loc_missing = add_localization_features(df)

    df.to_csv(OUT_PATH, index=False)

    new_cols = ["age_normalized", "age_is_missing", "sex_female", "sex_male"] + LOC_COLUMNS
    print("\n=== ISIC2019 External Validation Metadata Feature Summary ===")
    print(f"Input:  {IN_PATH}")
    print(f"Output: {OUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Age median used for imputation: {median}")
    print(f"Missing age_approx values (before imputation): {n_age_missing}")
    print(f"Missing sex values (before imputation, NaN treated as unknown): {n_sex_missing}")
    print(f"Missing anatom_site_general values (before imputation, NaN treated as unknown): {n_loc_missing}")
    print(f"New columns added ({len(new_cols)}): {new_cols}")
    print(f"Final shape: {df.shape}")

    print("\n--- output head ---")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()
