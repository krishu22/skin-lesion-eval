"""One-time data-prep script: add age/sex/localization feature columns to the
train/val/test split CSVs in outputs/splits/, in place. These are the same
files src.data.splits.get_lesion_level_splits caches image identity to and
src.data.metadata.load_metadata_features reads metadata features from — one
file per split is the single source of truth for both image loading and
metadata loading, so this script updates each split CSV directly rather than
writing a separate "_with_metadata" file. Safe to re-run: it recomputes and
overwrites the same feature columns rather than duplicating them.

Statistics used for imputation (age median) are computed from train.csv only
and applied identically to train/val/test to avoid leakage.
"""

import os

import pandas as pd

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "splits")

AGE_MIN = 0.0
AGE_MAX = 85.0

TOP_LOCALIZATIONS = [
    "back",
    "lower extremity",
    "trunk",
    "upper extremity",
    "abdomen",
    "face",
    "chest",
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


def add_age_features(df, train_median):
    is_missing = df["age"].isna()
    filled = df["age"].fillna(train_median)
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
    missing = df["localization"].isna()
    loc = df["localization"].where(~missing, "unknown")

    for col, name in zip(LOC_COLUMNS[:7], TOP_LOCALIZATIONS):
        df[col] = (loc == name).astype(int)
    df["loc_unknown"] = (loc == "unknown").astype(int)
    df["loc_other_site"] = (~loc.isin(TOP_LOCALIZATIONS + ["unknown"])).astype(int)
    return missing.sum()


def process(name, train_median):
    in_path = os.path.join(SPLITS_DIR, f"{name}.csv")
    out_path = in_path
    df = pd.read_csv(in_path)

    n_age_missing = add_age_features(df, train_median)
    n_sex_missing = add_sex_features(df)
    n_loc_missing = add_localization_features(df)

    df.to_csv(out_path, index=False)

    new_cols = ["age_normalized", "age_is_missing", "sex_female", "sex_male"] + LOC_COLUMNS
    print(f"\n=== {name} ===")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Train-derived age median used for imputation: {train_median}")
    print(f"Missing age values (before imputation): {n_age_missing}")
    print(f"Missing sex values (before imputation, NaN treated as unknown): {n_sex_missing}")
    print(f"Missing localization values (before imputation, NaN treated as unknown): {n_loc_missing}")
    print(f"New columns added ({len(new_cols)}): {new_cols}")
    print(f"Final shape: {df.shape}")
    return df


def main():
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    train_median = train_df["age"].median()
    print(f"Computed train-only age median: {train_median}")

    outputs = {}
    for name in ["train", "val", "test"]:
        outputs[name] = process(name, train_median)

    for name, df in outputs.items():
        print(f"\n--- {name}.csv head ---")
        print(df.head(3).to_string())


if __name__ == "__main__":
    main()
