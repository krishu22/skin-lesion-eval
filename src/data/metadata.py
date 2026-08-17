import os

import pandas as pd

# 2 age fields + 2 sex fields + 9 location fields = 13-dim metadata feature vector.
METADATA_FEATURE_COLUMNS = [
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


def load_metadata_features_from_path(path, id_column="image_id"):
    """Loads the 13-dim metadata feature columns from an arbitrary CSV, indexed by id_column.

    Returns only METADATA_FEATURE_COLUMNS, indexed by id_column so callers can
    look up the right row per image rather than relying on row order matching
    another dataframe. id_column defaults to "image_id" (HAM10000 splits);
    pass id_column="image" for outputs/isic2019_metadata.csv, which uses that
    name instead. Columns are always reindexed to METADATA_FEATURE_COLUMNS,
    so the returned column order is identical regardless of the source CSV's
    column order.
    """
    df = pd.read_csv(path)

    missing_cols = [c for c in METADATA_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing expected metadata columns: {missing_cols}")

    if df[id_column].duplicated().any():
        dupes = df.loc[df[id_column].duplicated(), id_column].tolist()
        raise ValueError(f"{path} has duplicate {id_column} values: {dupes[:5]}")

    return df.set_index(id_column)[METADATA_FEATURE_COLUMNS]


def load_metadata_features(split_name, splits_dir):
    """Loads the precomputed metadata feature columns for a HAM10000 split.

    Reads outputs/splits/{split_name}.csv — the same split file
    src.data.splits.get_lesion_level_splits caches image identity to and
    scripts/add_metadata_features.py adds the engineered metadata columns to
    in place. Image loading and metadata loading both resolve to this one
    file per split, so there is a single source of truth per split rather
    than two files that could drift out of sync.
    """
    path = os.path.join(splits_dir, f"{split_name}.csv")
    return load_metadata_features_from_path(path)
