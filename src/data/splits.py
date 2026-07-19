import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _find_image_path(image_id, image_dirs):
    for d in image_dirs:
        candidate = os.path.join(d, image_id + ".jpg")
        if os.path.exists(candidate):
            return candidate
    return None


def _load_metadata(data_cfg):
    metadata_path = os.path.join(data_cfg["data_dir"], data_cfg["metadata_csv"])
    meta = pd.read_csv(metadata_path)

    image_dirs = [os.path.join(data_cfg["data_dir"], d) for d in data_cfg["image_dirs"]]
    meta["filepath"] = meta["image_id"].apply(lambda i: _find_image_path(i, image_dirs))

    missing = meta["filepath"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} images not found on disk — check image_dirs paths.")
    meta = meta.dropna(subset=["filepath"]).reset_index(drop=True)

    class_to_idx = {c: i for i, c in enumerate(data_cfg["classes"])}
    meta["label"] = meta["dx"].map(class_to_idx)

    return meta


def _split_lesions(meta, split_cfg):
    group_col = split_cfg["group_col"]
    seed = split_cfg["seed"]

    lesion_level = meta.drop_duplicates(subset=group_col)[[group_col, "label"]]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=split_cfg["test_size"], random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(lesion_level, groups=lesion_level[group_col]))
    trainval_lesions = lesion_level.iloc[trainval_idx][group_col].values
    test_lesions = lesion_level.iloc[test_idx][group_col].values

    lesion_level_trainval = lesion_level[lesion_level[group_col].isin(trainval_lesions)]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=split_cfg["val_size"], random_state=seed)
    train_idx, val_idx = next(gss_val.split(lesion_level_trainval, groups=lesion_level_trainval[group_col]))
    train_lesions = lesion_level_trainval.iloc[train_idx][group_col].values
    val_lesions = lesion_level_trainval.iloc[val_idx][group_col].values

    return train_lesions, val_lesions, test_lesions


def _assert_no_leakage(train_df, val_df, test_df, group_col):
    train_set = set(train_df[group_col])
    val_set = set(val_df[group_col])
    test_set = set(test_df[group_col])

    assert not (train_set & val_set), "Leak: lesion overlap between train and val"
    assert not (train_set & test_set), "Leak: lesion overlap between train and test"
    assert not (val_set & test_set), "Leak: lesion overlap between val and test"


def get_lesion_level_splits(data_cfg, save_dir=None):
    """
    Returns train_df, val_df, test_df split at the lesion level.
    If save_dir is given and splits already exist there, loads them
    from disk instead of recomputing — so every run uses the exact
    same split.
    """
    if save_dir is not None:
        train_path = os.path.join(save_dir, "train.csv")
        val_path = os.path.join(save_dir, "val.csv")
        test_path = os.path.join(save_dir, "test.csv")

        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            print(f"Loading existing splits from {save_dir}")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            return train_df, val_df, test_df

    meta = _load_metadata(data_cfg)
    group_col = data_cfg["split"]["group_col"]

    train_lesions, val_lesions, test_lesions = _split_lesions(meta, data_cfg["split"])

    train_df = meta[meta[group_col].isin(train_lesions)].reset_index(drop=True)
    val_df = meta[meta[group_col].isin(val_lesions)].reset_index(drop=True)
    test_df = meta[meta[group_col].isin(test_lesions)].reset_index(drop=True)

    _assert_no_leakage(train_df, val_df, test_df, group_col)

    print(f"Train images: {len(train_df)} | Val images: {len(val_df)} | Test images: {len(test_df)}")
    print(f"Train lesions: {len(train_lesions)} | Val lesions: {len(val_lesions)} | Test lesions: {len(test_lesions)}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
        print(f"Saved splits to {save_dir}")

    return train_df, val_df, test_df