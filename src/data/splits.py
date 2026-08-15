import json
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


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
    from disk instead of recomputing — so every run (including every
    ablation, whether raw or segmented data) trains/evals on the exact
    same split. The cache stores only image identity (not filepath), and
    `filepath` is always re-resolved against the current data_cfg — so the
    same cached split is safely reusable across raw vs. segmented data_dirs.
    The cache is keyed on the resolved image count, so a download that was
    incomplete when the cache was first written won't silently poison later
    runs once the dataset is complete.
    """
    meta = _load_metadata(data_cfg)
    group_col = data_cfg["split"]["group_col"]

    if save_dir is not None:
        train_path = os.path.join(save_dir, "train.csv")
        val_path = os.path.join(save_dir, "val.csv")
        test_path = os.path.join(save_dir, "test.csv")
        manifest_path = os.path.join(save_dir, "manifest.json")

        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            cached_train_ids = pd.read_csv(train_path)["image_id"]
            cached_val_ids = pd.read_csv(val_path)["image_id"]
            cached_test_ids = pd.read_csv(test_path)["image_id"]
            cached_count = len(cached_train_ids) + len(cached_val_ids) + len(cached_test_ids)

            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    expected_count = json.load(f)["resolved_image_count"]
            else:
                expected_count = cached_count  # pre-existing cache with no manifest — trust it once

            if cached_count == expected_count == len(meta):
                train_df = meta[meta["image_id"].isin(cached_train_ids)].reset_index(drop=True)
                val_df = meta[meta["image_id"].isin(cached_val_ids)].reset_index(drop=True)
                test_df = meta[meta["image_id"].isin(cached_test_ids)].reset_index(drop=True)
                print(f"Loading existing splits from {save_dir}")
                return train_df, val_df, test_df

            raise RuntimeError(
                f"Cached splits at {save_dir} were computed from {expected_count} resolved images, "
                f"but the dataset currently resolves {len(meta)}. This usually means the cache was "
                f"written from an incomplete download. Delete {save_dir} and rerun to recompute."
            )

    train_lesions, val_lesions, test_lesions = _split_lesions(meta, data_cfg["split"])

    train_df = meta[meta[group_col].isin(train_lesions)].reset_index(drop=True)
    val_df = meta[meta[group_col].isin(val_lesions)].reset_index(drop=True)
    test_df = meta[meta[group_col].isin(test_lesions)].reset_index(drop=True)

    _assert_no_leakage(train_df, val_df, test_df, group_col)

    print(f"Train images: {len(train_df)} | Val images: {len(val_df)} | Test images: {len(test_df)}")
    print(f"Train lesions: {len(train_lesions)} | Val lesions: {len(val_lesions)} | Test lesions: {len(test_lesions)}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        train_df.drop(columns="filepath").to_csv(os.path.join(save_dir, "train.csv"), index=False)
        val_df.drop(columns="filepath").to_csv(os.path.join(save_dir, "val.csv"), index=False)
        test_df.drop(columns="filepath").to_csv(os.path.join(save_dir, "test.csv"), index=False)
        with open(os.path.join(save_dir, "manifest.json"), "w") as f:
            json.dump({"resolved_image_count": len(meta)}, f)
        print(f"Saved splits to {save_dir}")

    return train_df, val_df, test_df


def get_cv_fold(data_cfg, cv_cfg, save_dir=None):
    """
    Returns (train_df, val_df, test_df) for cv_cfg["fold_index"] of a stratified,
    lesion-grouped k-fold split over the combined train+val pool. test_df is always
    the same held-out test set produced by get_lesion_level_splits — CV never
    touches it. The fold assignment (which lesions land in which of cv_cfg["n_folds"]
    folds) is fully determined by cv_cfg["seed"], so every fold_index invocation sees
    the same partition.
    """
    train_df, val_df, test_df = get_lesion_level_splits(data_cfg, save_dir=save_dir)

    group_col = data_cfg["split"]["group_col"]
    pool_df = pd.concat([train_df, val_df], ignore_index=True)

    lesion_level = pool_df.drop_duplicates(subset=group_col)[[group_col, "label"]].reset_index(drop=True)

    n_folds = cv_cfg["n_folds"]
    fold_index = cv_cfg["fold_index"]
    seed = cv_cfg["seed"]

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(lesion_level, lesion_level["label"], groups=lesion_level[group_col]))
    train_lesion_idx, val_lesion_idx = folds[fold_index]

    train_lesions = lesion_level.iloc[train_lesion_idx][group_col].values
    val_lesions = lesion_level.iloc[val_lesion_idx][group_col].values

    fold_train_df = pool_df[pool_df[group_col].isin(train_lesions)].reset_index(drop=True)
    fold_val_df = pool_df[pool_df[group_col].isin(val_lesions)].reset_index(drop=True)

    _assert_no_leakage(fold_train_df, fold_val_df, test_df, group_col)

    print(f"[CV] fold={fold_index}/{n_folds} train lesions={len(train_lesions)} val lesions={len(val_lesions)}")
    print(f"[CV] fold={fold_index}/{n_folds} train images={len(fold_train_df)} val images={len(fold_val_df)}")

    return fold_train_df, fold_val_df, test_df