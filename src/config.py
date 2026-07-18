import os
import yaml

# Maps each key in the top-level "defaults" block to the folder
# where that category's yaml files live.
CATEGORY_TO_FOLDER = {
    "data": "configs/data",
    "model": "configs/model",
    "loss": "configs/loss",
    "train": "configs/train",
}


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _validate_config(cfg):
    required_keys = ["data", "model", "loss", "train", "run_name", "output_dir"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Config is missing required section: '{key}'")

    num_classes_declared = cfg["model"]["num_classes"]
    num_classes_listed = len(cfg["data"]["classes"])
    if num_classes_declared != num_classes_listed:
        raise ValueError(
            f"Mismatch: model.num_classes={num_classes_declared} but "
            f"data.classes has {num_classes_listed} entries."
        )

    valid_metrics = {"balanced_accuracy", "macro_f1"}
    metric = cfg["train"]["metric_for_best"]
    if metric not in valid_metrics:
        raise ValueError(
            f"train.metric_for_best='{metric}' is not one of {valid_metrics}"
        )


def load_config(top_level_path):
    top_level = _load_yaml(top_level_path)

    cfg = {}

    # Resolve each entry in "defaults" (e.g. data: ham10000)
    # into the actual contents of configs/data/ham10000.yaml
    defaults = top_level.get("defaults", {})
    for category, name in defaults.items():
        folder = CATEGORY_TO_FOLDER[category]
        sub_config_path = os.path.join(folder, f"{name}.yaml")
        cfg[category] = _load_yaml(sub_config_path)

    # Copy over everything else in the top-level file that ISN'T
    # "defaults" (e.g. run_name, output_dir, wandb block)
    for key, value in top_level.items():
        if key != "defaults":
            cfg[key] = value

    _validate_config(cfg)
    return cfg