import os
import yaml
from omegaconf import OmegaConf

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
    required_keys = ["data", "model", "loss", "train", "run_name", "output_dir", "splits_dir"]
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


def _split_overrides(overrides):
    """Split CLI overrides into (category switches, dotlist overrides).

    A category switch is a bare key matching one of CATEGORY_TO_FOLDER
    (e.g. "loss=focal") — it swaps which sub-config file gets loaded for
    that category. Everything else (e.g. "train.mixup.enabled=true",
    "run_name=foo") is a regular nested dotlist override applied after
    composition.
    """
    category_switches = {}
    dotlist = []
    for override in overrides:
        key, sep, value = override.partition("=")
        if sep and key in CATEGORY_TO_FOLDER and "." not in key:
            category_switches[key] = value
        else:
            dotlist.append(override)
    return category_switches, dotlist


def load_config(top_level_path, overrides=None):
    overrides = overrides or []
    category_switches, dotlist = _split_overrides(overrides)

    top_level = _load_yaml(top_level_path)

    cfg = {}

    # Resolve each entry in "defaults" (e.g. data: ham10000)
    # into the actual contents of configs/data/ham10000.yaml, applying any
    # CLI category switches (e.g. loss=focal) before resolving.
    defaults = dict(top_level.get("defaults", {}))
    defaults.update(category_switches)
    for category, name in defaults.items():
        folder = CATEGORY_TO_FOLDER[category]
        sub_config_path = os.path.join(folder, f"{name}.yaml")
        cfg[category] = _load_yaml(sub_config_path)

    # Copy over everything else in the top-level file that ISN'T
    # "defaults" (e.g. run_name, output_dir, wandb block)
    for key, value in top_level.items():
        if key != "defaults":
            cfg[key] = value

    if dotlist:
        merged = OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.from_dotlist(dotlist))
        cfg = OmegaConf.to_container(merged, resolve=True)

    _validate_config(cfg)
    return cfg
