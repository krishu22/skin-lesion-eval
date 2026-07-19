import torch.nn as nn


def build_loss(loss_cfg):
    name = loss_cfg["name"]
    params = loss_cfg.get("params", {})

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**params)

    raise ValueError(f"Unknown loss name: '{name}'")