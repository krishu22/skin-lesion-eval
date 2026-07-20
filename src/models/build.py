import torch
import timm


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inject_dropout_head(model, dropout_p):
    if dropout_p <= 0:
        return model

    for attr_name in ("head", "classifier", "fc"):
        if hasattr(model, attr_name):
            module = getattr(model, attr_name)
            if isinstance(module, torch.nn.Module):
                setattr(model, attr_name, torch.nn.Sequential(torch.nn.Dropout(dropout_p), module))
                break

    return model


def build_model(model_cfg):
    model = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
    )
    model = _inject_dropout_head(model, model_cfg.get("dropout", 0.0))
    device = get_device()
    model = model.to(device)
    return model