import torch
import timm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_cfg):
    model = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
    )
    device = get_device()
    model = model.to(device)
    return model