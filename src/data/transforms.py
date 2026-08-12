from torchvision import transforms


def build_train_transform(data_cfg):
    image_size = data_cfg["image_size"]
    aug = data_cfg["augmentation"]
    norm = data_cfg["normalize"]

    transform_list = []

    resize_size = image_size + aug.get("random_crop_pad", 32)
    if aug.get("use_random_resized_crop", True):
        scale = aug.get("random_resized_crop_scale", (0.7, 1.0))
        ratio = aug.get("random_resized_crop_ratio", (0.9, 1.1))
        transform_list.append(
            transforms.RandomResizedCrop(image_size, scale=scale, ratio=ratio)
        )
    else:
        transform_list.extend(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(image_size),
            ]
        )

    if aug.get("h_flip", False):
        transform_list.append(
            transforms.RandomHorizontalFlip(p=aug.get("h_flip_prob", 0.5))
        )

    if aug.get("v_flip", False):
        transform_list.append(
            transforms.RandomVerticalFlip(p=aug.get("v_flip_prob", 0.5))
        )

    transform_list.append(transforms.RandomRotation(aug.get("rotation_degrees", 20)))

    cj = aug.get("color_jitter", {})
    transform_list.append(
        transforms.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
            saturation=cj.get("saturation", 0.2),
            hue=cj.get("hue", 0.0),
        )
    )

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=norm["mean"], std=norm["std"]))

    re = aug.get("random_erasing", {})
    if re.get("prob", 0.0) > 0:
        transform_list.append(
            transforms.RandomErasing(p=re["prob"], scale=tuple(re.get("scale", (0.02, 0.33))))
        )

    return transforms.Compose(transform_list)

 
def build_eval_transform(data_cfg):
    image_size = data_cfg["image_size"]
    norm = data_cfg["normalize"]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])