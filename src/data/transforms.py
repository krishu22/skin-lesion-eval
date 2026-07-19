from torchvision import transforms


def build_train_transform(data_cfg):
    image_size = data_cfg["image_size"]
    aug = data_cfg["augmentation"]
    norm = data_cfg["normalize"]

    resize_size = image_size + aug["random_crop_pad"]

    transform_list = [
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(image_size),
    ]

    if aug["h_flip"]:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.RandomRotation(aug["rotation_degrees"]))

    cj = aug["color_jitter"]
    transform_list.append(
        transforms.ColorJitter(
            brightness=cj["brightness"],
            contrast=cj["contrast"],
            saturation=cj["saturation"],
        )
    )

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=norm["mean"], std=norm["std"]))

    return transforms.Compose(transform_list)

 
def build_eval_transform(data_cfg):
    image_size = data_cfg["image_size"]
    norm = data_cfg["normalize"]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])