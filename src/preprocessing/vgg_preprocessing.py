import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.preprocessing.crop_val_test_sets import crop_yolo_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_CLASSES = {0, 1, 4}

CLASS_NAMES_LIST = ["car", "Different-Traffic-Sign", "pedestrian"]


class CroppedImageDataset(Dataset):
    """
    Pairs each cropped image with its label.
    Each label file contains exactly one line: the class name.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.samples = []
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            lbl_path = self.label_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))
        if len(self.samples) == 0:
            raise RuntimeError("No cropped image/label pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        with open(lbl_path, "r") as f:
            class_name = f.readline().strip()
        label_idx = CLASS_NAMES_LIST.index(class_name)
        return img, label_idx


def create_loaders(
    train_image_dir="data/cropped/vgg_train_crop_images",
    train_label_dir="data/cropped/vgg_train_crop_labels",
    val_image_dir="data/cropped/val_crop_images",
    val_label_dir="data/cropped/val_crop_labels",
    test_image_dir="data/cropped/test_crop_images",
    test_label_dir="data/cropped/test_crop_labels",
    batch_size=32,
    create_train = False
):
    if create_train:
        crop_yolo_split(
            input_image_dir="data/split/golden_val_set/images",
            input_label_dir="data/split/golden_val_set/labels",
            output_image_dir="data/cropped/vgg_train_crop_images",
            output_label_dir="data/cropped/vgg_train_crop_labels"
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CroppedImageDataset(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        transform=train_transform
    )
    val_dataset = CroppedImageDataset(
        image_dir=val_image_dir,
        label_dir=val_label_dir,
        transform=val_test_transform
    )
    test_dataset = CroppedImageDataset(
        image_dir=test_image_dir,
        label_dir=test_label_dir,
        transform=val_test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader
    )
