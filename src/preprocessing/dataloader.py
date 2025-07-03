import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


CLASS_NAMES_LIST = ["car", "Different-Traffic-Sign", "pedestrian"]
NUM_CLASSES = len(CLASS_NAMES_LIST)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def get_resize_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])


def get_dataloaders(
    train_image_dir, train_label_dir,
    val_image_dir, val_label_dir,
    test_image_dir, test_label_dir,
    img_size=128,
    batch_size=64,
    num_workers=2,
    pin_memory=True
):
    resize_transform = get_resize_transform(img_size)

    train_dataset = CroppedImageDataset(train_image_dir, train_label_dir, transform=resize_transform)
    val_dataset   = CroppedImageDataset(val_image_dir, val_label_dir, transform=resize_transform)
    test_dataset  = CroppedImageDataset(test_image_dir, test_label_dir, transform=resize_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
