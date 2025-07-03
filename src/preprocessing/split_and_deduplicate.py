from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


def deduplicate_images(image_folder, label_folder, unique_image_folder='data\\unique\\unique_images', unique_label_folder='data\\unique\\unique_labels'):
    unique_image_folder = Path(unique_image_folder)
    unique_label_folder = Path(unique_label_folder)
    unique_image_folder.mkdir(parents=True, exist_ok=True)
    unique_label_folder.mkdir(parents=True, exist_ok=True)

    seen_prefixes = set()
    for image_path in Path(image_folder).glob("*.*"):
        prefix = image_path.stem.split('-')[0].split('_')[0]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            shutil.copy(image_path, unique_image_folder / image_path.name)
            label_path = Path(label_folder) / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, unique_label_folder / label_path.name)
      
              
def split_dataset(
    image_folder,
    label_folder,
    unique_image_folder='data/unique/unique_images',
    unique_label_folder='data/unique/unique_labels',
    train_ratio=0.6,
    val_ratio=0.2,
    data_splitted_base_dir='data/split',
    random_seed=42
):
    # Deduplicate first
    deduplicate_images(image_folder, label_folder, unique_image_folder, unique_label_folder)
    
    unique_image_folder = Path(unique_image_folder)
    unique_label_folder = Path(unique_label_folder)
    base_dir = Path(data_splitted_base_dir)


    golden_dirs = {
        "train": base_dir / "golden_train_set",
        "val":   base_dir / "golden_val_set",
        "test":  base_dir / "golden_test_set"
    }
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted([p for p in unique_image_folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])

    # Split images
    test_ratio = 1.0 - train_ratio - val_ratio
    train_imgs, temp_imgs = train_test_split(all_images, test_size=(1.0 - train_ratio), random_state=random_seed)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed)
    
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split, imgs in splits.items():
        img_dir = golden_dirs[split] / "images"
        lbl_dir = golden_dirs[split] / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_count = 0
        lbl_count = 0
        for img_path in imgs:
            shutil.copy2(img_path, img_dir / img_path.name)
            img_count += 1
            lbl_path = unique_label_folder / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dir / lbl_path.name)
                lbl_count += 1
        print(f"{split.capitalize():<5} set: {img_count:>5} images, {lbl_count:>5} labels copied to {img_dir.parent}")

    print("\nDataset was splitted successfully!\n")

# split_dataset(image_folder='data\\images',
#               label_folder='data\\labels')