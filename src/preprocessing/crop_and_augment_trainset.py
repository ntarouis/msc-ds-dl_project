import shutil
import random
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from collections import defaultdict


def crop_and_augment_yolo_train_set(
    train_image_dir,
    train_label_dir,
    output_image_dir,
    output_label_dir,
    allowed_classes={0, 1, 4},
    class_names={0: "car", 1: "Different-Traffic-Sign", 4: "pedestrian"},
    image_extensions={".jpg", ".jpeg", ".png", ".bmp"},
    num_aug=3,
    jitter_ratio=0.15,
    seed=42
):
    """
    Crop and augment YOLO-formatted images and labels in a training set.
    Saves cropped patches and label files for allowed classes with jitter augmentation.
    Prints image and label counts per class.
    """
    random.seed(seed)
    train_image_dir = Path(train_image_dir)
    train_label_dir = Path(train_label_dir)
    output_image_dir = Path(output_image_dir)
    output_label_dir = Path(output_label_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    def crop_yolo_subpix(img: np.ndarray, x_c: float, y_c: float, w: float, h: float) -> np.ndarray:
        H, W = img.shape[:2]
        patch_w = w * W
        patch_h = h * H
        center = (x_c * W, y_c * H)
        size = (int(round(patch_w)), int(round(patch_h)))
        return cv2.getRectSubPix(img, patchSize=size, center=center)

    def jitter_bbox_yolo(x_c, y_c, w, h, jitter_ratio=0.15):
        dx = (random.uniform(-jitter_ratio, jitter_ratio) * w)
        dy = (random.uniform(-jitter_ratio, jitter_ratio) * h)
        new_x_c = min(max(x_c + dx, 0), 1)
        new_y_c = min(max(y_c + dy, 0), 1)
        return new_x_c, new_y_c, w, h

    def process_single_image_with_jitter(
        img_path: Path,
        lbl_path: Path,
        counters: dict,
        output_image_dir: Path,
        output_label_dir: Path,
        num_aug: int = 3,
        jitter_ratio: float = 0.15
    ):
        pil_img = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_img.size
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if not lbl_path.exists():
            print(f"No label for {img_path.name}")
            return

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for row in lines:
            parts = row.split()
            cls_id = int(parts[0])
            if cls_id not in allowed_classes:
                continue

            x_c, y_c, w_n, h_n = map(float, parts[1:5])
            class_name = class_names.get(cls_id, str(cls_id))
            count = counters.get(class_name, 0) + 1
            counters[class_name] = count
            base_name = f"{class_name}_{count:05d}"

            patch_cv = crop_yolo_subpix(img_cv, x_c, y_c, w_n, h_n)
            patch_pil = Image.fromarray(cv2.cvtColor(patch_cv, cv2.COLOR_BGR2RGB))
            patch_pil.save(output_image_dir / f"{base_name}.jpg")
            with open(output_label_dir / f"{base_name}.txt", "w") as lf:
                lf.write(class_name + "\n")

            # Save jittered augmentations
            for aug_idx in range(1, num_aug + 1):
                aug_x_c, aug_y_c, aug_w_n, aug_h_n = jitter_bbox_yolo(x_c, y_c, w_n, h_n, jitter_ratio)
                aug_patch_cv = crop_yolo_subpix(img_cv, aug_x_c, aug_y_c, aug_w_n, aug_h_n)
                aug_patch_pil = Image.fromarray(cv2.cvtColor(aug_patch_cv, cv2.COLOR_BGR2RGB))
                aug_img_name = f"{base_name}_jittered{aug_idx}.jpg"
                aug_patch_pil.save(output_image_dir / aug_img_name)
                aug_lbl_name = f"{base_name}_jittered{aug_idx}.txt"
                with open(output_label_dir / aug_lbl_name, "w") as lf:
                    lf.write(class_name + "\n")

    counters = {}
    for img_path in Path(train_image_dir).iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        lbl_path = Path(train_label_dir) / f"{img_path.stem}.txt"
        process_single_image_with_jitter(
            img_path,
            lbl_path,
            counters,
            output_image_dir,
            output_label_dir,
            num_aug=num_aug,
            jitter_ratio=jitter_ratio
        )

    image_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for img_path in output_image_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            class_name = img_path.stem.split('_')[0]
            if class_name in class_names.values():
                image_counts[class_name] += 1

    for label_path in output_label_dir.iterdir():
        if label_path.suffix.lower() == ".txt":
            with open(label_path, 'r') as f:
                class_name = f.readline().strip()
                if class_name in class_names.values():
                    label_counts[class_name] += 1

    print("\nImage counts per class:")
    for class_name, count in image_counts.items():
        print(f"  {class_name}: {count}")

    print("\nLabel counts per class:")
    for class_name, count in label_counts.items():
        print(f"  {class_name}: {count}")

# crop_and_augment_yolo_train_set(
#     train_image_dir="data/split/golden_train_set/images",
#     train_label_dir="data/split/golden_train_set/labels",
#     output_image_dir="data/cropped/train_crop_images",
#     output_label_dir="data/cropped/train_crop_labels"
# )