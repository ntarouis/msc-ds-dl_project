from pathlib import Path
from PIL import Image
import cv2
import numpy as np

ALLOWED_CLASSES = {0, 1, 4}
CLASS_NAMES = {
    0: "car",
    1: "Different-Traffic-Sign",
    4: "pedestrian"
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def crop_yolo_subpix(img: np.ndarray, x_c: float, y_c: float, w: float, h: float) -> np.ndarray:
    H, W = img.shape[:2]
    patch_w = w * W
    patch_h = h * H
    center = (x_c * W, y_c * H)
    size = (int(round(patch_w)), int(round(patch_h)))
    return cv2.getRectSubPix(img, patchSize=size, center=center)


def crop_yolo_split(
    input_image_dir,
    input_label_dir,
    output_image_dir,
    output_label_dir,
    allowed_classes=ALLOWED_CLASSES,
    class_names=CLASS_NAMES,
    image_extensions=IMAGE_EXTENSIONS
):
    input_image_dir = Path(input_image_dir)
    input_label_dir = Path(input_label_dir)
    output_image_dir = Path(output_image_dir)
    output_label_dir = Path(output_label_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    counters = {}

    for img_path in input_image_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        lbl_path = input_label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"No label for {img_path.name}")
            continue

        pil_img = Image.open(img_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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

    print(f"\nCropped patches saved to {output_image_dir} and {output_label_dir}")
    print("Counts per class:")
    for class_name, count in counters.items():
        print(f"  {class_name}: {count}")


# # crop_yolo_split(
#     input_image_dir="data/split/golden_val_set/images",
#     input_label_dir="data/split/golden_val_set/labels",
#     output_image_dir="data/cropped/val_crop_images",
#     output_label_dir="data/cropped/val_crop_labels"
# )

# crop_yolo_split(
#     input_image_dir="data/split/golden_test_set/images",
#     input_label_dir="data/split/golden_test_set/labels",
#     output_image_dir="data/cropped/test_crop_images",
#     output_label_dir="data/cropped/test_crop_labels"
# )

