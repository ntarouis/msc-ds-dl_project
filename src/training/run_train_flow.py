import os
import torch
from src.preprocessing.dataloader import get_dataloaders
from src.training.training_utils import train_one_epoch, validate_one_epoch
from src.models.models import SimpleCNN, ImprovedCNN
from src.testing.evaluate import load_model, evaluate_model

from src.preprocessing import split_and_deduplicate
from src.preprocessing.crop_and_augment_trainset import crop_and_augment_yolo_train_set
from src.preprocessing.crop_val_test_sets import crop_yolo_split

def main():
    # --- Configs ---
    IMG_SIZE = 128
    VGG_IMG_SIZE = 224
    NUM_CLASSES = 3
    BATCH_SIZE = 64
    CLASS_NAMES_LIST = ["car", "Different-Traffic-Sign", "pedestrian"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---  Preprocessing ---
    print("Running preprocessing...")
    
    split_and_deduplicate.split_dataset(image_folder='data\\images', label_folder='data\\labels')
    
    crop_and_augment_yolo_train_set(train_image_dir="data/split/golden_train_set/images", train_label_dir="data/split/golden_train_set/labels",
                                    output_image_dir="data/cropped/train_crop_images", output_label_dir="data/cropped/train_crop_labels")
    
    crop_yolo_split(
        input_image_dir="data/split/golden_val_set/images",
        input_label_dir="data/split/golden_val_set/labels",
        output_image_dir="data/cropped/val_crop_images",
        output_label_dir="data/cropped/val_crop_labels"
    )

    crop_yolo_split(
        input_image_dir="data/split/golden_test_set/images",
        input_label_dir="data/split/golden_test_set/labels",
        output_image_dir="data/cropped/test_crop_images",
        output_label_dir="data/cropped/test_crop_labels"
    )

    train_img_dir = "data/cropped/train_crop_images"
    train_lbl_dir = "data/cropped/train_crop_labels"
    val_img_dir = "data/cropped/val_crop_images"
    val_lbl_dir = "data/cropped/val_crop_labels"
    test_img_dir = "data/cropped/test_crop_images"
    test_lbl_dir = "data/cropped/test_crop_labels"

    train_loader, val_loader, test_loader = get_dataloaders(
        train_img_dir, train_lbl_dir,
        val_img_dir, val_lbl_dir,
        test_img_dir, test_lbl_dir,
        img_size=128,
        batch_size=64,
        num_workers=2
    )

    print("Preprocessing completed.")

    # --- 2. Training ---
    print("Starting training...")
    model = SimpleCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    NUM_EPOCHS = 1
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    main()