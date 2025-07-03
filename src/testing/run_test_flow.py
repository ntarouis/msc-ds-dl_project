import torch
from src.models.models import ImprovedCNN, SimpleCNN
from src.preprocessing.dataloader import get_dataloaders  
from src.preprocessing.dataloader import get_dataloaders  
from src.preprocessing.vgg_preprocessing import create_loaders as vgg_create_loaders
from torchvision import models
import torch.nn as nn
from src.testing.evaluate import load_model, evaluate_model
from torchvision import transforms
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def main():
    MODEL_PATH = "best_improved_model.pth"   
    MODEL_TYPE = "simple"                
    IMG_SIZE = 128
    NUM_CLASSES = 3
    BATCH_SIZE = 64
    CLASS_NAMES_LIST = ["car", "Different-Traffic-Sign", "pedestrian"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_img_dir = "data/cropped/train_crop_images"
    train_lbl_dir = "data/cropped/train_crop_labels"
    val_img_dir = "data/cropped/val_crop_images"
    val_lbl_dir = "data/cropped/val_crop_labels"
    test_img_dir = "data/cropped/test_crop_images"
    test_lbl_dir = "data/cropped/test_crop_labels"

    _, _, test_loader = get_dataloaders(
        train_img_dir, train_lbl_dir,
        val_img_dir, val_lbl_dir,
        test_img_dir, test_lbl_dir,
        img_size=128,
        batch_size=64,
        num_workers=2
    )
    
    _, _, vgg_test_loader = vgg_create_loaders(
        batch_size=BATCH_SIZE
    )


    def get_vgg16(num_classes, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        vgg16 = models.vgg16(pretrained=False)
        vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASSES)
        vgg16 = vgg16.to(device)
        return vgg16
    
    models_info = [
        {
            "name": "SimpleCNN",
            "class": SimpleCNN,
            "path": "model_checkpoints/best_model_simplecnn.pth",
            "extra_args": {"img_size": IMG_SIZE},
            "test_loader": test_loader
        },
        {
            "name": "ImprovedCNN",
            "class": ImprovedCNN,
            "path": "model_checkpoints/best_improved_model.pth",
            "extra_args": {},
            "test_loader": test_loader
        },
        {
            "name": "VGG16",
            "class": get_vgg16,  
            "path": "model_checkpoints/best_vgg16.pth",
            "extra_args": {},
            "test_loader": vgg_test_loader
        }
    ]
    
    for model_info in models_info:
        print(f"\nEvaluating model: {model_info['name']}")
        if model_info["name"] == "VGG16":
            model = model_info["class"](NUM_CLASSES, device)
            model.load_state_dict(torch.load(model_info["path"], map_location=device))
        else:
            model_class = model_info["class"]
            extra_args = model_info["extra_args"]
            model = load_model(model_class, model_info["path"], device, NUM_CLASSES, **extra_args)
        model.eval()
        all_labels, all_preds = evaluate_model(
            model, model_info["test_loader"], CLASS_NAMES_LIST, model_name=model_info["name"]
        )
    
if __name__ == "__main__":
    main()