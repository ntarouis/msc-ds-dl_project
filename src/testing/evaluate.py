import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_class, model_path, device, num_classes, img_size=None):
    """
    Loads a model from file.
    model_class: class to instantiate (SimpleCNN, ImprovedCNN, etc)
    model_path: path to .pth file
    device: 'cuda' or 'cpu'
    num_classes: int
    img_size: int or None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img_size is not None:
        model = model_class(num_classes=num_classes, img_size=img_size).to(device)
    else:
        model = model_class(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, test_loader, class_names, model_name=None, print_report=True, plot_cm=True):
    all_preds = []
    all_labels = []
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(next(model.parameters()).device)
            labels = labels.to(next(model.parameters()).device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if print_report:
        print(f"\nClassification Report on Test Set for model {model_name}:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

    if plot_cm:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Test Set Confusion Matrix for model {model_name}")
        plt.show()
    return all_labels, all_preds
