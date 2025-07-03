import torch
from torchvision import models
import torch.nn as nn

from src.preprocessing.vgg_preprocessing import create_loaders
from src.training.training_utils import train_one_epoch, validate_one_epoch  # Adjust import if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

train_loader, val_loader, _ = create_loaders(batch_size=32,create_train=True)

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
    param.requires_grad = False
vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASSES)
vgg16 = vgg16.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.classifier.parameters(), lr=1e-4)

best_val_acc = 0.0
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    train_acc = train_one_epoch(epoch, vgg16, train_loader, optimizer, criterion, DEVICE)
    val_acc = validate_one_epoch(vgg16, val_loader, criterion, DEVICE)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(vgg16.state_dict(), "model_checkpoints/best_vgg16.pth")
