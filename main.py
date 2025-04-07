"""
Main Training Script

This script loads the training and validation datasets, computes class weights based on the
imbalance in the training data, builds an enhanced ResNet18 model, and trains the model using
the AdamW optimizer with a cosine annealing learning rate scheduler. The best model based on
validation accuracy is saved to the 'models' directory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FontDataset
from model import get_resnet18_model
import os
import numpy as np

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 5000
batch_size = 64
early_stop_patience = 50
learning_rate = 1e-4

# Load datasets
train_dataset = FontDataset('hdf5_files/train.h5', group_name='images')
val_dataset = FontDataset('hdf5_files/val.h5', group_name='images')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

# Compute class weights based on training data counts
num_classes = 7
class_counts = np.zeros(num_classes, dtype=np.int64)
for _, label, _, _ in train_dataset:
    class_counts[label] += 1
total_samples = class_counts.sum()
# Weight formula: weight = total_samples / (num_classes * count)
class_weights = total_samples / (num_classes * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class counts:", class_counts)
print("Class weights:", class_weights_tensor)

# Build the model and move to device
model = get_resnet18_model(num_classes=num_classes).to(device)

# Define loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

best_val_acc = 0.0
early_stop_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels, _, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels, _, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_counter = 0
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f"✅ New best model saved with val_acc: {best_val_acc:.2f}%")
    else:
        early_stop_counter += 1
        print(f"⚠️  No improvement in val_acc for {early_stop_counter} epochs.")

    if early_stop_counter >= early_stop_patience:
        print(f"⛔ Early stopping triggered at epoch {epoch + 1}")
        break

    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print('-' * 60)
