"""
Classifying Module

This module performs test time augmentation (TTA) on a test dataset of font images using a pretrained ResNet18 model.
It loads test images from an HDF5 file, applies multiple augmentations per sample, aggregates the predictions using majority voting,
and writes the final predictions along with image information to a CSV file ("results.csv").

Functions:
    - predict_with_tta(num_augmentations=5): Performs TTA on each test sample and returns aggregated predictions.
    - write_predictions(predictions, img_names, chars_list): Writes the predictions and associated data to a CSV file.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import FontDataset
from model import get_resnet18_model
import csv

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_with_tta(num_augmentations=5):
    """
    Perform test time augmentation (TTA) on the test dataset.

    For each test sample, this function applies 'num_augmentations' augmentations, makes predictions on each augmented version,
    and aggregates the predictions using majority voting.

    Args:
        num_augmentations (int): Number of augmentations to perform per test sample.

    Returns:
        tuple: (predictions, img_names, chars_list)
            predictions: List of final predicted class indices.
            img_names: List of source image names.
            chars_list: List of characters corresponding to each test sample.
    """
    test_dataset = FontDataset('hdf5_files/test.h5', group_name='images')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = get_resnet18_model(num_classes=7)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    img_names = []
    chars_list = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            preds = []
            for _ in range(num_augmentations):
                img, _, char, src_img = test_dataset[i]
                img = img.unsqueeze(0).to(device)
                output = model(img)
                pred = torch.argmax(output, dim=1).item()
                preds.append(pred)

            final_pred = int(np.bincount(preds).argmax())
            predictions.append(final_pred)
            img_names.append(src_img)
            chars_list.append(char)

    return predictions, img_names, chars_list

def write_predictions(predictions, img_names, chars_list):
    """
    Write predictions to a CSV file.

    The CSV file will contain columns: 'index', 'image', 'char', and 'font'.
    A font mapping dictionary is used to convert numeric class predictions to font names.

    Args:
        predictions (list): List of predicted class indices.
        img_names (list): List of source image names.
        chars_list (list): List of characters corresponding to each test sample.
    """
    font_map = {
        0: "always forever",
        1: "Skylark",
        2: "Sweet Puppy",
        3: "Ubuntu Mono",
        4: "VertigoFLF",
        5: "Wanted M54",
        6: "Flower Rose Brush"
    }

    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['index', 'image', 'char', 'font']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (img_name, char, pred) in enumerate(zip(img_names, chars_list, predictions)):
            writer.writerow({
                'index': idx,
                'image': img_name,
                'char': char,
                'font': font_map[pred]
            })

    print("✅ results.csv generated successfully!")

if __name__ == '__main__':
    preds, names, chars = predict_with_tta(num_augmentations=5)
    write_predictions(preds, names, chars)
