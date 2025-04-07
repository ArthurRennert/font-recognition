# Font Classification using Deep Learning

This project implements a font classification system that identifies the font used in an image of a single character. The model is based on a fine-tuned ResNet18 architecture and trained on a custom dataset built from annotated HDF5 files.

## 🔍 Project Overview
- **Goal**: Automatically classify fonts from cropped character images.
- **Architecture**: ResNet18 (pretrained on ImageNet, fine-tuned for 7 font classes)
- **Dataset**: Custom-built HDF5 dataset, containing images of characters extracted from word images.
- **Tools**: PyTorch, OpenCV, h5py, NumPy

## 📁 Repository Structure
```
.
├── data_preparation.py     # Crops characters from raw images using bounding boxes
├── dataset.py              # Custom PyTorch Dataset for loading HDF5 data
├── model.py                # Defines the enhanced ResNet18 model
├── main.py                 # Training loop with class weights and early stopping
├── classifying.py          # Test Time Augmentation and CSV prediction output
├── hdf5_files/             # Contains train/val/test HDF5 datasets
```

## 🧠 Model Architecture
An enhanced version of ResNet18:
- Removes the final FC layer
- Adds:
  - Linear layer (512 units)
  - BatchNorm
  - Dropout
  - ReLU
  - Final linear layer (7 outputs)

## 🗃️ Dataset
- Input: Raw HDF5 files with word images and bounding boxes (`charBB`, `wordBB`)
- Output: HDF5 datasets (`train.h5`, `val.h5`, `test.h5`) with individual 224x224 character images
- 7 Font Classes:
  - always forever
  - Skylark
  - Sweet Puppy
  - Ubuntu Mono
  - VertigoFLF
  - Wanted M54
  - Flower Rose Brush

## 🚀 Training
Run the training script:
```bash
python main.py
```
Includes:
- AdamW optimizer
- Cosine annealing scheduler
- Early stopping after 50 epochs without improvement
- Class weighting to handle data imbalance

## 🔍 Inference (with TTA)
Run predictions on test set:
```bash
python classifying.py
```
- Performs 5 test-time augmentations per sample
- Aggregates predictions via majority voting
- Outputs `results.csv`

## 💡 Highlights
- Fully custom data pipeline from raw bounding box data
- ResNet18 fine-tuning with strong generalization
- Test Time Augmentation for improved performance

## 📬 Contact
For questions or collaborations, feel free to contact me:
**Arthur Rennert**  
ArthurRennert@gmail.com
