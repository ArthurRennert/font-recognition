"""
FontDataset Module

This module defines a custom PyTorch Dataset for loading and processing font image data
stored in an HDF5 file. The HDF5 file is expected to contain a group (e.g., "images") where
each dataset represents an individual image sample. Each sample includes attributes such as:
    - 'char': the character shown in the image.
    - 'src_img': the original source image filename.
    - 'font': (for training/validation data) the font label as a string.

The dataset applies a series of transformations to each image:
    - Converts the image to a PIL image.
    - Resizes it to 224x224.
    - Converts the PIL image to a tensor.
    - Normalizes the tensor using a mean of [0.5, 0.5, 0.5] and standard deviation of [0.5, 0.5, 0.5].

For test data (when the file path is 'hdf5_files/test.h5'), the dataset returns a label index of -1,
since no font label is provided.
"""

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Mapping of font names to class indices
fonts_dict = {
    'always forever': 0,
    'Skylark': 1,
    'Sweet Puppy': 2,
    'Ubuntu Mono': 3,
    'VertigoFLF': 4,
    'Wanted M54': 5,
    'Flower Rose Brush': 6
}

class FontDataset(Dataset):
    def __init__(self, file_path, group_name):
        """
        Initializes the FontDataset.

        Args:
            file_path (str): Path to the HDF5 file containing the data.
            group_name (str): Name of the group within the HDF5 file (e.g., 'images').
        """
        self.f = h5py.File(file_path, 'r')
        self.file_path = file_path
        self.group_name = group_name
        self.images_names = list(self.f[group_name].keys())
        self.data_group = self.f[group_name]

        # Define image transformations (without augmentations)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.images_names)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, label_index, character, src_img)
                   For test data, label_index is -1.
                   For training/validation data, label_index is determined using fonts_dict.
        """
        img_name = self.images_names[idx]
        image_data = self.data_group[img_name][()]
        char = self.data_group[img_name].attrs['char']
        src_img = self.data_group[img_name].attrs['src_img']

        if self.file_path == 'hdf5_files/test.h5':
            label_idx = -1
            image_tensor = self.transform(image_data)
            return image_tensor, label_idx, char, src_img
        else:
            raw_label = self.data_group[img_name].attrs['font']
            label = raw_label.decode('utf-8') if isinstance(raw_label, bytes) else raw_label
            label_idx = fonts_dict[label]
            image_tensor = self.transform(image_data)
            return image_tensor, label_idx, char, src_img
