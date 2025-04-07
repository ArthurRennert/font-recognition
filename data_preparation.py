"""
Data Preparation Module

This module processes an input HDF5 dataset containing images of words with associated attributes
(e.g., 'txt', 'wordBB', 'charBB', and optionally 'font') to extract individual character images.
For each word in the 'txt' attribute, the module uses the corresponding bounding boxes from 'charBB'
to crop each character using a perspective transform. Each character is stored as a separate dataset
in the output HDF5 file along with metadata including the source image name, the original word, the
cropped character, and its bounding boxes.

Output HDF5 files are generated for training, validation, and test sets.
"""

import h5py
import numpy as np
import cv2
from tqdm import tqdm

# Input HDF5 files (update as needed)
train_filename = "hdf5_files/downloaded/train.h5"
test_filename = "hdf5_files/downloaded/test.h5"

# Output HDF5 files
train_output_filename = "hdf5_files/train.h5"
val_output_filename = "hdf5_files/val.h5"
test_output_filename = "hdf5_files/test.h5"

# Parameters for character cropping (target size 224x224 with margin 18)
X_RES, Y_RES = 224, 224
X_MARGIN, Y_MARGIN = 18, 18

def crop_bb(img, bb, x_res=X_RES, y_res=Y_RES, x_margin=X_MARGIN, y_margin=Y_MARGIN):
    """
    Applies a perspective transform to crop a character from the image based on its bounding box.

    Args:
        img (np.array): The source image.
        bb (np.array): The bounding box coordinates (expected shape: [2, 4]).
        x_res (int): Target width.
        y_res (int): Target height.
        x_margin (int): Horizontal margin.
        y_margin (int): Vertical margin.

    Returns:
        np.array: The cropped character image.
    """
    dest_bb = np.array([
        [x_margin, y_margin],
        [x_res - x_margin, y_margin],
        [x_res - x_margin, y_res - y_margin],
        [x_margin, y_res - y_margin]
    ], np.float32)

    t = cv2.getPerspectiveTransform(np.float32(bb.transpose()), dest_bb)
    return cv2.warpPerspective(img, t, (x_res, y_res))

def process_dataset(db, imgs_names, fonts_included):
    """
    Processes each image in the dataset to extract individual characters using bounding boxes.
    For each word in the 'txt' attribute, it uses the 'charBB' attribute to crop individual characters.
    Each character is assigned the corresponding font based on its index (using fonts[curr_char_idx]).

    Args:
        db (h5py.File): The input HDF5 file handle.
        imgs_names (list): List of image names (keys in the 'data' group) to process.
        fonts_included (bool): Indicates whether the 'font' attribute is included in the data.

    Returns:
        tuple: (char_images, char_ids, metadata)
            char_images: List of cropped character images.
            char_ids: List of unique dataset names for each character.
            metadata: Dictionary mapping dataset names to their metadata.
    """
    metadata = {}
    char_images = []
    char_ids = []

    for img_name in tqdm(imgs_names):
        try:
            img_data = db['data'][img_name][()]
            words = db['data'][img_name].attrs['txt']
            word_bbs = db['data'][img_name].attrs['wordBB']
            char_bbs = db['data'][img_name].attrs['charBB']

            if fonts_included:
                fonts = db['data'][img_name].attrs['font']

            curr_word_idx = 0
            curr_char_idx = 0

            for word in words:
                # Convert from bytes to string if necessary
                word_str = word.decode('utf-8') if isinstance(word, bytes) else word
                # Retrieve the bounding box for the word (additional info)
                word_bb = word_bbs[:, :, curr_word_idx]

                for char in word_str:
                    # Retrieve the bounding box for the current character
                    char_bb = char_bbs[:, :, curr_char_idx]
                    cropped_char_img = crop_bb(img_data, char_bb)

                    dataset_name = f"{img_name}_{curr_char_idx}"

                    data = {
                        'src_img': img_name,
                        'src_word': word_str,
                        'char': char,
                        'word_bb': word_bb,
                        'char_bb': char_bb
                    }

                    if fonts_included:
                        # Assign the corresponding font for the character based on curr_char_idx
                        font = fonts[curr_char_idx]
                        data['font'] = font.decode('utf-8') if isinstance(font, bytes) else font

                    char_images.append(cropped_char_img)
                    char_ids.append(dataset_name)
                    metadata[dataset_name] = data

                    curr_char_idx += 1
                curr_word_idx += 1

        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")

    return char_images, char_ids, metadata

def build_dataset(output_file_path, db, imgs_names, fonts_included):
    """
    Builds a new HDF5 file with a group 'images', where each individual character is stored as a separate dataset
    along with its metadata.

    Args:
        output_file_path (str): Path to save the output HDF5 file.
        db (h5py.File): The input HDF5 file handle.
        imgs_names (list): List of image names (keys in the 'data' group) to process.
        fonts_included (bool): Indicates whether the 'font' attribute is included.
    """
    print(f"\n➡ Processing {len(imgs_names)} images for {output_file_path}...")
    char_imgs, char_ids, metadata = process_dataset(db, imgs_names, fonts_included)

    with h5py.File(output_file_path, 'w') as out_db:
        images_group = out_db.create_group('images')

        for i, dataset_name in enumerate(char_ids):
            ds = images_group.create_dataset(dataset_name, data=char_imgs[i])
            ds.attrs['src_img'] = metadata[dataset_name]['src_img']
            ds.attrs['src_word'] = metadata[dataset_name]['src_word']
            ds.attrs['char'] = metadata[dataset_name]['char']
            ds.attrs['word_bb'] = metadata[dataset_name]['word_bb']
            ds.attrs['char_bb'] = metadata[dataset_name]['char_bb']

            if fonts_included and 'font' in metadata[dataset_name]:
                ds.attrs['font'] = metadata[dataset_name]['font']

    print(f"✅ Finished building {output_file_path} ({len(char_ids)} characters)\n")

def split_list(lst, train_ratio=0.7):
    """
    Splits a list of image names (keys from the 'data' group) into training and validation sets.

    Args:
        lst (list): List of image names.
        train_ratio (float): Ratio of images to use for training.

    Returns:
        tuple: (training image names, validation image names)
    """
    np.random.shuffle(lst)
    split_point = int(len(lst) * train_ratio)
    return lst[:split_point], lst[split_point:]

def start():
    """
    Main function to process the dataset.

    This function builds new HDF5 files for training, validation, and test sets using the respective
    input HDF5 files.
    """
    # Process TRAIN + VAL
    with h5py.File(train_filename, 'r') as db_train:
        img_names = list(db_train['data'].keys())
        train_names, val_names = split_list(img_names, train_ratio=0.7)

        build_dataset(train_output_filename, db_train, train_names, fonts_included=True)
        build_dataset(val_output_filename, db_train, val_names, fonts_included=True)

    # Process TEST
    with h5py.File(test_filename, 'r') as db_test:
        test_names = list(db_test['data'].keys())
        build_dataset(test_output_filename, db_test, test_names, fonts_included=False)

if __name__ == '__main__':
    start()
