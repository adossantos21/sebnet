import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert CamVid color labels.")
    parser.add_argument('data_root', type=str, help='Path to the CamVid root directory (e.g., datasets/CamVid).')
    return parser.parse_args()

def convert_rgb_to_index(label_path, output_path, palette):
    """Convert an RGB label image to single-channel index image in 'P' mode."""
    rgb_label = Image.open(label_path).convert('RGB')
    rgb_array = np.array(rgb_label)

    # Convert to index array
    h, w, _ = rgb_array.shape
    index_array = np.full((h, w), 255, dtype=np.uint8)  # Default to ignore_index (255)
    for idx, color in palette.items():
        mask = np.all(rgb_array == color, axis=-1)
        index_array[mask] = idx

    # Save as 'P' mode PNG
    index_img = Image.fromarray(index_array, mode='P')
    flattened_palette = np.array(list(palette.values())).flatten().tolist()
    flattened_palette += [0] * (3 * (256 - len(palette)))
    index_img.putpalette(flattened_palette)
    index_img.save(output_path)

def main():
    args = parse_args()
    # Define paths based on the CamVid directory structure
    data_root = args.data_root
    train_label_dir = os.path.join(data_root, 'train_labels')
    val_label_dir = os.path.join(data_root, 'val_labels')
    train_label_gray_dir = os.path.join(data_root, 'train_labels_gray')
    val_label_gray_dir = os.path.join(data_root, 'val_labels_gray')

    # Create output directories if they don't exist
    os.makedirs(train_label_gray_dir, exist_ok=True)
    os.makedirs(val_label_gray_dir, exist_ok=True)

    # Palette from CamVidDataset METAINFO (flattened for Pillow)
    palette = {
        0: [128, 128, 128],  # sky
        1: [128, 0, 0],      # building
        2: [192, 192, 128],  # pole
        3: [128, 64, 128],   # road
        4: [0, 0, 192],      # pavement
        5: [128, 128, 0],    # tree
        6: [192, 128, 128],  # signsymbol
        7: [64, 64, 128],    # fence
        8: [64, 0, 128],     # car
        9: [64, 64, 0],      # pedestrian
        10: [0, 128, 192]     # bicyclist
    }

    # Process train_labels
    for label_file in tqdm(os.listdir(train_label_dir), desc="Converting train labels", total=len(os.listdir(train_label_dir))):
        if label_file.endswith('.png'):
            label_path = os.path.join(train_label_dir, label_file)
            output_path = os.path.join(train_label_gray_dir, label_file)
            convert_rgb_to_index(label_path, output_path, palette)

    # Process val_labels
    for label_file in tqdm(os.listdir(val_label_dir), desc="Converting val labels", total=len(os.listdir(val_label_dir))):
        if label_file.endswith('.png'):
            label_path = os.path.join(val_label_dir, label_file)
            output_path = os.path.join(val_label_gray_dir, label_file)
            convert_rgb_to_index(label_path, output_path, palette)

if __name__ == "__main__":
    main()