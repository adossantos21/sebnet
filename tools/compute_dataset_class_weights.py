'''
File Name: compute_dataset_class_weights.py
Author: Alessandro Dos Santos
Description:
The original Cityscapes Dataset's class weights for balancing class imbalance are:
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]
To compute class weights for a custom dataset, we need to compute the number of pixels for each class in the dataset, 
then apply the weights_log function from OCNet to compute the class weights.
NOTE: See https://github.com/openseg-group/OCNet.pytorch/issues/14
'''

import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import torch
from typing import Dict
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_custom_path', action='store_true', help='use path for dataset to compute per class pixel count, otherwise use OCNet default per class pixel count.')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--dataset_dir', type=str, default='/path/to/cityscapes/gtFine/train/*/*_labelTrainIds.png')
    parser.add_argument('--ignore_index', type=int, default=255)
    args = parser.parse_args()
    return args

def weights_log(counts: Dict[str, int], num_classes: int) -> torch.Tensor:
    class_freq = torch.FloatTensor(list(counts.values()))
    weights = 1 / torch.log1p(class_freq)
    weights = num_classes * weights / torch.sum(weights)
    return weights

def main():
    # Initialize args
    args = parse_args()

    if args.use_custom_path:
        # Find all labelTrainIds files in the train split
        label_files = glob.glob(args.dataset_dir)

        num_images = len(label_files)
        if num_images == 0:
            raise ValueError("No labelTrainIds files found in the directory.")

        # Initialize counts for each class (0-18)
        final_counts = {}

        # Loop through each label image
        for file in tqdm(label_files, desc='Processing labels', total=num_images):
            # Load the label image (grayscale)
            img = np.array(Image.open(file))
            h, w = img.shape
            unique, counts = np.unique(img, return_counts=True)
            for u, c in zip(unique, counts):
                if f'class_{u}' not in final_counts:
                    final_counts[f'class_{u}'] = 0
                final_counts[f'class_{u}'] += c

        # Sort the dict by class id (u)
        final_counts = OrderedDict(sorted(final_counts.items(), key=lambda x: int(x[0].split("_")[1])))
        print(f"Class counts: {final_counts}")

        # Delete class_255 (background)
        if f'class_{args.ignore_index}' in final_counts:
            print(f"Deleting class {args.ignore_index}")
            del final_counts[f'class_{args.ignore_index}']
    else:
        cityscapes_per_class_pixel_count = [ # see https://github.com/openseg-group/OCNet.pytorch/issues/14
            2.01e+9, 2.98e+8, 9.96e+8, 3.39e+7, 4.50e+7, 6.54e+7,
            9.57e+7, 2.62e+7, 7.21e+8, 5.92e+7, 1.45e+8, 8.21e+7,
            1.00e+7, 4.13e+8, 1.45e+7, 1.28e+7, 1.45e+7, 5.64e+6, 2.57e+7
            ]
        final_counts = {f'class_{i}': int(cityscapes_per_class_pixel_count[i]) for i in range(len(cityscapes_per_class_pixel_count))}
        print(f"Class counts: {final_counts}")
    
    # Calculate log weights
    weights = weights_log(final_counts, args.num_classes)
    print(f"Class weights: {weights}")

if __name__ == "__main__":
    main()
