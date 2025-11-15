"""
Calculating distance threshold:
Given an input image with resolution 2048x1024:

1. Calculate the image diagonal
   Image diagonal = sqrt((2048^2)+(1024^2)) = 2289.73
2. Select a pixel tolerance.
   Pixel tolerance = 3 pixels
3. Calculate the distance threshold
   Distance threshold = 3 / 2289.73 = 0.001

Thus for an input with 2048x1024 resolution, and a pixel tolerance of 3 pixels, bd_thresh = 0.001
"""

from utils.bfscore import BFScore
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate BFScore')
    parser.add_argument('preds_path', type=str, help='the path to the predictions')
    parser.add_argument('masks_path', type=str, help='the path to the ground truth masks')
    parser.add_argument('--num_classes', type=int, default=19, help='the number of classes in the predictions and ground truths')
    parser.add_argument('--px_tol', type=int, default=3, help='the pixel tolerance for computing BFScore')
    parser.add_argument('--num_proc', type=int, default=10, help='the number of processes to spawn')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    metrics = BFScore(
        pred_path=args.preds_path, 
        mask_path=args.masks_path,
        num_classes=args.num_classes,
        num_proc=args.num_proc,
        px_tolerance=args.px_tol
        )
    results = metrics.forward()
    print(results)

if __name__ == "__main__":
    main()