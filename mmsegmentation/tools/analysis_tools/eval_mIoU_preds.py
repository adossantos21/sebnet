import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchmetrics.classification import MulticlassJaccardIndex

def parse_args():
    parser = argparse.ArgumentParser(description='Compute mIoU from generated dense predictions')
    parser.add_argument('--pred_dir', 
                        type=str, 
                        required=True,
                        help='Path to predictions')
    parser.add_argument('--gt_dir', 
                        type=str, 
                        required=True,
                        help='Path to ground truth masks')
    parser.add_argument('--cityscapes', 
                        action='store_true', 
                        default=False, 
                        help='whether cityscapes is the dataset')
    return parser.parse_args()

def compute_miou(pred_dir, gt_dir, cityscapes=True, num_classes=19, ignore_label=255):
    """
    Compute mean Intersection over Union (mIoU) for semantic segmentation predictions using torchmetrics.

    Requires PyTorch and torchmetrics to be installed.
    """
    # Initialize the metric with average=None to get per-class IoUs
    metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_label,
        average=None
    )
    
    # List all ground truth files
    if cityscapes:
        gt_files = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir)) if f.endswith('_gtFine_labelTrainIds.png')]
    else:
        gt_files = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir))]
    
    pred_files = [os.path.join(pred_dir, f) for f in sorted(os.listdir(pred_dir))]

    
    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), desc="Processing Confusion Matrix", total=len(gt_files)):
        
        if not os.path.exists(pred_file):
            print(f"Warning: Missing prediction for {pred_file}. Skipping.")
            continue
        
        # Load images as numpy arrays
        gt = np.array(Image.open(gt_file))
        pred = np.array(Image.open(pred_file))
        
        if gt.shape != pred.shape:
            print(f"Warning: Shape mismatch for {gt_file}. Skipping.")
            continue
        
        # Convert to torch tensors
        gt_tensor = torch.from_numpy(gt).long()
        pred_tensor = torch.from_numpy(pred).long()
        
        # Update the metric
        metric.update(pred_tensor, gt_tensor)
    
    # Compute per-class IoUs
    ious = metric.compute()
    
    # Handle NaNs for classes with no ground truth
    ious_np = ious.numpy()  # Convert to numpy for easier handling
    
    # Compute mIoU as nanmean
    miou = np.nanmean(ious_np)
    
    # Print per-class IoU and mIoU
    print("Per-class IoU:")
    for i in range(num_classes):
        val = ious_np[i]
        print(f"Class {i}: {val:.4f}" if not np.isnan(val) else f"Class {i}: N/A")
    
    print(f"\nMean IoU (mIoU): {miou:.4f}")
    
    return miou

def main():
    args = parse_args()
    
    mIoU = compute_miou(args.pred_dir, args.gt_dir, cityscapes=args.cityscapes)
    print(f"mIoU: {mIoU}")

if __name__ == "__main__":
    main()
