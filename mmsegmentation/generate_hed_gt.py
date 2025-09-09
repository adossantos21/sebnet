import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Process Cityscapes validation edge maps.")
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the Cityscapes validation subdirectory (e.g., gtFine/val).')
    parser.add_argument('--input_suffix', type=str, required=True, help='Suffix of input edge map files (e.g., "_gtProc_raw_edge.png").')
    parser.add_argument('--output_suffix', type=str, required=True, help='Suffix for output binary edge map files (e.g., "_gtProc_raw_hed_edge.png").')
    return parser.parse_args()

def unpack_edge_map(image_path):
    """
    Unpacks a multi-label edge map image into K binary edge maps.
    
    Args:
        image_path (str): Path to the edge map image.
    
    Returns:
        np.ndarray: Binary edge maps with shape (K, H, W), where K=19 for Cityscapes.
    """
    img = Image.open(image_path)
    rgb = np.array(img)  # Shape: (H, W, 3), dtype uint8
    
    # Threshold rgb
    binary = (rgb > 0).astype(np.uint8)

    # Roll axis and take max to yield binary edge map
    binary = np.max(np.rollaxis(binary, 2),axis=0)

    return binary

def main():
    args = parse_args()
    
    # Subdirectories in Cityscapes val
    subdirs = ['frankfurt', 'lindau', 'munster']
    
    # Collect all input files
    input_files = []
    for subdir in subdirs:
        dir_path = os.path.join(args.val_dir, subdir)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(args.input_suffix):
                    input_files.append(os.path.join(dir_path, file))
    
    # Process with tqdm for progress
    for input_path in tqdm(input_files, desc="Processing edge maps", unit="file"):
        # Unpack to (K, H, W)
        binary_map = unpack_edge_map(input_path) * 255
        
        # Extract basename without input_suffix
        basename = os.path.basename(input_path).replace(args.input_suffix, '')
        
        # Output path: same directory, basename + output_suffix
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, basename + args.output_suffix)
        
        # Save as PNG
        Image.fromarray(binary_map).save(output_path)

if __name__ == "__main__":
    main()