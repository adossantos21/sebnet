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

import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
from math import floor, sqrt

class BFScore:
    def __init__(
        self,
        pred_path: str,
        mask_path: str,
        num_classes: int = 19,
        num_proc: int = 10,
        image_height: int = 1024,
        image_width: int = 2048,
        px_tolerance: int = 3,
        ):
        assert isinstance(pred_path, str), f"pred_path should be of type string; instead it is of type: {type(pred_path)}"
        assert isinstance(mask_path, str), f"pred_path should be of type string; instead it is of type: {type(pred_path)}"
        self.preds = [os.path.join(pred_path, f) for f in sorted(os.listdir(pred_path))]
        self.masks = [os.path.join(mask_path, f) for f in sorted(os.listdir(mask_path))]
        assert len(self.preds) == len(self.masks), f"The total amount of preds should match the total amount of masks; instead, self.preds length is {len(self.preds)} while self.masks length is {len(self.masks)}"
        self.num_classes = num_classes
        self.num_proc = num_proc
        self.bd_thresh = round(float(px_tolerance / sqrt((image_height**2)+(image_width**2))), 4)
    def forward(self):
        bfscores = []
        counts = []
        for pred, mask in tqdm(zip(self.preds, self.masks), desc="Processing BFScores", total=len(self.preds)):
            seg_mask = np.expand_dims(np.array(Image.open(pred)), axis=0)
            gt_mask = np.expand_dims(np.array(Image.open(mask)), axis=0)
            fpc, fc = self.eval_mask_boundary(seg_mask, gt_mask, self.num_classes, self.num_proc, self.bd_thresh)
            bfscores.append(fpc)
            counts.append(fc)
        all_bf = np.array(bfscores) # [num_images, num_classes]
        all_counts = np.array(counts) # [num_images, num_classes]
        # Compute per-class means, handling zero counts
        per_class_bf = np.zeros(self.num_classes)
        for cl in range(self.num_classes):
            valid_images = all_counts[:, cl] > 0
            if np.any(valid_images):
                per_class_bf[cl] = np.sum(all_bf[valid_images, cl]) / np.sum(all_counts[valid_images, cl])
            else:
                per_class_bf[cl] = 0 # or np.nan if preferred
        mean_bf = per_class_bf.mean() # Mean over classes
        return mean_bf, per_class_bf
 
    def eval_mask_boundary(self, seg_mask,gt_mask,num_classes,num_proc=10,bound_th=0.008):
        """
        Compute F score for a segmentation mask
        Arguments:
            seg_mask (ndarray): segmentation mask prediction
            gt_mask (ndarray): segmentation mask ground truth
            num_classes (int): number of classes
        Returns:
            F (float): mean F score across all classes
            Fpc (listof float): F score per class
        """
        p = Pool(processes=num_proc)
        batch_size = seg_mask.shape[0]
     
        Fpc = np.zeros(num_classes)
        Fc = np.zeros(num_classes)
        for class_id in tqdm(range(num_classes)):
            args = [((seg_mask[i] == class_id).astype(np.uint8),
                    (gt_mask[i] == class_id).astype(np.uint8),
                    gt_mask[i] == 255,
                    bound_th)
                    for i in range(batch_size)]
            temp = p.map(self.db_eval_boundary_wrapper, args)
            temp = np.array(temp)
            Fs = temp[:,0]
            _valid = ~np.isnan(Fs)
            Fc[class_id] = np.sum(_valid)
            Fs[np.isnan(Fs)] = 0
            Fpc[class_id] = sum(Fs)
        return Fpc, Fc
    def db_eval_boundary_wrapper(self, args):
        foreground_mask, gt_mask, ignore, bound_th = args
        return self.db_eval_boundary(foreground_mask, gt_mask,ignore, bound_th)
    def db_eval_boundary(self, foreground_mask,gt_mask, ignore_mask,bound_th=0.008):
        """
        Compute mean,recall and decay from per-frame evaluation.
        Calculates precision/recall for boundaries between foreground_mask and
        gt_mask using morphological operators to speed it up.
        Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask (ndarray): binary annotated image.
        Returns:
            F (float): boundaries F-measure
            P (float): boundaries precision
            R (float): boundaries recall
        """
        assert np.atleast_3d(foreground_mask).shape[2] == 1
        bound_pix = bound_th if bound_th >= 1 else \
                np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))
        #print(bound_pix)
        #print(gt.shape)
        #print(np.unique(gt))
        foreground_mask[ignore_mask] = 0
        gt_mask[ignore_mask] = 0
        # Get the pixel boundaries of both masks
        fg_boundary = self.seg2bmap(foreground_mask);
        gt_boundary = self.seg2bmap(gt_mask);
        from skimage.morphology import binary_dilation,disk
        fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
        gt_dil = binary_dilation(gt_boundary,disk(bound_pix))
        # Get the intersection
        gt_match = gt_boundary * fg_dil
        fg_match = fg_boundary * gt_dil
        # Area of the intersection
        n_fg = np.sum(fg_boundary)
        n_gt = np.sum(gt_boundary)
        #% Compute precision and recall
        if n_fg == 0 and n_gt == 0:
            precision = 1
            recall = 1
        elif n_fg == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_fg > 0 and n_gt == 0:
            precision = 0
            recall = 1
        else:
            precision = np.sum(fg_match)/float(n_fg)
            recall = np.sum(gt_match)/float(n_gt)
        # Compute F measure
        if precision + recall == 0:
            F = 0
        else:
            F = 2*precision*recall/(precision+recall);
        return F, precision, recall
    def seg2bmap(self, seg,width=None,height=None):
        """
        From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries. The boundary pixels are offset by 1/2 pixel towards the
        origin from the actual segment boundary.
        Arguments:
            seg : Segments labeled from 1..k.
            width : Width of desired bmap <= seg.shape[1]
            height : Height of desired bmap <= seg.shape[0]
        Returns:
            bmap (ndarray): Binary boundary map.
        David Martin <dmartin@eecs.berkeley.edu>
        January 2003
    """
        seg = seg.astype(bool)
        seg[seg>0] = 1
        assert np.atleast_3d(seg).shape[2] == 1
        width = seg.shape[1] if width is None else width
        height = seg.shape[0] if height is None else height
        h,w = seg.shape[:2]
        ar1 = float(width) / float(height)
        ar2 = float(w) / float(h)
        assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
                'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)
        e = np.zeros_like(seg)
        s = np.zeros_like(seg)
        se = np.zeros_like(seg)
        e[:,:-1] = seg[:,1:]
        s[:-1,:] = seg[1:,:]
        se[:-1,:-1] = seg[1:,1:]
        b = seg^e | seg^s | seg^se
        b[-1,:] = seg[-1,:]^e[-1,:]
        b[:,-1] = seg[:,-1]^s[:,-1]
        b[-1,-1] = 0
        if w == width and h == height:
            bmap = b
        else:
            bmap = np.zeros((height,width))
            for x in range(w):
                for y in range(h):
                    if b[y,x]:
                        j = 1+floor((y-1)+height / h)
                        i = 1+floor((x-1)+width / h)
                        bmap[j,i] = 1;
        return bmap