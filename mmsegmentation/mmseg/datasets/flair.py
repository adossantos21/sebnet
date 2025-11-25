# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FLAIRDataset(BaseSegDataset):
    """FLAIR dataset.

    The French Land cover from Aerospace ImageRy (FLAIR) dataset is an 
    extensive dataset from the French National Institute of Geographical 
    and Forest Information (IGN) for land cover semantic segmentation.
    It contains high-resolution aerial imagery with 19 semantic classes.
    
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_MSK.png' for FLAIR dataset.
    """
    METAINFO = dict(
        classes=('building', 'pervious surface', 'impervious surface', 
                 'bare soil', 'water', 'coniferous', 'deciduous', 
                 'brushwood', 'vineyard', 'herbaceous vegetation', 
                 'agricultural land', 'plowed land', 'swimming_pool', 
                 'snow', 'clear cut', 'mixed', 'ligneous', 'greenhouse', 
                 'other'),
        palette=[[219, 14, 154], [147, 142, 123], [248, 12, 0],
                 [169, 113, 1], [21, 83, 174], [25, 74, 38],
                 [70, 228, 131], [243, 166, 13], [102, 0, 130],
                 [85, 255, 0], [255, 243, 13], [228, 223, 124],
                 [61, 230, 235], [255, 255, 255], [138, 179, 160],
                 [107, 113, 79], [197, 220, 66], [153, 153, 255],
                 [0, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_MSK.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)