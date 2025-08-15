# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class FeatureMapVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 interval: int = 50,
                 initial_maps: bool = True,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.initial_maps = initial_maps
        
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self._test_index = 0

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        '''
        if self.initial_maps==True and runner.iter>=50:
            # Determine save directory
            print(f"data_batch.keys(): {data_batch.keys()}")
            # Feature maps are definitely in outputs.data key.
            import sys
            sys.exit()
            feature_maps = "feature_maps"
            if self.out_dir is None:
                save_dir = osp.join(runner._log_dir, feature_maps)
            else:
                basename = osp.basename(runner.work_dir.rstrip(osp.sep))
                date = osp.basename(runner._log_dir.rstrip(osp.sep))
                self.out_dir = osp.join(self.out_dir, basename, date, feature_maps)
                save_dir = self.out_dir
                
            os.makedirs(save_dir, exist_ok=True)

        # Create save path with iteration number
            save_path = osp.join(
                save_dir, 
                f'gradient_flow_epoch_{runner.epoch}_iter_{runner.iter + 1}.png'
            )
            
            

            # Save feature maps
            self.save_feature_maps(
                runner.model.named_parameters(), 
                save_path=save_path
            )

        if self.every_n_train_iters(runner, self.interval):
            # Determine save directory
            feature_maps = "feature_maps"
            if self.out_dir is None:
                save_dir = osp.join(runner._log_dir, feature_maps)
            else:
                basename = osp.basename(runner.work_dir.rstrip(osp.sep))
                date = osp.basename(runner._log_dir.rstrip(osp.sep))
                self.out_dir = osp.join(self.out_dir, basename, date, feature_maps)
                save_dir = self.out_dir
                
            os.makedirs(save_dir, exist_ok=True)

        # Create save path with iteration number
            save_path = osp.join(
                save_dir, 
                f'gradient_flow_epoch_{runner.epoch}_iter_{runner.iter + 1}.png'
            )

            # Save feature maps
            self.save_feature_maps(
                runner.model.named_parameters(), 
                save_path=save_path
            )
        '''

