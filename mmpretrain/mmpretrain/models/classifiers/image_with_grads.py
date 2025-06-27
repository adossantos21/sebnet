from .image import ImageClassifier
from mmpretrain.registry import MODELS
from typing import Union, Dict
import torch
from mmengine.optim.optimizer import OptimWrapper

@MODELS.register_module()
class ImageClassifierWithGrads(ImageClassifier):
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation. Rather than calling 
        optim_wrapper.update_params() (see mmengine/model/base_model/
        base_model.py), we update params line-by-line to explicitly
        call a hook that triggers gradient plotting. IterBasedTrainingLoop
        or EpochBasedTrainingLoop calls :meth::`train_step` in 
        mmengine/runner/loops.py, which we are overwriting here. You'll
        need to create and register a new training loop that accepts the
        new arguments seen in this overriden method.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars, parsed_losses