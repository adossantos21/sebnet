from mmengine.runner import IterBasedTrainLoop
from mmseg.registry import LOOPS
from typing import Sequence

@LOOPS.register_module()
class GradientsFeaturesIterTrainLoop(IterBasedTrainLoop):
    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss, returned from decode head.
        # logits should be a dict of logits, returned from decode head.
        outputs, logits = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        
        # update_params is where self.backward() is called. So the hook needs to be called after self.backward()
        self.runner.call_hook(
            'after_backward_pass', batch_idx=self._iter, data_batch=data_batch)
        
        if self.runner.optim_wrapper.should_update():
            self.runner.optim_wrapper.step()
            self.runner.optim_wrapper.zero_grad()
        
        # self.runner.call_hook() will call all hooks associated with the first argument.
        # If you make edits to one of the default Hook methods, e.g., 'after_train_iter', 
        # in a custom hook that inherits from Hook, all other hooks that use the default
        # Hook method must also be edited. For example, if FeatureMapVisualizationHook
        # added `logits` as an argument to 'after_train_iter', `logits` would be added to
        # CheckpointHook in mmengine. Thus, we create a new Hook method 
        # 'custom_train_iter'.
        self.runner.call_hook(
            'custom_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs,
            logits=logits
        )
        
        self.runner.call_hook( 
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1