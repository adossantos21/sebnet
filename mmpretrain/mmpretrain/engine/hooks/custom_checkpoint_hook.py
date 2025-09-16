import os
import os.path as osp
from collections import deque
from mmengine.hooks import CheckpointHook
from mmengine.registry import HOOKS
from mmengine.fileio import FileClient, get_file_backend
from mmengine.dist import is_main_process
from math import inf

'''
Using custom checkpoint hook to change the checkpoint save path from
`runner.work_dir` to osp.join(`runner._log_dir`, 'checkpoints').
'''
@HOOKS.register_module()
class CustomCheckpointHook(CheckpointHook):
    def before_train(self, runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = osp.join(runner._log_dir, 'checkpoints')
            os.makedirs(self.out_dir, exist_ok=True)

        # If self.file_client_args is None, self.file_client will not
        # used in CheckpointHook. To avoid breaking backward compatibility,
        # it will not be removed util the release of MMEngine1.0
        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        if self.file_client_args is None:
            self.file_backend = get_file_backend(
                self.out_dir, backend_args=self.backend_args)
        else:
            self.file_backend = self.file_client

        runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                if 'best_ckpt' not in runner.message_hub.runtime_info:
                    self.best_ckpt_path = None
                else:
                    self.best_ckpt_path = runner.message_hub.get_info(
                        'best_ckpt')
            else:
                for key_indicator in self.key_indicators:
                    best_ckpt_name = f'best_ckpt_{key_indicator}'
                    if best_ckpt_name not in runner.message_hub.runtime_info:
                        self.best_ckpt_path_dict[key_indicator] = None
                    else:
                        self.best_ckpt_path_dict[
                            key_indicator] = runner.message_hub.get_info(
                                best_ckpt_name)

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = []
            if 'keep_ckpt_ids' in runner.message_hub.runtime_info:
                keep_ckpt_ids = runner.message_hub.get_info('keep_ckpt_ids')

                while len(keep_ckpt_ids) > self.max_keep_ckpts:
                    step = keep_ckpt_ids.pop(0)
                    if is_main_process():
                        path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(step))
                        if self.file_backend.isfile(path):
                            self.file_backend.remove(path)
                        elif self.file_backend.isdir(path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(path)

            self.keep_ckpt_ids: deque = deque(keep_ckpt_ids,
                                              self.max_keep_ckpts)
    
    def _save_checkpoint_with_step(self, runner, step, meta):
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step))

                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info('keep_ckpt_ids',
                                               list(self.keep_ckpt_ids))

        ckpt_filename = self.filename_tmpl.format(step)
        self.last_ckpt = self.file_backend.join_path(self.out_dir,
                                                     ckpt_filename)
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args)

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return

        save_file = osp.join(runner._log_dir, 'checkpoints', 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(self.last_ckpt)  # type: ignore

    def _save_best_checkpoint(self, runner, metrics):
        """Save the best checkpoint with a fixed filename, overwriting previous."""
        if not is_main_process():
            return

        best_ckpt_updated = False

        for key_indicator, rule in zip(self.key_indicators, self.rules):
            key_score = metrics.get(key_indicator, None)
            if key_score is None:
                runner.logger.warning(f'The key "{key_indicator}" is not found in metrics, skip save best checkpoint for it.')
                continue

            best_score_key = f'best_score_{key_indicator}' if len(self.key_indicators) > 1 else 'best_score'

            if best_score_key not in runner.message_hub.runtime_info:
                best_score = self.init_value_map[rule]
                runner.message_hub.update_info(best_score_key, best_score)
            else:
                best_score = runner.message_hub.get_info(best_score_key)

            if not self.is_better_than[key_indicator](key_score, best_score):
                continue

            best_ckpt_updated = True

            runner.message_hub.update_info(best_score_key, key_score)

            ckpt_filename = f'best_{key_indicator}.pth'
            ckpt_path = self.file_backend.join_path(self.out_dir, ckpt_filename)

            # Remove previous best checkpoint if it exists
            if self.file_backend.exists(ckpt_path):
                is_removed = False
                if self.file_backend.isfile(ckpt_path):
                    self.file_backend.remove(ckpt_path)
                    is_removed = True
                elif self.file_backend.isdir(ckpt_path):
                    self.file_backend.rmtree(ckpt_path)
                    is_removed = True
                if is_removed:
                    runner.logger.info(
                        f'The previous best checkpoint {ckpt_path} is removed')

            meta = dict(epoch=runner.epoch, iter=runner.iter)

            runner.save_checkpoint(
                self.out_dir,
                filename=ckpt_filename,
                file_client_args=self.file_client_args,
                save_optimizer=True,
                save_param_scheduler=True,
                meta=meta,
                by_epoch=False,
                backend_args=self.backend_args,
                **self.args
            )

            if len(self.key_indicators) == 1:
                self.best_ckpt_path = ckpt_path
                runner.message_hub.update_info('best_ckpt', self.best_ckpt_path)
            else:
                self.best_ckpt_path_dict[key_indicator] = ckpt_path
                runner.message_hub.update_info(f'best_ckpt_{key_indicator}', self.best_ckpt_path_dict[key_indicator])

            runner.logger.info(
                f'Now best checkpoint for {key_indicator} is saved as {ckpt_filename} '
                f'with {key_indicator} = {key_score:.4f}.'
            )

        if best_ckpt_updated and self.last_ckpt is not None:
            if self.by_epoch:
                step = runner.epoch + 1
            else:
                step = runner.iter + 1
            meta = dict(epoch=runner.epoch, iter=runner.iter)
            self._save_checkpoint_with_step(runner, step, meta)