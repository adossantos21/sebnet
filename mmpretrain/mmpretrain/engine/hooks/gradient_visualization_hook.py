import os
import matplotlib.pyplot as plt
import numpy as np
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class GradFlowVisualizationHook(Hook):
    """Hook for visualizing gradient flow during training.
    
    This hook visualizes the gradients flowing through different layers 
    of the network during training to help identify gradient vanishing 
    or exploding problems.
    
    Args:
        interval (int): Interval for plotting gradient flow. Default: 100.
        save_dir (str, optional): Directory to save gradient flow plots. 
            If None, plots will be saved in runner.work_dir. Default: None.
        show_plot (bool): Whether to display plots. Default: False.
        priority (int or str): Priority of the hook. Default: 'NORMAL'.
    """
        
    def __init__(self, 
                 interval=100, 
                 initial_grads=True,
                 out_dir=None, 
                 show_plot=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.interval = interval
        self.initial_grads = initial_grads
        self.out_dir = out_dir
        self.show_plot = show_plot
        
    def plot_grad_flow(self, named_parameters, save_path=None):
        """Plot gradient flow through network layers.
        
        Args:
            named_parameters: Iterator of (name, parameter) tuples from model
            save_path (str, optional): Path to save the plot
        """
        ave_grads, max_grads, layers = [], [], []
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())

        if not layers:  # No gradients to plot
            print(f"No gradients to plot")
            return
            
        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.8, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.8, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads)), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("Gradient magnitude")
        plt.title("Gradient Flow")
        plt.legend(['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if self.show_plot:
            plt.show()
        else:
            plt.close()
    
    def after_backward_pass(self, runner, batch_idx, data_batch=None, outputs=None):
        """Hook method called after each training iteration.
        
        Args:
            runner: The runner of the training process
            batch_idx: The index of the current batch in the train loop
            data_batch: Data from dataloader
            outputs: Outputs from model
        """
        if self.every_n_train_iters(runner, self.interval):
            # Determine save directory
            if self.out_dir is None:
                save_dir = runner.work_dir
            else:
                save_dir = self.out_dir
                
            os.makedirs(save_dir, exist_ok=True)
            
            # Create save path with iteration number
            save_path = os.path.join(
                save_dir, 
                f'gradient_flow_iter_{runner.iter + 1}.png'
            )
            

            # Plot gradient flow
            self.plot_grad_flow(
                runner.model.named_parameters(), 
                save_path=save_path
            )
            
            runner.logger.info(
                f'Gradient flow plot saved to {save_path} at iteration {runner.iter + 1}'
            )

        if self.initial_grads==True and runner.iter==100:
            # Determine save directory
            print(f"runner.iter: {runner.iter}")
            if self.out_dir is None:
                save_dir = runner.work_dir
            else:
                save_dir = self.out_dir
                
            os.makedirs(save_dir, exist_ok=True)
            
            # Create save path with iteration number
            save_path = os.path.join(
                save_dir, 
                f'gradient_flow_iter_{runner.iter}.png'
            )
            

            # Plot gradient flow
            self.plot_grad_flow(
                runner.model.named_parameters(), 
                save_path=save_path
            )
            
            runner.logger.info(
                f'Gradient flow plot saved to {save_path} at iteration {runner.iter}'
            )
