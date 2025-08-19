import re
import matplotlib.pyplot as plt

# Function to parse the log file
def parse_log_file(log_file_path):
    epochs = []
    top1_accuracies = []
    top5_accuracies = []
    learning_rates = []

    curr_epoch = None
    last_lr_for_epoch = None

    with open(log_file_path, 'r') as f:
        for line in f:
            # Parse training epoch learning rate
            train_lr_match = re.search(r"Epoch\(train\)\s*\[(\d+)\].*?lr: ([0-9\.e+-]+)", line)
            if train_lr_match:
                curr_epoch = int(train_lr_match.group(1))
                last_lr_for_epoch = float(train_lr_match.group(2))
                continue

            # Parse validation accuracy at end of val epoch
            val_acc_match = re.search(r"Epoch\(val\) \[(\d+)\].*accuracy/top1: ([0-9\.]+)\s+accuracy/top5: ([0-9\.]+)", line)
            if val_acc_match:
                epoch = int(val_acc_match.group(1))
                top1 = float(val_acc_match.group(2))
                top5 = float(val_acc_match.group(3))
                # Save values
                epochs.append(epoch)
                top1_accuracies.append(top1)
                top5_accuracies.append(top5)
                # Save learning rate for the epoch
                # Assuming last_lr_for_epoch corresponds to the training epoch learning rate used
                learning_rates.append(last_lr_for_epoch)

    return epochs, top1_accuracies, top5_accuracies, learning_rates

# Function to plot the accuracies and learning rate
def plot_accuracies_and_lr(epochs, top1, top5, lrs, out_path):
    fig, ax1 = plt.subplots()
    
    ax1.set_ylabel('Learning Rate')  
    line1, = ax1.plot(epochs, lrs, label='Learning Rate', color='tab:green')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    line2, = ax2.plot(epochs, top1, label='Top-1 Accuracy', color='tab:blue')
    line3, = ax2.plot(epochs, top5, label='Top-5 Accuracy', color='tab:orange')
    ax2.set_ylim(0, 100)

    # Combine all lines and labels into one legend
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), framealpha=0.8)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Validation Accuracy and Training Learning Rate vs Epoch')
    plt.savefig(out_path)

log_file_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170401/20250721_170401.log'
out_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170401/log_plot.png'
epochs, top1, top5, lrs = parse_log_file(log_file_path)
plot_accuracies_and_lr(epochs, top1, top5, lrs, out_path)

log_file_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170430/20250721_170430.log'
out_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170430/log_plot.png'
epochs, top1, top5, lrs = parse_log_file(log_file_path)
plot_accuracies_and_lr(epochs, top1, top5, lrs, out_path)

log_file_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170610/20250721_170610.log'
out_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170610/log_plot.png'
epochs, top1, top5, lrs = parse_log_file(log_file_path)
plot_accuracies_and_lr(epochs, top1, top5, lrs, out_path)

log_file_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170637/20250721_170637.log'
out_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170637/log_plot.png'
epochs, top1, top5, lrs = parse_log_file(log_file_path)
plot_accuracies_and_lr(epochs, top1, top5, lrs, out_path)

log_file_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170837/20250721_170837.log'
out_path = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain01_staged_lr_adamw_1xb64_in1k/20250721_170837/log_plot.png'
epochs, top1, top5, lrs = parse_log_file(log_file_path)
plot_accuracies_and_lr(epochs, top1, top5, lrs, out_path)