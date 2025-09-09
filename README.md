# SEBNet

## Overview
Semantic Boundary-Conditioned Network (SEBNet) is a family of real-time CNNs developed for the semantic segmentation task. SEBNet leverages principles from the Semantic Boundary Detection (SBD) task to improve the segmentation quality of a real-time architecture based on PIDNet [[1]](#1). SEBNet also leverages the OTFGT module from [[2]](#2) and pyEdgeEval from [[3]](#3) for SBD peformance improvements and evaluation, respectively.

## Getting Started

### Install dependencies by following [install.md](https://github.com/adossantos21/paper_2/blob/main/install.md). This is required.

### If you wish to simply download the weights and evaluate performance:
1. Download the weights:

   **Without Mapillary Pre-training:**
   | Model (Cityscapes)              | Val mIoU (%)                                                                  | Test mIoU (%)                                                                    |  FPS  |
   |---------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------|-------|
   | BaselinePDSBDBASHead            | [81.1](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes/20250906_102604/checkpoints/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes/20250906_102604/best_mIoU.pth) | [Pending](https://github.com/<your-username>/<your-repo>/raw/main/largefile.ext) |  31.1 |
   | ConditionalBaselinePSBDBASHead  | [80.7](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250906_102650/checkpoints/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250906_102650/best_mIoU.pth) | [Pending](https://github.com/<your-username>/<your-repo>/raw/main/otherfile.ext) |  35.4 |
   | BaselinePDBASHead (PIDNet, Ours)      | [80.5](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/checkpoints/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/best_mIoU.pth)                              | [Pending](https://github.com/adossantos21/paper_2)                         |  31.1 |
   
3. Activate the conda environment you created from [install.md](https://github.com/adossantos21/paper_2/blob/main/install.md):
   ```
   conda activate venv_sebnet
   ```
#### For semantic edge detection (SBD) evaluation
1. Enter the `mmsegmentation` directory:
   ```
   cd mmsegmentation/
   ```
2. Generate your SBD predictions
    - Configure `generate_cityscapes_preds.sh`
        - Set the model config file
        - Set the relevant model's checkpoint
        - Set the output path for your SBD predictions
    - Run the executable:
      ```
      ./generate_cityscapes_preds.sh
      ```
3. Generate your SBD ground truth
    - Your environment will have installed the packages required for the following shell commands.
    - Configure `generate_cityscapes_gt.sh`
        - Set the root path of your cityscapes dataset
        - (Optional) - Set the output path *relative* to the cityscapes dataset root path
    - Run the executable:
      ```
      ./generate_cityscapes_gt.sh
      ```
4. Navigate to the parent directory
   ```
   cd ../
   ```
5. Configure `eval_edgeMetrics.sh`
    - Add `-h` flag to see optional arguments and corresponding descriptions.
6. Run the executable:
   ```
   ./eval_edgeMetrics.sh
   ```
#### For image classification or semantic segmentation evaluation
1. Navigate to either `mmpretrain/` or `mmsegmentation/` directory.
2. Find the corresponding config file path, e.g., `configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes.py`
3. In `./eval.sh`, update <config_file_path> with the path from the previous step. Also, add the checkpoint weights `.pth` file. eval.sh, which should look something like:
   ```
   export CUDA_VISIBLE_DEVICES=0
   python tools/test.py \
       <config_file_path> \
       <ckpt.pth>
   ```
4. Run the executable:
   ```
   ./eval.sh
   ```
### If you wish to repeat the experiments outlined in the Description section:
**To pretrain a model:**
1. Navigate to the `mmpretrain/` directory:
   ```
   cd mmpretrain/
   ```
2. Find the config file path, e.g., `configs/alex_sebnet/pretrain01_tests/pretrain01_staged_1xb64_in1k.py`
3. Update <config_file_path> with the path from the previous step in `./main.sh`, which should look something like:
   ```
   export CUDA_VISIBLE_DEVICES=0
   python tools/train.py <config_file_path>
   ```
4. Activate the conda environment you created from [install.md](https://github.com/adossantos21/paper_2/blob/main/install.md):
   ```
   conda activate venv_sebnet
   ```
5. Run the executable:
   ```
   ./main.sh
   ```

**To finetune a model:**
1. Navigate to the `mmsegmentation/` directory:
   ```
   cd mmsegmentation/
   ```
2. Find the config file path, e.g., `configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes.py`
3. Update <config_file_path> with the path from the previous step in `./main.sh`, which should look something like:
   ```
   export CUDA_VISIBLE_DEVICES=0
   python tools/train.py <config_file_path>
   ```
4. Activate the conda environment you created from [install.md](https://github.com/adossantos21/paper_2/blob/main/install.md):
   ```
   conda activate venv_sebnet
   ```
5. Run the executable:
   ```
   ./main.sh
   ```

## Description
The development of SEBNet was sequential and comprehensive. There are two stages.

### Stage 1 - Pre-training
To begin, a vanilla CNN backbone is adapted from the integral (I) branch of PIDNet. Two variations of the backbone are pre-trained on the ImageNet Dataset: 
1. The first backbone is trained traditionally, it is the vanilla backbone.
2. The second backbone is trained with an attached KoLeo Regularizer for better feature discrimination between similar features.

### Stage 2 - Finetuning
Next, a decoder is attached for the downstream semantic segmentation task. A baseline is established prior to 9 ablation studies that examine the effects of different heads. These heads either directly contribute to the dense prediction yielded by SEBNet, or they condition the backbone.
1.  **Ablation 01** - A baseline is established by attaching a pyramid pooling module (DAPPM or PAPPM) and a vanilla segmentation head.
2.  **Ablation 02** - Baseline + CASENet SBD Head, Edge Width 2, SBD Loss Weight 5.0 (Default)
3.  **Ablation 03** - Baseline + DFF SBD Head, Edge Width 2, SBD Loss Weight 5.0 (Default)
4.  **Ablation 04** - Baseline + BEM SBD Head, Edge Width 2, SBD Loss Weight 5.0 (Default)
5.  **Ablation 05** - Baseline + MIMIR SBD Head, Edge Width 2, SBD Loss Weight 5.0 (Default)
6.  **Ablation 06** - Baseline + SBD Head, Edge Width 1, SBD Loss Weight 5.0
7.  **Ablation 07** - Baseline + SBD Head, Edge Width 4, SBD Loss Weight 5.0
8.  **Ablation 08** - Baseline + SBD Head, Edge Width 8, SBD Loss Weight 5.0
9.  **Ablation 09** - Baseline + SBD Head, Best Edge Width, SBD Loss Weight 1.0
10. **Ablation 10** - Baseline + SBD Head, Best Edge Width, SBD Loss Weight 10.0
11. **Ablation 11** - Baseline + SBD Head, Best Edge Width, SBD Loss Weight 20.0
12. **Ablation 12** - Baseline + D Head (from PIDNet's D Branch)
13. **Ablation 13** - Baseline + P Head (from PIDNet's P Branch)
14. **Ablation 14** - Baseline + P Head + D Head
15. **Ablation 15** - Baseline + D Head + SBD Head
16. **Ablation 16** - Baseline + P Head + SBD Head (Conditioning)
17. **Ablation 17** - Baseline + P Head + SBD Head (Fusion)
18. **Ablation 18** - Baseline + P Head + SBD Head + BAS Loss (Conditioning)
19. **Ablation 19** - Baseline + P Head + SBD Head + BAS Loss (Fusion)
20. **Ablation 20** - Baseline + P Head + D Head + BAS Loss (PIDNet)
21. **Ablation 21** - Baseline + P Head + D Head + SBD Head
22. **Ablation 22** - Baseline + P Head + D Head + SBD Head + BAS Loss (PIDNet + SBD)
23. **Ablation 23** - Best Model + Mapillary Pre-training

## Results
Results are pending. The target date for segmentation results is September 18th, 2025. The target date for boundary results is October 10th, 2025. Model weights for the best performing networks will be uploaded following experimentation.

## References
<a id="1">[1]</a> 
J. Xu, Z. Xiong, and S. P. Bhattacharyya, "PIDNet: A real-time semantic segmentation network inspired by PID controllers." https://doi.org/10.48550/arXiv.2206.02066

<a id="2">[2]</a>
H. Ishikawa, Y. Aoki, "Boosting Semantic Segmentation by Conditioning the Backbone with Semantic Boundaries." Sensors (Basel). 2023 Aug 6;23(15):6980. doi: https://doi.org/10.3390/s23156980

<a id="3">[3]</a>
H. Ishikawa, "pyEdgeEval: Python Edge Evaluation Tools." 2022. https://github.com/haruishi43/py-edge-eval
