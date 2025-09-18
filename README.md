# SEBNet

## Overview

Semantic Boundary-Conditioned Network (SEBNet) is a family of real-time CNNs developed for the semantic segmentation task. SEBNet leverages principles from the Semantic Boundary Detection (SBD) task to improve the segmentation quality of a real-time architecture based on PIDNet [[1]](#1). SEBNet also leverages the OTFGT module from [[2]](#2) and pyEdgeEval from [[3]](#3) for SBD peformance improvements and evaluation, respectively.

## Installation

**This step is required.**

You have two options for software setup:

- Install dependencies via the [virtual environment](install/virt_env/install.md) approach
- Install dependencies via the [Docker](install/docker/reproduction/docker_reproduction.md) approach 

## Quick Evaluation

If you've already generated your SBD predictions and ground truth labels, and you wish to evaluate your edge metrics solely, build your container via the [SBD Docker Image](install/docker/sbd_evaluation/docker_evaluate_sbd.md) approach.

Otherwise, download weights from the table below and follow [evaluate.md](docs/evaluate.md) to generate SBD predictions and ground truth labels.

**Without Mapillary Pre-training:**

   | Model (Cityscapes)              | Val mIoU (%)                                                                  | Test mIoU (%)                                                                    |  FPS  |
   |---------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------|-------|
   | BaselinePDSBDBASHead            | [81.1](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes/20250906_102604/checkpoints/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes/20250906_102604/best_mIoU.pth) | [Pending](https://github.com/adossantos21/paper_2) |  31.1 |
   | ConditionalBaselinePSBDBASHead  | [80.7](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250906_102650/checkpoints/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250906_102650/best_mIoU.pth) | [Pending](https://github.com/adossantos21/paper_2) |  35.4 |
   | BaselinePDBASHead (PIDNet, Ours)      | [80.5](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/checkpoints/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/best_mIoU.pth)                              | [Pending](https://github.com/adossantos21/paper_2)                         |  31.1 |

## Reproducing Experiments / Training your own models

For pre-training and fine-tuning ablations, see [reproduction.md](docs/reproduction.md).

## Description

The development of SEBNet was sequential and comprehensive. There are two stages.

### Stage 1 - Pre-training

To begin, a vanilla CNN backbone is adapted from the integral (I) branch of PIDNet: 

1. The vanilla backbone is trained on ImageNet-1K traditionally.

### Stage 2 - Finetuning

Next, a decoder is attached for the downstream semantic segmentation task. A baseline is established prior to 9 ablation studies that examine the effects of different heads. These heads either directly contribute to the dense prediction yielded by SEBNet, or they condition the backbone.

1.  **Ablation 01** - A baseline is established by attaching a pyramid pooling module (DAPPM or PAPPM) and a vanilla segmentation head.
2.  **Ablation 02** - Baseline + P Head (from PIDNet's P Branch)
3.  **Ablation 03** - Baseline + D Head (from PIDNet's D Branch), Edge Width 2, BD Loss Weight 5.0
4.  **Ablation 04** - Baseline + CASENet SBD Head, Edge Width 2, SBD Loss Weight 5.0
5.  **Ablation 05** - Baseline + DFF SBD Head, Edge Width 2, SBD Loss Weight 5.0
6.  **Ablation 06** - Baseline + BEM SBD Head, Edge Width 2, SBD Loss Weight 5.0
7.  **Ablation 07** - Baseline + D Multi-Label SBD Head, Edge Width 2, SBD Loss Weight 5.0
8.  **Ablation 08** - Baseline + D Earlier Layers Head (from PIDNet's D Branch), Edge Width 2, BD Loss Weight 5.0
9.  **Ablation 09** - Baseline + CASENet Earlier Layers SBD Head, Edge Width 2, SBD Loss Weight 5.0
10. **Ablation 10** - Baseline + DFF Earlier Layers SBD Head, Edge Width 2, SBD Loss Weight 5.0
11. **Ablation 11** - Baseline + BEM Earlier Layers SBD Head, Edge Width 2, SBD Loss Weight 5.0
12. **Ablation 12** - Baseline + D Multi-Label Earlier Layers SBD Head, Edge Width 2, SBD Loss Weight 5.0
13. **Ablation 13** - Baseline + SBD Head, Edge Width 1, SBD Loss Weight 5.0
14. **Ablation 14** - Baseline + SBD Head, Edge Width 4, SBD Loss Weight 5.0
15. **Ablation 15** - Baseline + SBD Head, Edge Width 8, SBD Loss Weight 5.0
16. **Ablation 16** - Baseline + SBD Head, Best Edge Width, SBD Loss Weight 1.0
17. **Ablation 17** - Baseline + SBD Head, Best Edge Width, SBD Loss Weight 10.0
18. **Ablation 18** - Baseline + SBD Head, Best Edge Width, SBD Loss Weight 20.0
19. **Ablation 19** - Baseline + P Head + D Head
20. **Ablation 20** - Baseline + D Head + SBD Head
21. **Ablation 21** - Baseline + P Head + SBD Head (Conditioning)
22. **Ablation 22** - Baseline + P Head + SBD Head (Fusion)
23. **Ablation 23** - Baseline + P Head + SBD Head + BAS Loss (Conditioning)
24. **Ablation 24** - Baseline + P Head + SBD Head + BAS Loss (Fusion)
25. **Ablation 25** - Baseline + P Head + D Head + BAS Loss (PIDNet)
26. **Ablation 26** - Baseline + P Head + D Head + SBD Head
27. **Ablation 27** - Baseline + P Head + D Head + SBD Head + BAS Loss (PIDNet + SBD)
28. **Ablation 28** - Best Model + Mapillary Pre-training

## Results

Results are pending. The target date for segmentation results is September 18th, 2025. The target date for boundary results is October 10th, 2025. Model weights for the best performing networks will be uploaded following experimentation.

## References

<a id="1">[1]</a> 
J. Xu, Z. Xiong, and S. P. Bhattacharyya, "PIDNet: A real-time semantic segmentation network inspired by PID controllers." https://doi.org/10.48550/arXiv.2206.02066

<a id="2">[2]</a>
H. Ishikawa, Y. Aoki, "Boosting Semantic Segmentation by Conditioning the Backbone with Semantic Boundaries." Sensors (Basel). 2023 Aug 6;23(15):6980. doi: https://doi.org/10.3390/s23156980

<a id="3">[3]</a>
H. Ishikawa, "pyEdgeEval: Python Edge Evaluation Tools." 2022. https://github.com/haruishi43/py-edge-eval
