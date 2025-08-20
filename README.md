# SEBNet

## Overview
Semantic Boundary-Conditioned Network (SEBNet) is a family of real-time CNNs developed for the semantic segmentation task. SEBNet leverages principles from the Semantic Boundary Detection (SBD) task to improve the segmentation quality of a real-time architecture based on PIDNet [[1]](#1). SEBNet also leverages the OTFGT module from [[2]](#2) and pyEdgeEval from [[3]](#3) for SBD peformance improvements and evaluation, respectively.

## Description
The development of SEBNet was sequential and comprehensive. There are two stages.

### Stage 1 - Pre-training
To begin, a vanilla CNN backbone is adapted from the integral (I) branch of PIDNet. Two variations of the backbone are pre-trained on the ImageNet Dataset: 
1. The first backbone is trained traditionally, it is the vanilla backbone.
2. The second backbone is trained with an attached KoLeo Regularizer for better feature discrimination between similar features.

### Stage 2 - Finetuning
Next, a decoder is attached for the downstream semantic segmentation task. A baseline is established prior to 9 ablation studies that examine the effects of different heads. These heads either directly contribute to the dense prediction yielded by SEBNet, or they condition the backbone.
1.  **Ablation 01** - A baseline is established by attaching a pyramid pooling module (DAPPM or PAPPM) and a vanilla segmentation head.
2.  **Ablation 02** - Baseline + P Head (from PIDNet's P Branch)
3.  **Ablation 03** - Baseline + D Head (from PIDNet's D Branch)
4.  **Ablation 04** - Baseline + CASENet SBD Head
5.  **Ablation 05** - Baseline + DFF SBD Head
6.  **Ablation 06** - Baseline + BEM SBD Head
7.  **Ablation 07** - Baseline + P Head + D Head
8.  **Ablation 08** - Baseline + P Head + SBD Head
9.  **Ablation 09** - Baseline + D Head + SBD Head
10. **Ablation 10** - Baseline + P Head + D Head + BAS Loss (PIDNet)
11. **Ablation 11** - Baseline + P Head + D Head + SBD Head
12. **Ablation 12** - Baseline + P Head + D Head + SBD Head + BAS Loss (PIDNet + SBD)
13. **Ablation 13** - Best Model + Mapillary Pre-training

## Results
Results are pending. The target date for segmentation results is September 15th, 2025. The target date for boundary results is October 15th, 2025. Model weights for the best performing networks will be uploaded following experimentation.

## References
<a id="1">[1]</a> 
J. Xu, Z. Xiong, and S. P. Bhattacharyya, "PIDNet: A real-time semantic segmentation network inspired by PID controllers." https://doi.org/10.48550/arXiv.2206.02066

<a id="2">[2]</a>
H. Ishikawa, Y. Aoki, "Boosting Semantic Segmentation by Conditioning the Backbone with Semantic Boundaries." Sensors (Basel). 2023 Aug 6;23(15):6980. doi: https://doi.org/10.3390/s23156980

<a id="3">[3]</a>
H. Ishikawa, "pyEdgeEval: Python Edge Evaluation Tools." 2022. https://github.com/haruishi43/py-edge-eval
