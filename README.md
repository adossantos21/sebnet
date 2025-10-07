# SEBNet

## Overview

Semantic Boundary-Conditioned Network (SEBNet) is a family of real-time CNNs developed for the semantic segmentation task. SEBNet leverages principles from the Semantic Boundary Detection (SBD) task to improve the segmentation quality of a real-time architecture based on PIDNet [[1]](#1). SEBNet also leverages the OTFGT module from [[2]](#2) and pyEdgeEval from [[3]](#3) for SBD peformance improvements and evaluation, respectively.

## Installation

**This step is required.**

You have a few options for software setup:

- Install dependencies via the [virtual environment](install/virt_env/install.md) approach
- Install dependencies via the [Docker](install/docker/reproduction/docker_reproduction.md) approach
   - If you've already generated your HED/SBD predictions, and you only wish to evaluate edge-based metrics, build a container via the [SBD Docker Image](install/docker/sbd_evaluation/docker_evaluate_sbd.md) approach.

## Quick Evaluation

Download weights from the table below and follow [evaluate.md](docs/evaluate.md).

**Best Results so far, Without Mapillary Pre-training:**

   | Model (Cityscapes)              | Val mIoU (%)                                                                  | Test mIoU (%)                                                                    |  FPS  |
   |---------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------|-------|
   | Ablation 40                     | [81.1](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes/20250906_102604/checkpoints/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes/20250906_102604/best_mIoU.pth) | [Pending](https://github.com/adossantos21/paper_2) |  31.1 |
   | Ablation 33                     | [80.7](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250906_102650/checkpoints/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250906_102650/best_mIoU.pth) | [Pending](https://github.com/adossantos21/paper_2) |  35.4 |
   | Ablation 31                     | [80.5](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/checkpoints/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/best_mIoU.pth)                              | [Pending](https://github.com/adossantos21/paper_2)                         |  31.1 |

## Reproducing Experiments / Training your own models

For pre-training and fine-tuning ablations, see [reproduction.md](docs/reproduction.md).

## Description

Nomenclature:

- Holistically-Nested Edge Detection [(HED)](https://arxiv.org/pdf/1504.06375.pdf)
- Semantic Boundary Detection [(SBD)](https://arxiv.org/pdf/1705.09759)
- Boundary Awareness [(BAS)](https://arxiv.org/pdf/2206.02066)

The development of SEBNet was sequential and comprehensive. There are two stages.

### Stage 1 - Pre-training

To begin, a vanilla CNN backbone is adapted from the integral (I) branch of PIDNet: 

1. The vanilla backbone is trained on ImageNet-1K traditionally.

### Stage 2 - Finetuning

Next, a decoder is attached for the downstream semantic segmentation task. A baseline is established prior to 40 ablation studies that examine the effects of different heads. These heads either directly contribute to the dense prediction yielded by SEBNet, or they condition the backbone. An asterisk indicates the ablation study is complete.
#### <ins>Section 01 - Train Baseline</ins>

1.  **Ablation 01*** - A baseline is established by attaching a pyramid pooling module (DAPPM or PAPPM) and a vanilla segmentation head.

#### <ins>Section 02 - Find best P Head</ins>

Assume best P Head following this section.

2.  **Ablation 02*** - Baseline + P Head (from PIDNet's P Branch)
3.  **Ablation 03** - Baseline + P Head (Pag1 supervised, conditioning only)
4.  **Ablation 04** - Baseline + P Head (Pag2 supervised, conditioning only)
5.  **Ablation 05** - Baseline + P Head (Last layer supervised, conditioning only)

#### <ins>Section 03 - Find best Edge Head</ins>

Following this section, "Edge Head" will be the best edge architecture determined during these ablation studies. Note that any given head can have multiple loss signals, e.g., HED loss signal, SBD loss signal, BAS loss signal, OHEM (variant of Cross Entropy) loss signal, etc.

6.  **Ablation 06*** - Baseline + Edge Head (from PIDNet's D Branch), HED Signal, Edge Width 2, BD Loss Weight 5.0
7.  **Ablation 07*** - Baseline + CASENet Head, SBD Signal, Edge Width 2, SBD Loss Weight 5.0
8.  **Ablation 08*** - Baseline + DFF Head, SBD Signal, Edge Width 2, SBD Loss Weight 5.0
9.  **Ablation 09*** - Baseline + BEM Head, SBD Signal Edge Width 2, SBD Loss Weight 5.0
10. **Ablation 10*** - Baseline + Edge Head (from PIDNet's D Branch), SBD Signal, Edge Width 2, SBD Loss Weight 5.0
11. **Ablation 11*** - Baseline + Edge Earlier Layers Head (from PIDNet's D Branch), HED Signal, Edge Width 2, BD Loss Weight 5.0
12. **Ablation 12*** - Baseline + CASENet Earlier Layers Head, SBD Signal, Edge Width 2, SBD Loss Weight 5.0
13. **Ablation 13*** - Baseline + DFF Earlier Layers Head, SBD Signal, Edge Width 2, SBD Loss Weight 5.0
14. **Ablation 14*** - Baseline + BEM Earlier Layers Head, SBD Signal, Edge Width 2, SBD Loss Weight 5.0
15. **Ablation 15*** - Baseline + Edge Earlier Layers Head (from PIDNet's D Branch), SBD Signal, Edge Width 2, SBD Loss Weight 5.0
16. **Ablation 16*** - Baseline + Best Edge Head, SBD Signal, Edge Width 1, SBD Loss Weight 5.0
17. **Ablation 17*** - Baseline + Best Edge Head, SBD Signal, Edge Width 4, SBD Loss Weight 5.0
18. **Ablation 18*** - Baseline + Best Edge Head, SBD Signal, Edge Width 8, SBD Loss Weight 5.0
19. **Ablation 19*** - Baseline + Best Edge Head, SBD Signal, Best Edge Width, SBD Loss Weight 1.0
20. **Ablation 20*** - Baseline + Best Edge Head, SBD Signal, Best Edge Width, SBD Loss Weight 10.0
21. **Ablation 21*** - Baseline + Best Edge Head, SBD Signal, Best Edge Width, SBD Loss Weight 20.0

#### <ins>Section 04 - Condition vs Fusion Grid Search, 1 Edge Head with 1-2 signals (HED and/or SBD)</ins>

***Conditioning*** involves attaching an auxiliary head and corresponding signal to your baseline architecture during training. While the model trains, learned features from the auxiliary head backwards propagate into the backbone of your baseline, creating a shared representation. During evaluation, the shared representation preserved in the backbone allows us to detach the auxiliary head, preventing latency overhead. "(Conditioned)" will indicate that the corresponding head was only used to backwards propagate learned features into the backbone. *All edge heads in the prior section were tested via conditioning*.

***Fusion*** also involves attaching an auxiliary head and corresponding signal to your baseline architecture during training. The main difference being that the deepest features of the auxiliary head are used in a feature fusion module (FFM) to create the shared representation. During evaluation, the auxiliary head is not detached, introducing latency overhead. "(Fused)" will indicate that the corresponding head's deepest features were fused with the output of the backbone via a FFM.

22. **Ablation 22*** - Baseline + Edge Head, HED and SBD Signals for Edge Head
23. **Ablation 23** - Baseline + P Head (Conditioned) + Edge Head (Conditioned), HED Signal, *Appendix ablation, since SBD Signal proved to be more effective than HED Signal by itself in Section 03*
24. **Ablation 24** - Baseline + P Head (Fused) + Edge Head (Conditioned), HED Signal, *Appendix ablation, since SBD Signal proved to be more effective than HED Signal by itself in Section 03*
25. **Ablation 25*** - Baseline + P Head (Fused) + Edge Head (Fused), HED Signal, *Appendix ablation, since SBD Signal proved to be more effective than HED Signal by itself in Section 03*
26. **Ablation 26** - Baseline + P Head (Conditioned) + Edge Head (Conditioned), SBD Signal
27. **Ablation 27*** - Baseline + P Head (Fused) + Edge Head (Conditioned), SBD Signal
28. **Ablation 28*** - Baseline + P Head (Fused) + Edge Head (Fused), SBD Signal
29. **Ablation 29** - Baseline + P Head (Conditioned) + Edge Head (Conditioned), HED and BAS Signals, *Appendix ablation, since SBD Signal proved to be more effective than HED Signal by itself in Section 03*
30. **Ablation 30** - Baseline + P Head (Fused) + Edge Head (Conditioned), HED and BAS Signals, *Appendix ablation, since SBD Signal proved to be more effective than HED Signal by itself in Section 03*
31. **Ablation 31*** - Baseline + P Head (Fused) + D Head (Fused), HED and BAS Signals (PIDNet equivalent), *Needed for study comparison*
32. **Ablation 32** - Baseline + P Head (Conditioned) + Edge Head (Conditioned), SBD and BAS Signals
33. **Ablation 33*** - Baseline + P Head (Fused) + Edge Head (Conditioned), SBD and BAS Signals
34. **Ablation 34*** - Baseline + P Head (Fused) + Edge Head (Fused), SBD and BAS Signals
35. **Ablation 35** - Baseline + P Head (Conditioned) + Edge Head (Conditioned), HED and SBD Signals for Edge Head
36. **Ablation 36** - Baseline + P Head (Fused) + Edge Head (Conditioned), HED and SBD Signals for Edge Head
37. **Ablation 37*** - Baseline + P Head (Fused) + Edge Head (Fused), HED and SBD Signals for Edge Head
38. **Ablation 38** - Baseline + P Head (Conditioned) + Edge Head (Conditioned), HED and SBD Signals for Edge Head, BAS Signal for Semantic Head
39. **Ablation 39** - Baseline + P Head (Fused) + Edge Head (Conditioned), HED and SBD Signals for Edge Head, BAS Signal for Semantic Head
40. **Ablation 40*** - Baseline + P Head (Fused) + Edge Head (Fused), HED and SBD Signals for Edge Head, BAS Signal for Semantic Head (PIDNet + SBD Signal)

#### <ins>Section 05 - Condition vs Fusion Grid Search, 2 Edge Heads (HED and SBD) and their respective signals</ins>

41. **Ablation 41** - Baseline + Edge Head (HED) + Edge Head (SBD), HED and SBD signals for respective heads.
42. **Ablation 42** - Baseline + P Head (Conditioned) + Edge Head (Conditioned, HED) + Edge Head (Conditioned, SBD), HED and SBD signals for respective heads.
43. **Ablation 43** - Baseline + P Head (Fused) + Edge Head (Conditioned, HED) + Edge Head (Conditioned, SBD), HED and SBD signals for respective heads.
44. **Ablation 44** - Baseline + P Head (Fused) + Edge Head (Fused, HED) + Edge Head (Conditioned, SBD), HED and SBD signals for respective heads.
45. **Ablation 45** - Baseline + P Head (Fused) + Edge Head (Conditioned, HED) + Edge Head (Fused, SBD), HED and SBD signals for respective heads.
46. **Ablation 46** - Baseline + P Head (Conditioned) + Edge Head (Conditioned, HED) + Edge Head (Conditioned, SBD), HED and SBD signals for respective heads, BAS signal for Semantic Head.
47. **Ablation 47** - Baseline + P Head (Fused) + Edge Head (Conditioned, HED) + Edge Head (Conditioned, SBD), HED and SBD signals for respective heads, BAS signal for Semantic Head.
48. **Ablation 48** - Baseline + P Head (Fused) + Edge Head (Fused, HED) + Edge Head (Conditioned, SBD), HED and SBD signals for respective heads, BAS signal for Semantic Head.
49. **Ablation 49** - Baseline + P Head (Fused) + Edge Head (Conditioned, HED) + Edge Head (Fused, SBD), HED and SBD signals for respective heads, BAS signal for Semantic Head.

#### <ins>Section 06 - Mapillary

50. **Ablation 50** - Best Model + Mapillary Pre-training

## Results

Results are pending. The target date for all results is October 10th, 2025. Model weights for the best performing networks will be uploaded following experimentation.

## References

<a id="1">[1]</a> 
J. Xu, Z. Xiong, and S. P. Bhattacharyya, "PIDNet: A real-time semantic segmentation network inspired by PID controllers." https://doi.org/10.48550/arXiv.2206.02066

<a id="2">[2]</a>
H. Ishikawa, Y. Aoki, "Boosting Semantic Segmentation by Conditioning the Backbone with Semantic Boundaries." Sensors (Basel). 2023 Aug 6;23(15):6980. doi: https://doi.org/10.3390/s23156980

<a id="3">[3]</a>
H. Ishikawa, "pyEdgeEval: Python Edge Evaluation Tools." 2022. https://github.com/haruishi43/py-edge-eval
