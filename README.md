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

**Best Results on Cityscapes:**

   | Model (Cityscapes)              | Val mIoU (%)                                                                  | Test mIoU (%)                                                                    |  FPS  |
   |---------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------|-------|
   | Ablation 20                     | [81.6](https://github.com/adossantos21/sebnet/blob/main/mmsegmentation/checkpoints/ablation20/val/best_mIoU.pth) | [80.9](https://github.com/adossantos21/sebnet/blob/main/mmsegmentation/checkpoints/ablation20/test/best_mIoU.pth) |  60.5 |
   | Ablation 12                     | [81.0](https://github.com/adossantos21/sebnet/blob/main/mmsegmentation/checkpoints/ablation12/val/best_mIoU.pth) | [Pending](https://github.com/adossantos21/sebnet/blob/main/mmsegmentation/checkpoints/ablation12/test/best_mIoU.pth) |  69.1 |

## Reproducing Experiments / Training your own models

For pre-training and fine-tuning ablations, see [reproduction.md](docs/reproduction.md).

## Description

Nomenclature:

- Holistically-Nested Edge Detection [(HED)](https://arxiv.org/pdf/1504.06375.pdf)
- Semantic Boundary Detection [(SBD)](https://arxiv.org/pdf/1705.09759)
- Boundary Awareness [(BAS)](https://arxiv.org/pdf/2206.02066)

The development of SEBNet was sequential and comprehensive. There are two stages. You will find the tables below correspond to the config files used to reproduce results.

### Stage 1 - Pre-training

To begin, a vanilla CNN backbone is adapted from the integral (I) branch of PIDNet: 

1. The vanilla backbone is trained on ImageNet-1K traditionally.

### Stage 2 - Finetuning

Next, a decoder is attached for the downstream semantic segmentation task. A baseline is established prior to 19 ablation studies that examine the effects of different heads. These heads either directly contribute to the dense prediction yielded by SEBNet, or they condition the backbone.

#### <ins>Section 01 - Establish baseline and test detailed head</ins>

A baseline is established by attaching a pyramid pooling module (DAPPM or PAPPM) and a vanilla segmentation head. Following baseline evaluation, we attach an auxiliary head that preserves detailed features. The auxiliary head is taken from PIDNet's P Branch.

| Ablation | Backbone | Detailed Head | mIoU |
|----------|----------|---------------|------|
| Ablation 01 | SEBNet | N/A | 72.7 |
| Ablation 02 | SEBNet | PIDNet P Branch | 79.0 |

#### <ins>Section 02 - Identify best edge-based architecture</ins>

Similar to the detailed head testing, edge heads and corresponding signals are attached to the baseline architecture to determine the best edge-based architecture. For example, CASENet is the edge head, while SBD is the learning signal.

| Ablation | Backbone | Edge Head | mIoU | mF (ODS) |
|----------|----------|-----------|------|----------|
| Ablation 01 | SEBNet | N/A | 72.7 | - |
| Ablation 03 | SEBNet | CASENet, SBD | 71.0 | 60.0 (SBD) |
| Ablation 04 | SEBNet | DFF, SBD | 72.5 | 63.5 (SBD) |
| Ablation 05 | SEBNet | PIDNet D Branch, HED | 73.2 | 81.7 (HED) |
| Ablation 06 | SEBNet | PIDNet D Branch, SBD | **74.8** | **67.6 (SBD)** |
| Ablation 07 | SEBNet | PIDNet D Branch, HED, SBD | 74.7 | **82.4 (HED)**<br>67.3 (SBD) |

#### <ins>Section 03 - Find the best configuration</ins>

We select the PIDNet D Branch as our best edge-based architecture and perform more ablation studies to determine the best overall model. Two approaches are taken. The first uses conditioning to learn edge features while simultaneously reducing latency overhead. The second fuses edge features, introducing latency overhead.

A dash for the PAG column indicates no detailed head and corresponding signal were attached to the backbone.

| Ablation | Conditioning | Fusion | PAG | HED | SBD | BAS | mIoU | mF (ODS) | FPS |
|----------|:------------:|:------:|:---:|:---:|:---:|:---:|:----:|:---------|:---:|
| Ablation 07 | ✓ | - | - | ✓ | ✓ | - | 74.7 | 82.4 (HED)<br>67.3 (SBD) | **96.8** |
| Ablation 08 | ✓ | - | - | ✓ | ✓ | ✓ | 75.0 | 82.4 (HED)<br>67.8 (SBD) | 95.8 |
| Ablation 09 | ✓ | - | ✓ | ✓ | - | - | 79.6 | **82.6** (HED) | 68.7 |
| Ablation 10 | ✓ | - | ✓ | ✓ | - | ✓ | 80.0 | **82.6** (HED) | 68.9 |
| Ablation 11 | ✓ | - | ✓ | - | ✓ | - | 79.6 | 67.9 (SBD) | 68.6 |
| Ablation 12 | ✓ | - | ✓ | - | ✓ | ✓ | **80.4** | 67.7 (SBD) | 68.9 |
| Ablation 13 | ✓ | - | ✓ | ✓ | ✓ | - | 80.1 | 82.2 (HED)<br>**68.4** (SBD) | 69.0 |
| Ablation 14 | ✓ | - | ✓ | ✓ | ✓ | ✓ | 80.2 | 82.3 (HED)<br>67.8 (SBD) | 68.6 |
| Ablation 15 | - | ✓ | ✓ | ✓ | - | - | 79.6 | 82.3 (HED) | 60.2 |
| Ablation 16 | - | ✓ | ✓ | ✓ | - | ✓ | 80.0 | 82.4 (HED) | 60.3 |
| Ablation 17 | - | ✓ | ✓ | - | ✓ | - | 79.8 | 67.9 (SBD) | 60.2 |
| Ablation 18 | - | ✓ | ✓ | - | ✓ | ✓ | 80.1 | 68.3 (SBD) | 60.3 |
| Ablation 19 | - | ✓ | ✓ | ✓ | ✓ | - | 79.8 | **82.6** (HED)<br>68.1 (SBD) | 60.4 |
| Ablation 20 | - | ✓ | ✓ | ✓ | ✓ | ✓ | 80.3 | 82.5 (HED)<br>67.9 (SBD) | 60.5 |

#### <ins>Section 04 - Extend trainings, then finetune on Mapillary pre-trained weights</ins>

We extended the trainings from 160K iterations to 240K iterations for the two best ablation studies (12 and 20) and found state-of-the-art performance on the Cityscapes dataset. Mapillary pre-training further improved results.

| Ablation | Mapillary Pre-training | mIoU | mF (ODS) | bFScore | FPS |
|----------|:----------------------:|:----:|:--------:|:--------|:---:|
| Ablation 12 | - | 80.7 | 67.7 (SBD) | 74.2 | **69.1** |
| Ablation 20 | - | 81.1 | **82.4** (HED)<br>69.1 (SBD) | 75.0 | 60.5 |
| Ablation 12 | ✓ | 81.0 | **69.5** (SBD) | **75.4** | **69.1** |
| Ablation 20 | ✓ | **81.6** | 82.3 (HED)<br>68.8 (SBD) | 75.0 | 60.5 |


## References

<a id="1">[1]</a> 
J. Xu, Z. Xiong, and S. P. Bhattacharyya, "PIDNet: A real-time semantic segmentation network inspired by PID controllers." https://doi.org/10.48550/arXiv.2206.02066

<a id="2">[2]</a>
H. Ishikawa, Y. Aoki, "Boosting Semantic Segmentation by Conditioning the Backbone with Semantic Boundaries." Sensors (Basel). 2023 Aug 6;23(15):6980. doi: https://doi.org/10.3390/s23156980

<a id="3">[3]</a>
H. Ishikawa, "pyEdgeEval: Python Edge Evaluation Tools." 2022. https://github.com/haruishi43/py-edge-eval
