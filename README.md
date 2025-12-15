# HiC-SuperNet
# A Multi-Scale Dual-Attention Network for Enhanced Hi-C Chromatin Interaction Matrix Resolution

# 📌 Overview
This research proposes HiC-SuperNet, a novel deep learning architecture for enhancing the resolution of Hi-C chromatin interaction matrices. Our method addresses critical limitations in existing approaches by introducing:
1. Multi-Scale Dilated Residual Blocks with three parallel dilation rates (d=1,2,4) for comprehensive receptive field coverage.
2. Dual Attention Mechanism combining channel and spatial attention for better feature selection and localization.
3. Biologically Informed Multi-Component Loss Function incorporating MSE, MAE, Pearson correlation, and SSIM for structural preservation

# Architecture Pipeline
![Result Image](https://github.com/proloy190902/HiC-SuperNet/blob/5c428b4721c0879e3f39ec5cebc66621281d6163/HiC-SuperNet%20Architecture.png)

# 📂 Dataset
This work uses data from the DiCARN-DNase project.

Source: OluwadareLab / DiCARN_DNase (Zenodo)

Format: Pre-processed .npz files

Task: 10 kb (LR) → 40 kb (HR) Hi-C resolution enhancement

Matrix Size: 40 × 40

Channels: Single-channel Hi-C matrices

# Implementation Details

| Parameter           | Value          | Justification                                     |
| ------------------- | -------------- | ------------------------------------------------- |
| Base filters        | 64             | Balance between model capacity and efficiency     |
| Number of blocks    | 8              | Sufficient depth without overfitting              |
| Dilation rates      | 1, 2, 4        | Covers short- to long-range genomic interactions  |
| Attention frequency | Every 2 blocks | Reduces computational overhead                    |
| Learning rate       | 0.001          | Ensures stable and efficient convergence          |
| Batch size          | 16             | Limited by available GPU memory                   |
| Optimizer           | Adam           | Adaptive learning rate for faster convergence     |
| Training epochs     | 100            | Combined with early stopping to avoid overfitting |

# Dataset
For testing purpose, we used  GM12878 and HMEC cell line Dataset
  Resolution: 10kb (HR) downsampled to 40kb (LR)
  Training: Chromosomes 1-22 (excluding 4,14,16,20)
  Validation: Chromosomes 2,6,10,12
  Testing: Chromosomes 4,14,16,20
  
  Samples: 190,000 training matrices (40×40 patches)

