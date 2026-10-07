# MLSI: Multi-scale Low-frequency Structural Injection for Burst Image Denoising

This repository contains the implementation of the paper:

> **Multi-Scale Low-Frequency Structural Injection for Burst Image Denoising**  
> Jinyan Lin, Xinyi Sun, Kun Wang, Yong He, Yikun Liu, Jianying Zhou, Jubo Zhu

## Overview

Burst image denoising recovers a clean image from multiple noisy observations of the same scene. Existing learning-based methods usually capture inter-frame relationships implicitly through feature extraction, alignment, or adaptive aggregation, and may not explicitly expose structural relations across frames.

We propose a **Multi-scale Low-frequency Structural Injection (MLSI)** mechanism that constructs explicit inter-frame structural cues from the grayscale burst and hierarchically injects them into a denoising network. Given a burst sequence, MLSI:

1. Converts each frame to grayscale.
2. Extracts multi-scale low-frequency representations via average pooling.
3. Computes three complementary maps between a reference frame and each non-reference frame:
   - **Mean map** — structural consensus;
   - **Difference map** — inter-frame variation;
   - **Normalized similarity map** — local structural consistency.
4. Processes these maps with lightweight convolutions and channel attention.
5. Injects them into corresponding encoder stages through residual fusion.

The original feature extraction pathway is retained, so MLSI complements rather than replaces learned features.

## Code Base

This implementation is developed based on the open-source KPN reproduction repository:

> **KPN_Denoising_Pytorch**  
> https://github.com/WenxueCui/KPN_Denoising_Pytorch  
> by Wenxue Cui

That repository is a PyTorch reimplementation of:

> B. Mildenhall, J. T. Barron, J. Chen, D. Sharlet, R. Ng, R. Carroll,  
> *Burst Denoising with Kernel Prediction Networks*, CVPR 2018.

Starting from that codebase, we introduce the proposed MLSI mechanism. The main modifications include:

- Adding a separate grayscale burst branch for multi-scale low-frequency structural map construction;
- Computing mean, difference, and normalized similarity maps between the reference frame and each non-reference frame at multiple scales;
- Processing these maps with lightweight convolutions and channel attention;
- Injecting the processed structural features into corresponding encoder stages through residual fusion;
- Retaining the original KPN feature extraction and kernel prediction pipeline.


## Dataset
We evaluate MLSI on the **Open Images V6** dataset. Following the synthetic burst generation protocol of KPN,
