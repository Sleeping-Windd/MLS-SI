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

The base KPN reproduction code is from:

> WenxueCui/KPN_Denoising_Pytorch  
> https://github.com/WenxueCui/KPN_Denoising_Pytorch

That repository does not include a license. Therefore, this repository
does **not** redistribute its source code. To run our method, please first
clone the original repository, then integrate the MLSI modules provided here
following the instructions below.

Our contribution is the MLSI mechanism and its integration code, including:
- `models/MLSI_KPN.py`
- `configs/kpn_mlsi.yaml`
- `train_mlsi.py`
- `evaluate_mlsi.py`

We thank the author of the original repository for making the code publicly available.

Starting from that codebase, we introduce the proposed MLSI mechanism. The main modifications include:

- Adding a separate grayscale burst branch for multi-scale low-frequency structural map construction;
- Computing mean, difference, and normalized similarity maps between the reference frame and each non-reference frame at multiple scales;
- Processing these maps with lightweight convolutions and channel attention;
- Injecting the processed structural features into corresponding encoder stages through residual fusion;
- Retaining the original KPN feature extraction and kernel prediction pipeline.


## Dataset
We evaluate MLSI on the **Open Images V6** dataset. Following the synthetic burst generation protocol of KPN,
