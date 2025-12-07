# SKANet: Skip-Enhanced Kolmogorov-Arnold Network for Robust Cardiac Segmentation

> **🚧 News:** The code and pretrained weights will be released coming soon.

## 👥 Authors & Affiliations

**Song Li\*, Shuaichao Du\*, Xiaofeng Li**

*\* Equal contribution*

**College of Computer Science and Technology, Harbin University of Science and Technology**

---

## 📖 Introduction

This repository contains the official implementation of **SKANet**, a novel deep learning framework designed for robust multi-modal cardiac image segmentation.

Cardiac image segmentation faces significant challenges due to complex anatomical variability and the limitations of traditional CNNs and MLPs in capturing high-frequency non-linear features. **SKANet** addresses these issues by integrating the mathematical interpretability of the **Kolmogorov-Arnold Theorem** with modern deep learning architectures.

Our approach introduces a skip-enhanced encoder-decoder structure that effectively balances pixel-level boundary precision with macroscopic anatomical consistency.

## 🚀 Method Overview

SKANet introduces several key architectural innovations:

### 1. Dynamic Group Rational Kolmogorov-Arnold Network (DGRKAN)
Instead of static activation functions, DGRKAN utilizes learnable **rational basis functions** to perform deep non-linear transformations.
* **Grouped Parameter Sharing:** Reduces parameter explosion while maintaining computational efficiency.
* **Adaptive Dropout:** Dynamically balances model expressiveness and generalization capabilities during training.
* **Decoupled Design:** Separates non-linear activation from linear transformation ($Q$ matrix) for flexible feature fitting.

### 2. Multi-scale Kolmogorov-Arnold Attention Module (MSKAM)
A comprehensive attention mechanism designed to aggregate multi-scale context:
* **Parallel Attention Extraction (PAE):** Simultaneously processes Channel Attention (CAM) and Spatial Attention (SAM) to preserve information fidelity and avoid sequential bias.
* **Dual-Attention Fusion Mechanism (DAFM):** Efficiently integrates attention streams using lightweight convolutions.
* **Multi-scale KAN Processor (MKAP):** An inverted residual block that embeds DGRKAN to refine features across different scales.

### 3. Entity Region Consistency Constraint Loss ($L_E$)
To ensure anatomical plausibility, we introduce a novel loss function ($L_E$) that enforces topological consistency, significantly reducing outliers and improving segmentation reliability.

## 📝 Citation

If you find this work helpful for your research, please consider citing our paper:

```bibtex
@article{SKANet2025,
  title={SKANet: Skip-Enhanced Kolmogorov-Arnold Network for Robust Cardiac Segmentation},
  author={Li, Song and Du, Shuaichao and Li, Xiaofeng},
  journal={Under Review},
  year={2025}
}
