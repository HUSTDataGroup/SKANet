# SKANet: Skip-Enhanced Kolmogorov-Arnold Network for Robust Cardiac Segmentation

> **🚧 News:** The code and pretrained weights will be released coming soon.

## 👥 Authors & Affiliations

**Song Li\*, Shuaichao Du\*, Xiaofeng Li**

*\* Equal contribution*

**College of Computer Science and Technology, Harbin University of Science and Technology**

---

## 💻 Training Environment & Experiment Details

### GPU Configuration
All experiments were conducted using the following setup:
* **GPU Model:** NVIDIA RTX 3090
* **CUDA:** 11.6
* **PyTorch:** 1.12.1

### Experiment Settings
* **Dataset:** ACDC (Automatic Cardiac Diagnosis Challenge)
* **Training Time:** 400 epochs
* **Optimizer:** AdamW
* **Learning Rate:** $3 \times 10^{-4}$
* **Weight Decay:** $1 \times 10^{-5}$
* **Loss Function:** Combined loss function including Entity Region Consistency Constraint Loss ($L_E$)

---

## 📦 Get Started

### Data Preparation
The ACDC dataset is used for training. Please organize your data structure as follows:
