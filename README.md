# SKANet: Skip-Enhanced Kolmogorov-Arnold Network for Robust Cardiac Segmentation

> 🚧 **The code will be released coming soon.**

This repository holds the official implementation of the paper: **SKANet: Skip-Enhanced Kolmogorov-Arnold Network for Robust Cardiac Segmentation**.

## 🚀 Getting Started

### Installation

1. Create a virtual environment conda create -n SKANet python=3.11 and activate it conda activate SKANet
2. Install Pythorch
3. git clone https://github.com/HUSTDataGroup/SKANet
4. Enter the SKANet folder cd SKANet and run pip install -r requirements.txt

### Training
```bash
python train.py --dataset ACDC --vit_name R50-ViT-B_16
```
### Testing
```bash
python test.py --dataset ACDC --vit_name R50-ViT-B_16
```
## 🙋‍♀️ Feedback and Contact
Shuaichao Du: [2420410171@stu.hrbust.edu.cn](2420410171@stu.hrbust.edu.cn)

## 🛡️ License
This project is under the Apache-2.0 license. See LICENSE for details.

## 🙏 Acknowledgement
We would like to thank the research group of Prof. Song Li at the College of Computer Science and Technology, Harbin University of Science and Technology, Prof. Xiaofeng Li, and Prof. AAA for their support of this research.

## 📝 Citation
If our work is helpful in your research, please cite this as follows.

```bash
@article{Li2025SKANet,
    title={SKANet: Skip-Enhanced Kolmogorov-Arnold Network for Robust Cardiac Segmentation},
    author={Song Li and Shuaichao Du and Xiaofeng Li and AAAA},
    journal={Under Review},
    year={2025}
}
```
