# Structure-Preserving Kolmogorov-Arnold Network with Anatomical Priors for Cardiac Segmentationn

Cardiac Magnetic Resonance imaging provides comprehensive morphological and functional information, making it the gold standard for cardiovascular diagnosis. However, the clinical deployment of deep learning models is hindered by challenges in handling data heterogeneity, balancing context aggregation with interpretability, and incorporating anatomical priors. To address these challenges, we propose the Structure-Aware Kolmogorov-Arnold Network, named SKANet. Specifically, a Dynamic Group Rational Kolmogorov-Arnold Network is integrated into skip connections to adaptively refine features against domain shifts. Furthermore, a Multi-scale Kolmogorov-Arnold Attention Module is devised to capture fine-grained contexts while improving model transparency. Additionally, we design an Entity Region Consistency Constraint Loss to explicitly incorporate anatomical priors for semantically consistent predictions. Extensive experiments are conducted on three diverse benchmark datasets (ACDC, MSCMRSeg, and M&Ms). On the ACDC dataset, SKANet achieves state-of-the-art performance, yielding a Dice score of 92.65% and an HD95 of 1.21 mm. Furthermore, zero-shot evaluations on the M&Ms dataset explicitly validate the framework's robust cross-domain generalization, while experiments on the MSCMRSeg dataset demonstrate its exceptional multi-sequence robustness and annotation-efficient capabilities. This paper indicates that SKANet provides a robust foundation for accurate cardiac segmentation across diverse pathologies and imaging protocols, which is crucial for standardized and reliable cardiovascular diagnosis. 

```
## 🙋‍♀️ Feedback and Contact
Shuaichao Du: [2420410171@stu.hrbust.edu.cn](2420410171@stu.hrbust.edu.cn)

## 🛡️ License
This project is under the Apache-2.0 license. See LICENSE for details.

