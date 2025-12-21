# Data Preparation

This document outlines the datasets used in our study. To comprehensively evaluate the performance, robustness, and generalization capability of the proposed method, we selected three representative MICCAI cardiac MRI benchmark datasets: ACDC, MSCMRSeg, and M&Ms. All experiments strictly adhere to official subject-level splitting schemes, ensuring complete patient independence across training, validation, and testing sets.

The core segmentation task involves three key anatomical structures: the **Right Ventricle (RV)**, **Myocardium (Myo)**, and **Left Ventricle (LV)**. Evaluations are conducted at both the End-Diastole (ED) and End-Systole (ES) phases.

## Datasets

### 1. Automated Cardiac Diagnosis Challenge (ACDC)
Derived from MICCAI 2017, this dataset serves as the gold standard for assessing algorithm performance under diverse cardiac pathological conditions. It comprises Cine MRI scans categorized into five clinical subgroups: **Normal (NOR)**, **Myocardial Infarction (MINF)**, **Dilated Cardiomyopathy (DCM)**, **Hypertrophic Cardiomyopathy (HCM)**, and **Abnormal Right Ventricle (ARV)**. Strictly following the official patient-level partition, we utilize **70 cases for training, 10 for validation, and 20 for testing**.

### 2. Multi-Sequence Cardiac MR Segmentation (MSCMRSeg 2019)
Sourced from MICCAI 2019, this dataset provides bSSFP, T2, and LGE sequences from 45 cardiomyopathy patients. The primary challenge lies in the LGE sequences, where the signal intensity of scar tissue closely resembles the blood pool. We utilize the fully annotated subset for supervised training to evaluate the model's intrinsic robustness in parsing high-interference clinical images. The data is partitioned at the subject level, with **25 cases for training, 5 for validation, and 15 for testing**.

### 3. Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation (M&Ms)
Originating from MICCAI 2020, this dataset is an authoritative benchmark for testing domain generalization capabilities, featuring data from six centers and four vendors. In this study, the M&Ms dataset is employed as a **Zero-Shot Cross-Dataset Generalization** benchmark. Models are trained exclusively on ACDC and directly evaluated on the publicly released SOTA benchmark version (**136 cases**) to gauge generalization against unseen domain shifts.

---
**Note on Data Access:**
Due to licensing and privacy restrictions, we do not host or distribute these datasets. All datasets are publicly available through their respective official MICCAI challenge platforms. Users should search for the official dataset names to register and download the data according to the organizers' policies.