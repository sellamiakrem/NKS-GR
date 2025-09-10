# NKS-GR


This repository provides the official implementation of our paper:

> **Adaptive non-convex sparse regression under Nyström-based kernel graph regularization for semi-supervised feature selection**  
> *Akrem Sellami*  
> [Link to paper ()](link)

---

## Overview
We propose **NKS-GR**, a scalable feature selection framework for **hyperspectral image (HSI) classification**.  
Our method integrates:
- **Spectral and spatial graph regularization**  
- **Sparsity-inducing penalties** ($\ell_1$, $\ell_2$, log-sum, etc.)  
- **Nyström approximation** for efficient pixel-wise graph construction  

The framework is evaluated on benchmark HSI datasets such as *Indian Pines*, *Salinas*, *Botswana*, and *Brain*.  
It achieves **comparable or better accuracy** than full graph construction while drastically reducing **runtime** and **memory usage**.

---

##  Installation
Clone the repository:
```bash
git clone https://github.com/sellamiakrem/NKS-GR.git
cd NKS-GR
```
---

## Project structure
```bash
NKS-GR/
│
├── data/                 # HSI datasets
├── src/
│   ├── models/           # Implementation of NKS-GR, SVM, KNN
│   ├── graphs/           # Full graph & Nyström Laplacian construction
│   ├── utils/            # Helper functions (evaluation, metrics, plots)
│   └── main.py           # Main training & evaluation script
│
├── results/              # Logs, metrics, and saved models
├── requirements.txt
└── README.md
```
For questions,  please send email to akrem.sellami@univ-lille.fr

