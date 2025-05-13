# Introduction

This repository contains a student project created by me and two colleagues as part of the *Natural Language Processing* course at the Faculty of Mathematics, University of Warsaw (2024). Our work investigates two recent advancements in Transformer architectures:

- **Lory** – a fully differentiable Mixture-of-Experts mechanism (Zhong et al., 2023)
- **Multi-Head Mixture-of-Experts** – a technique applying MoE independently across attention heads (Wu et al., 2023)

We combined these approaches into a novel architecture, **Multi-Head Lory**, and evaluated its performance on a language modeling task. A detailed explanation of our methods and experiments can be found in [`NLP_paper.pdf`](./NLP_paper.pdf).

### My contributions:
- Implemented all required Transformer variants in PyTorch:
  - Standard Mixture of Experts (baseline)
  - Multi-Head Mixture of Experts
  - Lory
  - Multi-Head Lory  
  All models are modular and configurable via a single `config` object passed to the `Transformer` class.
- Built the full training and validation pipeline using PyTorch Lightning, including:
  - GPU training support  
  - Model checkpointing  
  - Training loss logging
- Conducted model training and hyperparameter search on available GPU resources
- Co-authored the final report

## 🛠️ Technologies Used

- Python
- PyTorch
- PyTorch Lightning

Below is the original README of the repository, describing the code structure and usage in more detail.

---

# Overview

Welcome to the repository! Here, you will find implementations of three different versions of the Mixture of Experts (MoE) layer in a Transformer model.

## Code Structure

### Building Blocks for Transformer

The `model_classes.py` file includes essential components of the Transformer model, such as:

- **Attention Mechanism**
- **Rotary Positional Encoding**
- **Layer Normalization (LayerNorm)**
- **Transformer Block**
- **Router**
- **Vectorized MoE**

The `dataloader.py` file contains the implementation of our custom dataloader. For training our models, we used the "20220301.simple" subset from the Wikipedia dataset. You can download this dataset by following the instructions in the `dataloader` module.


### Standard MoE Layer

For the implementation of a conventional Mixture of Experts layer, refer to the `VectorizedMoE()` class located in the `model_classes.py` file.

### Multi-Head MoE

For the vectorized implementation of the Multi-Head MoE (MH-MoE) layer, please refer to the `MH-MoE.py` file.

### Lory

For the implementation of a fully differentiable MoE layer named Lory, navigate to the `MH_Lory.py` file. Set the number of heads to one to obtain the traditional Lory module.

### MH-Lory

In the `MH_Lory.py` file, increasing the number of heads will configure the Multi-Head Lory model, a novel architecture we propose.

### More Information

For more detailed information and mathematical formulations, please refer to the report section and the `main.tex` file where we describe our project in detail.
