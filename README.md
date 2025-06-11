# Spiking Neural Networks (SNNs) for Learning Tasks

This reserach work contains Python implementations exploring two distinct approaches to Spiking Neural Networks (SNNs) for different learning paradigms. It delves into both fundamental SNN principles and their application in advanced machine learning challenges like few-shot learning.

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
  - [1. Custom SNN with Reward-Modulated STDP (NumPy)](#1-custom-snn-with-reward-modulated-stdp-numpy)
  - [2. SNN-based Prototypical Networks for Few-Shot Learning (snnTorch)](#2-snn-based-prototypical-networks-for-few-shot-learning-snntorch)
- [Results](#results)
- [License](#license)

---

## Introduction

Spiking Neural Networks (SNNs) are biologically inspired neural networks that process information through discrete events called _"spikes."_ Unlike traditional Artificial Neural Networks (ANNs), SNNs incorporate the dimension of time, offering potential advantages in energy efficiency and the natural processing of spatio-temporal data.

This project provides two SNN implementations:

- A **custom-built SNN using NumPy** to illustrate the foundational concepts of neuron dynamics (Leaky Integrate-and-Fire) and a biologically plausible learning rule: **Reward-modulated Spike-Timing Dependent Plasticity (R-STDP)**.
- An **advanced SNN-based Prototypical Network built with snnTorch** for few-shot image classification on the MNIST dataset.

---

## Features

### 1. Custom SNN with Reward-Modulated STDP (NumPy)

- **Leaky Integrate-and-Fire (LIF) Neuron Model**: Simulates neuron membrane potential.
- **Reward-Modulated STDP (R-STDP)**: Updates synaptic weights based on spike timing and a reward signal.
- **Basic SNN Architecture**: Demonstrates spike propagation and plasticity.
- **Manual Simulation**: Step-by-step simulation of SNN dynamics in Python using NumPy.

### 2. SNN-based Prototypical Networks for Few-Shot Learning (snnTorch)

- **Spiking Convolutional Feature Extractor** using snnTorch.
- **Rate Coding**: Converts image pixel intensities into spike trains.
- **Surrogate Gradient Learning**: Enables end-to-end training via backpropagation.
- **Prototypical Network Framework**: Performs few-shot classification using learned SNN features.
- **Episodic Meta-Training**: Teaches the model to rapidly adapt to new tasks.
- **MNIST Integration**: Trains and tests on few-shot learning tasks derived from MNIST.

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Installation

Clone the repository:

```bash
git clone https://github.com/dreamboat26/Exploration-of-Spiking-Neural-Networks-for-Learning-Tasks.git
cd Exploration-of-Spiking-Neural-Networks-for-Learning-Tasks
```
Install the required dependencies:
```bash
pip install torch torchvision numpy snntorch
```

### How to Run
Run the main script:
```bash
python Spiking_Neural_Network.py
```
This will:
- Download the MNIST dataset (if not already present).
- Train the snnTorch-based Prototypical Network on few-shot tasks.
- Run the NumPy-based SNN simulations.
- Print outputs and training progress to the console.

## Project Structure
All code is within a single Python file, organized into:
- Configuration: Global constants for both SNN implementations.
- NumPy SNN Implementation: LIF neuron, R-STDP, simulation.
- MNISTFewShotDataset: Custom dataset for episodic few-shot learning.
- SNNFeatureExtractor: Convolutional SNN model with snnTorch.
- Prototypical Network Logic: Prototypes, distance, loss, and accuracy.
- Training & Evaluation: train_meta_model() and test_meta_model().
- Main Block: Orchestrates execution.

## Implementation Details
1. Custom SNN with Reward-Modulated STDP (NumPy)
- Neuron Model: LIF neuron with spike generation and refractory handling.
- R-STDP Learning Rule: Updates weights using pre-/post-synaptic spike timing and global reward signal.
- Basic Architecture: Multi-layer SNN with manual weight updates.
- Simulation: Step-by-step execution with membrane dynamics and plasticity.

2. SNN-based Prototypical Networks for Few-Shot Learning (snnTorch)
- Feature Extractor: Convolutional layers with spiking neurons.
- Input Encoding: Rate coding based on pixel intensity over time steps.
- Surrogate Gradient: snnTorch.surrogate.fast_sigmoid enables gradient-based learning.
- Prototypes & Inference: Class prototypes from support set, query classification via Euclidean distance.
- Loss & Optimization: Cross-entropy on negative distances with Adam optimizer.
- Meta-Training: Learns to embed features for fast adaptation across tasks.

## Results
- Meta-Training: Shows decreasing loss and improving accuracy on few-shot tasks.
- Meta-Testing: Evaluates generalization to unseen classes.
- NumPy SNN: Demonstrates neuron firing dynamics and R-STDP learning behavior.

## 📄 License

This project is licensed under the MIT License.
