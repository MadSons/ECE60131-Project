# Modeling Analog Error in a CVAE for Scene-Aware Multiple Trajectory Prediction

## Project Overview

This repository contains a complete pipeline for trajectory-prediction using both digital and analog in-memory computing (AIMC) simulations. The project enables direct comparison between digital PyTorch inference and analog device-level simulation using CrossSim.

## Important Note on CrossSim

The analog memory simulation done uses cross-sim which does not natively support simulating the GRU. The project was completed using the cross_sim_development repository, a non-public repository which does support the GRU. The Analog evaluations with cross-sim still supports Conv & Linear layers and the results are largely similar.

### Key Features

- **CVAE-based trajectory prediction models**
- **MANTRA memory-augmented networks**
- **CrossSim analog modeling** for SONOS RRAM arrays
- **Dataset processing utilities** for KITTI-derived trajectories
- **Automated multi-model evaluation** with result aggregation
- **Digital vs. Analog comparison** framework

---

## Repository Structure

### Top-Level Folders

```
├── cross-sim/                          # CrossSim stable release submodule
├── cross_sim_development/              # Development/experimental CrossSim branch (Not Accessible to the user)
├── maps/                               # Static environment maps and road masks for the dataset
├── models/                             # PyTorch models
├── pretrained_models/                  # Stored pretrained weights
├── results_multiple_cvae/              # Analog & Digital CVAE evaluation results
├── results_multiple_cvae_cross_sim/    # Analog & Digital CVAE evaluation results (using cross-sim with no AnalogGRU)
├── results_multiple_mantra/            # Analog & Digital MANTRA evaluation results
├── results_multiple_mantra_cross-sim/  # Analog & Digital MANTRA evaluation results (using cross-sim with no AnalogGRU)
├── runs/                               # TensorBoard logs and training metadata
├── test/                               # Evaluation Outputs
├── trainer/                            # Training Scripts for MANTRA
└── training/                           # Model checkpoints during training
```

#### Folder Details

**`cross-sim/`**
- CrossSim stable release submodule
- Provides analog device modeling, crossbar simulation
- Submodule

**`cross_sim_development/`**
- Experimental CrossSim version for testing with AnalogGRU. Not a public repository.

**`maps/`**
- Scene-aware trajectory prediction maps

**`models/`**
- PyTorch model definitions including `cvae_predictor.py`
- MANTRA modules and auxiliary network components

**`pretrained_models/`**
- Best checkpoints from MANTRA training

**`results_multiple_*/`**
- Organized evaluation results containing metrics

**`runs/`**
- TensorBoard logs

**`trainer/`**
- MANTRA training scripts

---

## Python Scripts

### Training Scripts

**`train_cvae.py`**
- Main training loop for CVAE Predictor model
- Implements ELBO (reconstruction + KL divergence)
- Supports β-annealing
- Outputs: `cvae_best.pt` and `cvae_last.pt`

**`train_controllerMem.py`**
-  MANTRA training script for the controller

**`train_IRM.py`**
-  MANTRA training script for the IRM

**`train_ae.py`**
-  MANTRA training script for the autoencoder

### Evaluation Scripts

**`eval_cvae.py`**
- Core evaluation framework for CVAE models
- Produces K-sample trajectory predictions
- Computes ADE/FDE/Euclidean and horizon metrics

**`eval_multiple_cvae.py`**
- Full evaluation pipeline for CVAEs across digital and analog modes
- Runs digital inference once, analog inference multiple times
- Averages device randomness and saves comprehensive results

**`evaluate_MemNet.py`**
- Evaluates MANTRA memory network models on test dataset
- Computes ADE/FDE metrics with top-K trajectory selection

**`evaluate_multiple_MemNet.py`**
- Batch evaluation wrapper for MANTRA
- Supports digital vs. analog comparison
- Outputs aggregated metrics and comparison plots

### Utility Scripts

**`dataset_invariance.py`**
- Fom the MANTRA paper, used for data preprocessing.

**`compare_metrics.py`**
- Comapre CVAE and MANTRA metrics

**`index_qualitative.py`**
- Fom the MANTRA paper, used for data preprocessing.

### Data

**`kitti_dataset.json`**
- Main dataset file with trajectories

---

## Typical Workflow

### 1. Train CVAE Model

```bash
python train_cvae.py
```

### 2. Evaluate Single Model

```bash
python eval_cvae.py
```

### 3. Evaluate Multiple Models (Digital + Analog)

Uses pretrained MANTRA

```bash
python eval_multiple_cvae.py
python evaluate_multiple_MemNet.py
```

### 4. Compare Results

```bash
python compare_metrics.py
```

---

## Environment Setup

### Install Dependencies

I used Python 3.10

```bash
pip install -r requirements.txt
```

You possibly have to change cupy-cuda11x to cupy or another version based on your cuda version as well as the PyTorch download version.
My code supports running on a GPU, can run on CPU by changing a few lines of code.

### Initialize CrossSim Submodule

```bash
git submodule update --init --recursive
```

---

## Analog Simulation Details

Analog inference uses:
- **SONOS device models**
- **Non-idealities:** programming error
- **Crossbar precision constraints:** Rmin/Rmax, Vread

### Sources of Digital-Analog Differences

- Device quantization
- Asymmetric weight programming
- Noise injection
- Crossbar range clipping

### Handling Stochasticity

Scripts like `eval_multiple_cvae.py` explicitly average across multiple analog runs to account for device stochasticity and provide statistically robust comparisons.

---

## Metrics

The evaluation framework computes:
- **ADE** (Average Displacement Error)
- **FDE** (Final Displacement Error)
- **Top-K trajectory selection** metrics

---

## Citation
MANTRA paper used in this work:
- Marchetti, F., Becattini, F., Seidenari, L., & Del Bimbo, A. (2021). MANTRA: Memory augmented networks for multiple trajectory prediction. arXiv. https://arxiv.org/abs/2006.03340
---
