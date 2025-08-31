# A-Tale-of-Two-Criticalities
Full Code Implementation for MRes Thesis, Submitted September 2025

This provides the full code implementation for the model and results prepared for "A Tale of Two Criticalities: How the Brain Learns" MRes Thesis. Below is a map of how everything is structured. 

Structural and Temporal Criticality codes are seperated into two folders, based on the parameters provided in the paper. Below is a brief list of what each file contains

### Strucutral Criticality: 
#### Baseline Tc:
Create Model + Compute Tc:
- Model building for MNIST classification
- Defining functions for Tc calculation
- Baseline Tc calculation

Formal Plotting + Power law: 
- Graph shown as in Paper
- Plotting in log-log graph

#### Noise & Forgetting: 
Noise and Forgetting Tc & Eigenvals:
  - Plots the Tc and full eigenvalue spectrum for baseline, noise, and forgetting
  - Showing that Tc doesn't calculate unless noise is added midway through

Min&Max Eigenvals:
  - Minimum and Maximum eigenvalues for baseline, noise, and forgetting
  - Comparison for noise and forgetting given at the bottom

Loss Curves:
  - Plots the validation and training loss curves for baseline, noise, and forgetting

### Temporal Criticality: 
Model Creation + d2 for all: 
  - Model building for MNIST classification
  - Defining functions for d2 calculation
  - Defining functions for noise and forgetting from beginning
  - Baseline, noise, forgetting d2 calculation
  - Population Spiking Raster Plots for baseline, noise, forgetting
