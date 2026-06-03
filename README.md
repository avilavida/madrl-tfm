# Multi-Agent Deep Reinforcement Learning (MADRL): Algorithmic Comparison

## Overview

This project evaluates and compares multiple Multi-Agent Deep Reinforcement Learning (MADRL) algorithms across different environments, including both cooperative and competitive settings.

The main objective is to analyze the performance, stability, scalability, and computational efficiency of value-based and policy-based MARL methods.

In addition, the project incorporates sustainability analysis by tracking carbon emissions during training using CodeCarbon.

---

## Key Contributions

- Systematic comparison of value-based and policy-based MARL algorithms
- Evaluation across cooperative (MPE) and competitive (SMAC) environments
- Analysis of training stability, scalability, and sample efficiency
- Integration of carbon footprint tracking for sustainable AI assessment

---

## Algorithms

The following Multi-Agent Reinforcement Learning algorithms are implemented and evaluated:

- IPPO (Independent Proximal Policy Optimization)
- MAPPO (Multi-Agent PPO)
- QMIX (Value Decomposition Networks)
- VDN (Value Decomposition Networks)

---

## Environments

Experiments are conducted in the following environments:

### MPE (Multi-Agent Particle Environment)
- simple_adversary
- simple_push
- simple_spread
- simple_reference
- simple_tag

### SMAC (StarCraft Multi-Agent Challenge)
- 3m, 8m, 13m scenarios
- Cooperative combat tasks with configurable difficulty levels

---

## Project Structure

The project is organized as follows:
```
src/
├── runner.py # Training pipeline
├── evaluate_seed.py # Evaluation of trained models
├── analyze.py # Results analysis and visualization
├── launcher.py # Experiment execution (grid search)
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/avilavida/madrl-tfm.git
cd madrl-tfm
```

### 2. Create environment
```bash
conda env create -f environment.yml
conda activate marllib
```

### 3. External dependencies
This project depends on the following external frameworks:
- MARLlib: https://marllib.readthedocs.io/en/latest/handbook/installation.html
- SMAC: https://github.com/oxwhirl/smac

---

## Results

The framework generates:

- Learning curves (reward vs timesteps)
- Final performance comparison
- Boxplots and barplots
- Carbon emissions analysis (CodeCarbon)

---

## External Libraries
- MARLlib: https://github.com/Replicable-MARL/MARLlib
- SMAC: https://github.com/oxwhirl/smac
- Ray: https://docs.ray.io/
- PyTorch: https://pytorch.org/
- CodeCarbon: https://github.com/mlco2/codecarbon

---

## Author

Albert Avila Vidal  
Master’s Thesis
Master’s in Data Science  
Universitat Oberta de Catalunya (UOC)
