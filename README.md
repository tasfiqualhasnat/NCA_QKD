# Variance Clamping: A Physics-Informed Approach for Exposing Noise-Camouflaged Attacks on QKD

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-XGBoost%20%7C%20TensorFlow-orange)](https://xgboost.readthedocs.io/)

This repository contains the **official simulation engine, dataset generation tools, and benchmark scripts** for the paper:

**_“Variance Clamping: A Physics-Informed Approach for Exposing Noise-Camouflaged Attacks on Quantum Key Distribution (QKD)”_**

We introduce the **Noise-Camouflaged Attack (NCA)** — a sophisticated eavesdropping strategy in which an adversary (Eve) synchronizes her interception with the **natural thermal drift** of the quantum channel. We show that while deep learning models (LSTM) struggle with long-range temporal dependencies and high inference latency, a **physics-informed XGBoost classifier** leveraging higher-order statistical moments (Fano Factor, Kurtosis) achieves **superior detection accuracy** with **orders-of-magnitude lower hardware cost**.

---

## 📄 Abstract

Attack detection in Quantum Key Distribution (QKD) systems has traditionally relied on observable increases in Quantum Bit Error Rate (QBER). However, modern adversaries can evade such detection by intercepting only a very small fraction of photons, blending their activity into the natural thermal noise of the fiber.

In this work, we define such threats as **Noise-Camouflaged Attacks (NCA)**. These attacks remain hidden within thermal drift, rendering conventional thresholds and deep learning defenses ineffective. We demonstrate that LSTM-based approaches suffer from high computational latency, poor interpretability, and limited sensitivity to sub-Poissonian statistics.

We propose a **physics-informed detection framework** based on the observation that NCA forces a **variance clamping effect**, driving the photon statistics into a **sub-Poissonian regime (F < 1)** — a condition impossible under natural thermal noise. By explicitly training a lightweight **XGBoost classifier** on Fano Factor artifacts, we achieve superior detection performance (AUC 0.85) compared to LSTM baselines (AUC 0.76), while enabling **wire-speed inference (~20 ns)** suitable for **real-time FPGA deployment**.

---

## 🚀 Repository Structure

```text
NCA_QKD/
│
├── README.md               # Project overview (this file)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
│
├── src/                    # Core simulation engine
│   ├── __init__.py
│   ├── simulation.py       # Eve logic, channel drift, production injector
│   └── feature_engine.py   # Sliding windows, Fano factor, entropy metrics
│
├── notebooks/              # Experiments & reproducibility
│   ├── 01_Physics_Validation.ipynb   # Lag sweep (Fig. 2)
│   ├── 02_Data_Generation.ipynb      # 10M-pulse Monte Carlo simulation
│   └── 03_Benchmark_Results.ipynb    # XGBoost vs LSTM benchmarking
│
├── data/                   # Dataset storage (ignored by Git)
│   ├── production_run/     # Raw pulse data (CSV)
│   └── windows/            # Feature windows (CSV)
│
└── results/                # Paper artifacts
    ├── figures/            # Generated plots
    └── models/             # Saved XGBoost / LSTM models
```

## 🧪 How to Run :

To reproduce the results presented in the paper, run the notebooks in the following order:

#### Step 1: Clone the repository
```
git clone https://github.com/tasfiqualhasnat/NCA_QKD.git
cd NCA_QKD
```
- Ensure the directory structure matches the project tree.
  (The code will auto-create `data/` and `results/` if missing.)

#### Step 2: Create a virtual environment (recommended) : 
```python -m venv venv```

#### Activate it:
- Linux / macOS : ```source venv/bin/activate```

- Windows: ```venv\Scripts\activate```

- Install dependencies: ```pip install -r requirements.txt```

### Step 3: Run  Physics Validation
- ```notebooks/01_Physics_Validation.ipynb```  
   Simulates the lag sweep experiment to identify the gray zone where Eve remains correlated (ρ ≈ 0.7) but undetectable by standard QBER thresholds.  
   **Output:** `results/figures/fig2_lag_sweep.png`

### Step 2: Large-Scale Data Generation
```notebooks/02_Data_Generation.ipynb```  
   - Generates 10^7 photon pulses using the `ProductionInjector` class and splits data into *Normal* (thermal drift) and *Attack* (NCA) regimes.  
   - Applies sliding windows and extracts physics features
   **Output:** Raw CSV files in `data/production_run/` &  windowed features to `data/windows/`

### Step 3: Benchmarking & AI Training
```notebooks/03_Benchmark_Results.ipynb```
   - Extracts statistical features (mean, variance, skewness, kurtosis, Fano factor), trains **XGBoost (proposed)** and **LSTM (baseline)** models, and generates ROC curves and feature-importance plots.  
   **Output:** `fig3_roc_curves.png`, `fig4_feature_importance.png`

---

## 📊 Key Results

| Model             | AUC  | Latency (FPGA Est.) | Hardware Cost       |
|------------------|------|--------------------|------------------|
| XGBoost (Ours)   | 0.85 | ~20 ns             | LUTs (abundant)  |
| LSTM (Baseline)  | 0.76 | > 1 μs             | DSP slices (scarce) |
| LRT (Standard)   | 0.64 | N/A                | N/A              |

---

## 🔗 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{nca2026,
  title={Variance Clamping: A Physics-Informed Approach for Exposing Noise-Camouflaged Attacks on QKD},
  author={Anonymous Author(s)},
  booktitle={Proceedings of the IEEE QPAIN Conference},
  year={2026}
}

