[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-XGBoost%20%7C%20TensorFlow-orange)](https://xgboost.readthedocs.io/)

This repository contains the official simulation engine, dataset generation tools, and benchmark scripts for the paper **"Exposing Slow-Fading QKD Attacks via Physics-Motivated Statistical Features"**.

We introduce the **Slow-Fading Stealth Attack (SFSA)**, a sophisticated threat where an eavesdropper (Eve) synchronizes her intercept strategy with the natural thermal drift of a quantum channel. We demonstrate that while deep learning (LSTM) struggles with the long-range temporal dependencies of this attack, a physics-informed **XGBoost** classifier leveraging higher-order statistical moments (Kurtosis, Fano Factor) achieves superior detection with a fraction of the hardware cost.

## 📄 Abstract

Quantum Key Distribution (QKD) theoretically guarantees security based on the laws of physics, but practical implementations remain vulnerable to "stealth" attacks that hide within the variance of environmental noise. Standard countermeasures typically rely on simple Quantum Bit Error Rate (QBER) thresholds, which sophisticated adversaries can evade by mimicking natural channel fading. 

In this work, we introduce the Slow-Fading Stealth Attack (SFSA), a physically realizable threat model where an adversary modulates intercepts to synchronize with thermal drift. Using a calibrated dataset of $10^7$ pulses, we demonstrate that standard Likelihood-Ratio Tests (LRT) fail to detect this attack (AUC 0.64) due to its statistical camouflage. However, we show that **physics-informed feature engineering**—specifically leveraging the higher-order moments (kurtosis, variance) of the photon count distribution—enables a static gradient boosting model (XGBoost, AUC 0.85) to significantly outperform deep sequence learning approaches (LSTM, AUC 0.76). Feature analysis reveals that the attack effectively "cools" the photon statistics into a **sub-Poissonian regime ($F < 1$)**, creating a detectable artifact.

## 🚀 Repository Structure

```text
SFSA-QKD-Attack/
│
├── README.md               <-- You are here
├── requirements.txt        <-- Dependencies
├── LICENSE                 <-- MIT License
│
├── src/                    <-- The Simulation Engine
│   ├── __init__.py
│   ├── simulation.py       <-- Eve's Logic, Channel Drift, and ProductionInjector classes
│   └── feature_engine.py   <-- Sliding window logic, Fano Factor, & Entropy calcs
│
├── notebooks/              <-- Experiments & Reproducibility
│   ├── 01_Physics_Validation.ipynb  <-- Lag Sweep Experiment (Fig. 2 Generation)
│   ├── 02_Data_Generation.ipynb     <-- Runs the 10M pulse Monte Carlo simulation
│   └── 03_Benchmark_Results.ipynb   <-- Trains XGBoost/LSTM & plots ROC/Feature Imp (Fig. 3 & 4)
│
├── data/                   <-- Dataset Storage (Ignored by Git)
│   ├── production_run/     <-- Raw pulse data (CSV)
│   └── windows/            <-- Feature vectors (CSV)
│
└── results/                <-- Paper Artifacts
    ├── figures/            <-- Generated PNGs for the paper
    └── models/             <-- Saved XGBoost JSON / LSTM H5 models
```

🛠️ InstallationClone the repository:Bashgit clone [https://github.com/your-username/sfsa-qkd-attack.git](https://github.com/your-username/sfsa-qkd-attack.git)
cd sfsa-qkd-attack
Create a virtual environment (Recommended):Bashpython -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:Bashpip install -r requirements.txt
🧪 Usage WorkflowTo reproduce the paper's results, run the notebooks in the following order:01_Physics_Validation.ipynb:Simulates the "Lag Sweep" to find the "Gray Zone" where Eve is correlated ($\rho \approx 0.7$) but undetectable by standard thresholds.Output: results/figures/fig2_lag_sweep.png02_Data_Generation.ipynb:Generates $10^7$ pulses using the ProductionInjector class.Splits data into 'Normal' (Thermal Drift) and 'Attack' (SFSA) chunks.Output: Raw CSV files in data/production_run/.03_Benchmark_Results.ipynb:Extracts statistical features (Mean, Variance, Skewness, Kurtosis).Trains XGBoost (Proposed) and LSTM (Baseline).Generates ROC Curves and Feature Importance plots.Output: fig3_roc_curves.png, fig4_feature_importance.png.📊 Key ResultsModelAUCLatency (FPGA Est.)Hardware ResourceXGBoost (Ours)0.85~20 nsLUTs (Abundant)LSTM (Baseline)0.76> 1 $\mu$sDSP Slices (Scarce)LRT (Standard)0.64N/AN/A🔗 CitationIf you use this code or dataset, please cite our paper:Code snippet@inproceedings{sfsa2025,
  title={Exposing Slow-Fading QKD Attacks via Physics-Motivated Statistical Features},
  author={Anonymous Author(s)},
  booktitle={Proceedings of the IEEE QPAIN Conference},
  year={2025}
}
📜 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.