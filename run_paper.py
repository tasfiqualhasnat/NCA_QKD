# src/simulation.py
import numpy as np
import pandas as pd
import os

# Default parameters from your notebook's "HARD PARAMETERS"
DEFAULT_PARAMS = {
    "drift_period": 50000,
    "lag": 5000,
    "intercept_limit": 0.10,
    "mu": 0.006,
    "process_noise": 1e-5,
    "measure_noise": 1e-2,
    "base_qber": 0.015,
    "seed": 42
}

class ProductionInjector:
    """
    The Core Physics Engine: Simulates QKD channel drift and Eve's 
    'Slow-Fading' intercept strategy.
    
    Exact logic from: sfsa-validation-ipynb (4).ipynb
    """
    def __init__(self, drift_period=50000, lag=5000, output_dir="data/production_run"):
        self.drift_period = drift_period
        self.lag = lag
        self.output_dir = output_dir
        
        # Load constraints from default params
        self.base_qber = DEFAULT_PARAMS["base_qber"]
        self.intercept_limit = DEFAULT_PARAMS["intercept_limit"]
        self.process_noise = DEFAULT_PARAMS["process_noise"]
        self.measure_noise = DEFAULT_PARAMS["measure_noise"]
        self.mu = DEFAULT_PARAMS["mu"]
        
        # Internal state
        self.est_drift = 0.0
        self.est_var = 1.0

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_chunk(self, n_samples=100000, chunk_id=0, is_attack=False):
        # Time array based on chunk_id to keep sine wave continuous
        # Assuming sequential generation
        time_offset = chunk_id * n_samples
        t = np.arange(n_samples) + time_offset
        
        # ---------------------------------------------------------
        # A. Simulate Nature (Exact Match)
        # ---------------------------------------------------------
        drift = 0.005 * np.sin(2 * np.pi * t / self.drift_period)
        walk = np.cumsum(np.random.normal(0, 0.0001, size=n_samples))
        
        # Note: Your notebook scales walk by 0.01
        true_noise = self.base_qber + drift + (walk * 0.01)
        true_noise = np.clip(true_noise, 0.005, 0.05)
        
        # ---------------------------------------------------------
        # B. Simulate Eve (SFSA) (Exact Match)
        # ---------------------------------------------------------
        intercept_frac = np.zeros(n_samples)
        
        if is_attack:
            noise_series = pd.Series(true_noise)
            
            # 1. Vectorized Lag: shift, fill
            # Logic: lagged_noise = noise_series.shift(self.lag).fillna(method='bfill')
            lagged_noise = noise_series.shift(self.lag).bfill().values
            
            # 2. Measure with noise
            measured = lagged_noise + np.random.normal(0, 0.002, size=n_samples)
            
            # 3. Eve's Estimator: EWM (Exponential Weighted Moving Average)
            # Logic: eve_est = pd.Series(measured).ewm(alpha=0.1).mean()
            eve_est = pd.Series(measured).ewm(alpha=0.1).mean().values
            
            # 4. Normalize and calculate Intercept Fraction
            norm_drift = (eve_est - 0.005) / (0.05 - 0.005)
            norm_drift = np.clip(norm_drift, 0, 1)
            intercept_frac = self.intercept_limit * norm_drift
            
            self.est_drift = eve_est[-1]

        # ---------------------------------------------------------
        # C. Physics Telemetry (Exact Match)
        # ---------------------------------------------------------
        # Logic: counts = np.random.poisson(PARAMS["mu"], size=n_samples)
        counts = np.random.poisson(self.mu, size=n_samples)
        
        # Logic: counts = np.random.binomial(counts, 1 - intercept_frac)
        counts = np.random.binomial(counts, 1 - intercept_frac)
        
        # Logic: qber = true_noise + (intercept_frac * 0.25)
        qber = true_noise + (intercept_frac * 0.25)
        qber += np.random.normal(0, 0.002, size=n_samples)
        qber = np.clip(qber, 0, 1)

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': t,
            'counts': counts.astype(np.int8),
            'qber': qber.astype(np.float32),
            'bg_noise': true_noise.astype(np.float32),
            'eve_intensity': intercept_frac.astype(np.float32),
            'attack_label': 1 if is_attack else 0
        })

        # Save Logic
        label = "ATTACK" if is_attack else "NORMAL"
        fname = f"chunk_{chunk_id:03d}_{label}.csv"
        fpath = os.path.join(self.output_dir, fname)
        df.to_csv(fpath, index=False)
        return fpath

# Alias for backward compatibility if needed
SlowFadingAttack = ProductionInjector