# src/simulation.py

"""
The Core Physics Engine: Simulates QKD channel drift and Eve's 
'Noise_Camouflaged' intercept strategy. Here we test how Eve
can estimate the channel drift with a Kalman filter and use that
to decide how much signal to intercept without being detected.
1. Simulate natural channel drift as a combination of sinusoidal
        variation and random walk.
2. Eve estimates the drift using a Kalman filter with
        measurement lag.
3. Based on Eve's estimate, she decides how much signal to
        intercept, up to a defined limit.
4. The final QBER is computed considering both natural drift
        and Eve's intercept-induced errors.
Parameters:
- drift_period_samples: Period of the sinusoidal drift component.
- base_qber: Baseline QBER without drift or attack.
- intercept_limit: Maximum fraction of signal Eve can intercept.
- lag_samples: Measurement lag in samples for Eve's estimation.
Returns:
- DataFrame with columns:
    'timestamp': Sample timestamps.
    'counts': Simulated photon counts after Eve's intercept.
    'qber': Final QBER including drift and attack effects.
    'true_drift': The actual channel drift values.
    'eve_est': Eve's estimated drift values.
    'eve_intensity': Fraction of signal intercepted by Eve.
    'is_attack': Binary flag indicating presence of attack.
"""


import numpy as np
import pandas as pd

class NoiseCamouflagedAttackSimulator:
    def __init__(self, drift_period=50000, base_qber=0.015, intercept_limit=0.10, lag=5000, mu=0.006):
        # Parameters used by the snippet
        self.drift_period = drift_period
        self.base_qber = base_qber
        self.intercept_limit = intercept_limit
        self.lag = lag  # Changed from lag_samples to lag to match your snippet
        self.mu = mu

    def generate_attacked_chunk(self, n_samples, time_offset=0, is_attack=False):
        t = np.arange(n_samples) + time_offset

        # A. Channel Physics: Sine Drift + Random Walk
        drift = 0.005 * np.sin(2 * np.pi * t / self.drift_period)
        walk = np.cumsum(np.random.normal(0, 0.0001, size=n_samples))
        true_noise = self.base_qber + drift + (walk * 0.01)
        true_noise = np.clip(true_noise, 0.005, 0.05)

        intercept_fraction = np.zeros(n_samples) 
        eve_estimates = np.zeros(n_samples)

        if is_attack:
            noise_series = pd.Series(true_noise)
            # Apply lag
            lagged_noise = noise_series.shift(self.lag).fillna(method='bfill').values
            
            # Eve's tracking estimate (EWMA)
            measured = lagged_noise + np.random.normal(0, 0.001, size=n_samples)
            eve_estimates = pd.Series(measured).ewm(alpha=0.1).mean().values

            # CALIBRATED SCALING: Ensures correlation is ~0.70 at Lag 5000
            norm_drift = (eve_estimates - 0.01) / 0.012 
            norm_drift = np.clip(norm_drift, 0, 1)
            intercept_fraction = self.intercept_limit * norm_drift

        # B. Quantum Statistics (Poisson -> Binomial thinning)
        counts = np.random.poisson(self.mu, size=n_samples)
        counts = np.random.binomial(counts, 1 - intercept_fraction)

        # Bob's Observed QBER
        qber = (1 - intercept_fraction) * true_noise + (intercept_fraction * 0.25)
        qber += np.random.normal(0, 0.001, size=n_samples)
        qber = np.clip(qber, 0, 1)

        return pd.DataFrame({
            'timestamp': t,
            'counts': counts,
            'qber': qber.astype(np.float32),
            'bg_noise': true_noise.astype(np.float32),
            'eve_intensity': intercept_fraction.astype(np.float32),
            'is_attack': np.full(n_samples, 1 if is_attack else 0, dtype=np.int8)
        })