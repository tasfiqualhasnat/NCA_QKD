# src/feature_engine.py

import numpy as np
import pandas as pd
import scipy.stats
import os

def compute_autocorr_vectorized(matrix):
    """Calculates Lag-1 Autocorrelation for thousands of windows instantly."""
    mu = np.mean(matrix, axis=1, keepdims=True)
    var = np.var(matrix, axis=1, keepdims=True) + 1e-9
    # Shifted matrix multiplication
    x_t = matrix[:, 1:] - mu
    x_tm1 = matrix[:, :-1] - mu
    cov = np.sum(x_t * x_tm1, axis=1) / (matrix.shape[1] - 1)
    return cov / var.flatten()

def process_chunk(file_path, chunk_id, window_size=50):
    df = pd.read_csv(file_path)
    
    # 1. Align data to window size
    n_windows = len(df) // window_size
    limit = n_windows * window_size
    
    # 2. Reshape into 2D Blocks (The Secret to Speed)
    # Shape: (Number of Windows, 50 Pulses)
    counts_2d = df['counts'].values[:limit].reshape(n_windows, window_size)
    qber_2d = df['qber'].values[:limit].reshape(n_windows, window_size)
    
    # Check for label naming consistency
    label_col = 'is_attack' if 'is_attack' in df.columns else 'attack_label'
    labels_2d = df[label_col].values[:limit].reshape(n_windows, window_size)

    # 3. Vectorized Math 
    features = pd.DataFrame({
        'chunk_id': chunk_id,
        # Moments (The Physics Fingerprint)
        'mean_counts': np.mean(counts_2d, axis=1),
        'var_counts': np.var(counts_2d, axis=1),
        'skew_counts': scipy.stats.skew(counts_2d, axis=1),
        'kurt_counts': scipy.stats.kurtosis(counts_2d, axis=1),
        
        # QBER Stats
        'mean_qber': np.mean(qber_2d, axis=1),
        'std_qber': np.std(qber_2d, axis=1),
        'autocorr_qber': compute_autocorr_vectorized(qber_2d),
        
        # Physics Metric: Fano Factor (Var/Mean) - Very strong for NCA detection
        'fano_factor': np.var(counts_2d, axis=1) / (np.mean(counts_2d, axis=1) + 1e-9),
        
        # Final Label (Majority vote in window)
        'attack_label': (np.mean(labels_2d, axis=1) > 0.5).astype(int)
    })

    # 4. Save
    out_dir = "../data/windows"
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"windowed_chunk_{chunk_id:03d}.csv"
    features.to_csv(os.path.join(out_dir, out_name), index=False)
    
    return len(features)