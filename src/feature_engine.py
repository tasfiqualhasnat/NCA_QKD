# src/feature_engine.py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy

def calculate_entropy(series):
    """Calculates Shannon Entropy of the distribution"""
    counts = series.value_counts()
    return entropy(counts)

def compute_autocorr_lag1(series):
    """Calculates Lag-1 Autocorrelation"""
    return series.autocorr(lag=1)

def process_chunk(file_path, chunk_id, window_size=50):
    """
    Loads a raw CSV chunk and applies the Sliding Window transformation.
    Returns the number of windows generated.
    """
    df = pd.read_csv(file_path)
    
    features = []
    # Sliding Window Loop
    for i in range(0, len(df) - window_size, window_size):
        win = df.iloc[i : i + window_size]
        
        # Extract Features from 'counts' (Photon Number)
        c = win['counts'].values
        q = win['qber'].values
        
        feat = {
            'chunk_id': chunk_id,
            # 1. Moments (The Physics)
            'mean_counts': np.mean(c),
            'var_counts': np.var(c),
            'skew_counts': skew(c),
            'kurt_counts': kurtosis(c),
            # 2. QBER Stats
            'mean_qber': np.mean(q),
            'std_qber': np.std(q),
            'max_qber': np.max(q),
            # 3. Advanced
            'entropy_counts': calculate_entropy(win['counts']),
            'autocorr_qber': compute_autocorr_lag1(win['qber']),
            # LABEL (If >50% of window is attack, label as 1)
            'attack_label': 1 if win['is_attack'].mean() > 0.5 else 0
        }
        features.append(feat)
        
    # Save Feature Window CSV
    df_feat = pd.DataFrame(features)
    out_dir = "data/windows"
    import os
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    out_name = f"windowed_chunk_{chunk_id:03d}.csv"
    df_feat.to_csv(os.path.join(out_dir, out_name), index=False)
    
    return len(df_feat)