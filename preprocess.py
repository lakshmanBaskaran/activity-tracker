# File: data_preprocess.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib

# Constants
WINDOW_SIZE = 200   # e.g., ~4 seconds at 50 Hz
STEP        = 100   # 50% overlap

# Paths to your raw CSVs
ACC_CSV = 'data/raw/Phones_accelerometer.csv'
GYR_CSV = 'data/raw/Phones_gyroscope.csv'

def load_sensor(csv_path, axes):
    """Load one sensor CSV and keep only timestamp, axes, and 'gt' label."""
    df = pd.read_csv(csv_path)
    cols = ['Arrival_Time'] + axes + ['gt']
    return df[cols]

def merge_sensors(df_acc, df_gyr):
    """Merge accelerometer + gyroscope data on timestamp and label."""
    return pd.merge(
        df_acc, df_gyr,
        on=['Arrival_Time', 'gt'],
        suffixes=('_acc', '_gyr')
    )

def window_data(df):
    """Slice into overlapping windows and one-hot encode the activity."""
    df = df.sort_values('Arrival_Time').reset_index(drop=True)

    # Encode the 'gt' activity labels
    le = LabelEncoder()
    df['act_enc'] = le.fit_transform(df['gt'])
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/activity_encoder.joblib')

    # Dynamically discover sensor columns
    acc_cols = [c for c in df.columns if c.endswith('_acc')]
    gyr_cols = [c for c in df.columns if c.endswith('_gyr')]
    seq_cols = acc_cols + gyr_cols
    if not seq_cols:
        raise ValueError(f"No *_acc or *_gyr columns found. Available columns: {df.columns.tolist()}")

    # Build the sensor matrix
    seq = df[seq_cols].values   # shape: (num_samples, num_channels)
    labels = df['act_enc'].values

    X, y = [], []
    n = len(df)
    for start in range(0, n - WINDOW_SIZE + 1, STEP):
        end = start + WINDOW_SIZE
        X.append(seq[start:end])
        # choose the majority-vote label in this window
        lab = np.bincount(labels[start:end]).argmax()
        y.append(lab)

    X = np.stack(X)  # shape: (n_windows, WINDOW_SIZE, num_channels)
    y = to_categorical(y, num_classes=len(le.classes_))

    os.makedirs('data/processed', exist_ok=True)
    np.savez('data/processed/simple_seq.npz', X=X, y=y)
    print(f"Prepared {len(X)} windows → data/processed/simple_seq.npz")

if __name__ == '__main__':
    # Load and merge the two sensor streams
    df_acc = load_sensor(ACC_CSV, axes=['x','y','z'])
    df_gyr = load_sensor(GYR_CSV, axes=['x','y','z'])
    df = merge_sensors(df_acc, df_gyr)
    window_data(df)
