# File: train.py

import os
import numpy as np
import pandas as pd
from scipy.fftpack import rfft
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

from lightgbm import LGBMClassifier  # LightGBM classifier

def load_data(path='data/processed/simple_seq.npz'):
    """Load windowed sequences and one-hot labels from disk."""
    arr = np.load(path)
    return arr['X'], arr['y']

def extract_features(X):
    """
    From raw windows X of shape (n_windows, T, C),
    compute per-channel features:
      - time-domain: mean, std, median, iqr, zero-crossing rate
      - freq-domain: mean & std of the rfft magnitudes
    Returns a pandas DataFrame of shape (n_windows, C*7) with named columns.
    """
    n, T, C = X.shape
    feats = []
    col_names = []
    for c in range(C):
        col_names += [
            f"ch{c}_mean", f"ch{c}_std", f"ch{c}_med", f"ch{c}_iqr", f"ch{c}_zcr",
            f"ch{c}_freq_mean", f"ch{c}_freq_std"
        ]

    for i in range(n):
        stats = []
        for c in range(C):
            x = X[i, :, c]
            # time-domain
            stats.append(x.mean())
            stats.append(x.std())
            stats.append(np.median(x))
            stats.append(np.percentile(x, 75) - np.percentile(x, 25))
            stats.append(np.sum(np.abs(np.diff(np.sign(x)))) / 2)
            # freq-domain
            mags = np.abs(rfft(x))
            stats.append(mags.mean())
            stats.append(mags.std())
        feats.append(stats)

    return pd.DataFrame(feats, columns=col_names)

def train():
    os.makedirs('models', exist_ok=True)

    # 1) Load & featurize
    X, y = load_data()
    X_df = extract_features(X)
    y_labels = y.argmax(axis=1)

    # 2) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df, y_labels,
        test_size=0.2,
        stratify=y_labels,
        random_state=42
    )

    # 3) Scale
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(
        scaler.fit_transform(X_tr),
        columns=X_tr.columns, index=X_tr.index
    )
    X_te_s = pd.DataFrame(
        scaler.transform(X_te),
        columns=X_te.columns, index=X_te.index
    )
    joblib.dump(scaler, 'models/simple_scaler.joblib')

    # 4) LightGBM hyperparam search, with extra weight on class 6 (“nan”)
    #    to compensate its under-representation
    class_weight = {cls: (2 if cls == 6 else 1) for cls in np.unique(y_tr)}
    param_dist = {
        'n_estimators':  [100, 200, 400, 600],
        'num_leaves':    [31, 63, 127],
        'max_depth':     [-1, 10, 20, 30],
        'learning_rate': [0.1, 0.05, 0.01],
    }
    lgb = LGBMClassifier(
        random_state=42, n_jobs=-1, class_weight=class_weight, verbose=-1
    )
    search = RandomizedSearchCV(
        estimator=lgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_tr_s, y_tr)
    best_model = search.best_estimator_
    print("Best hyperparameters:", search.best_params_)

    joblib.dump(best_model, 'models/simple_lgbm_model.joblib')

    # 5) Evaluate with a confidence threshold for “nan”
    #    Only label nan if P(nan)>0.6, otherwise pick next best
    probs = best_model.predict_proba(X_te_s)
    raw_pred = np.argmax(probs, axis=1)
    thresh = 0.6
    adjusted_pred = []
    for i, p in enumerate(probs):
        if raw_pred[i] == 6 and p[6] < thresh:
            # zero out p[nan] and re-argmax
            p2 = p.copy()
            p2[6] = 0.0
            adjusted_pred.append(np.argmax(p2))
        else:
            adjusted_pred.append(raw_pred[i])
    adjusted_pred = np.array(adjusted_pred)

    print("\nClassification Report (with nan-threshold):")
    print(classification_report(y_te, adjusted_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, adjusted_pred))

    # 6) Label mapping
    le = joblib.load('models/activity_encoder.joblib')
    print("\nLabel mapping (index → activity):")
    for i, cls in enumerate(le.classes_):
        print(f"  {i}: {cls}")

if __name__ == '__main__':
    train()
