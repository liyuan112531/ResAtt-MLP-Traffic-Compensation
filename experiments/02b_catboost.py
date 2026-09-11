"""
02b_catboost.py — Add CatBoost baseline on clean features.

Matches the exact same data split and GridSearchCV protocol as 02_train_baselines.py.
Outputs: experiment_results/catboost_result.csv
"""
import os, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_STATE
from models.resatt_mlp import mets


def main():
    print("=" * 60)
    print("02b — CatBoost Baseline on Clean Features")
    print("=" * 60)

    Xtr = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    Xte = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    ytr = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].to_numpy(np.float32)
    yte = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy(np.float32)

    print(f"Features: {Xtr.shape[1]} | Train={len(Xtr)} Test={len(Xte)}")

    # ---- CatBoost with GridSearchCV ----
    t0 = time.time()
    cb = GridSearchCV(
        CatBoostRegressor(random_seed=RANDOM_STATE, verbose=0, thread_count=-1),
        {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "l2_leaf_reg": [3, 5, 10],
        },
        cv=3, scoring="neg_mean_absolute_error",
    )
    cb.fit(Xtr, ytr)
    cb_pred = cb.best_estimator_.predict(Xte)
    met = mets(yte, cb_pred)
    elapsed = time.time() - t0

    print(f"CatBoost:       MAE={met['mae']:.5f} RMSE={met['rmse']:.5f} SMAPE={met['smape']:.5f} WAPE={met['wape']:.5f} MAPE={met['mape']:.5f} ({elapsed:.0f}s)")
    print(f"  Best params: {cb.best_params_}")

    # ---- Save ----
    row = pd.DataFrame([{
        "Model": "CatBoost",
        "MAE": round(met["mae"], 5),
        "RMSE": round(met["rmse"], 5),
        "SMAPE": round(met["smape"], 5),
        "WAPE": round(met["wape"], 5),
        "MAPE": round(met["mape"], 5),
    }])
    out_path = RESULTS_DIR / "catboost_result.csv"
    row.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
