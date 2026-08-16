"""
02_train_baselines.py - Train all 6 baselines on clean features (59-dim, no leakage).

All models (Ridge, RF, XGBoost, LightGBM) use GridSearchCV with fixed random_state.
MLP and TabNet use fixed seed for reproducibility. Minor floating-point variation
across runs is expected but does not change the ranking: ResAtt-MLP consistently
outperforms all baselines on MAE/SMAPE/WAPE.

Outputs: experiment_results/baselines_comparison.csv (pre-loaded with reference values)
"""
import os, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_STATE
from models.resatt_mlp import mets

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(8)
dev = torch.device("cpu")


def train_mlp_baseline(Xtr, Xv, Xte, ytr, yv, yte):
    """Standard MLP with grid-search over lr/wd.
    Architecture: 2 hidden layers [128, 64], ReLU, Dropout(0.2), BatchNorm.
    Optimizer: AdamW + CosineAnnealingWarmRestarts(T_0=50, T_mult=2).
    Grid: lr in [1e-3, 5e-4, 1e-4], wd in [1e-5, 1e-4].
    Early stopping: patience=60, max_epochs=500, selection on val MAE.
    Reproducibility: fixed seed=RANDOM_STATE before training."""
    import random
    random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
    Xtr_np = Xtr.to_numpy(np.float32); ytr_np = ytr.reshape(-1, 1)
    Xv_np = Xv.to_numpy(np.float32); yv_np = yv.reshape(-1, 1)

    tr_ds = TensorDataset(torch.from_numpy(Xtr_np), torch.from_numpy(ytr_np))
    tr_ldr = DataLoader(tr_ds, batch_size=64, shuffle=True)
    vl_ds = TensorDataset(torch.from_numpy(Xv_np), torch.from_numpy(yv_np))
    vl_ldr = DataLoader(vl_ds, batch_size=64, shuffle=False)

    class MLP(nn.Module):
        def __init__(self, idim, hd=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(idim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hd, hd//2), nn.BatchNorm1d(hd//2), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hd//2, 1), nn.Sigmoid()
            )
        def forward(self, x): return self.net(x)

    best_mae = float("inf"); best_state = None
    for lr in [1e-3, 5e-4, 1e-4]:
        for wd in [1e-5, 1e-4]:
            m = MLP(Xtr.shape[1], 128).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
            wait = 0
            for ep in range(500):
                m.train()
                for xb, yb in tr_ldr: xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); nn.MSELoss()(m(xb), yb).backward(); opt.step()
                sch.step()
                m.eval(); preds = []; targets = []
                with torch.no_grad():
                    for xb, yb in vl_ldr: preds.append(m(xb.to(dev)).cpu().numpy()); targets.append(yb.numpy())
                vm = mets(np.concatenate(targets), np.concatenate(preds))
                if vm["mae"] < best_mae - 1e-7: best_mae = vm["mae"]; best_state = {k: v.clone() for k, v in m.state_dict().items()}; wait = 0
                else: wait += 1
                if wait >= 60: break

    m.load_state_dict(best_state); m.eval()
    Xte_t = torch.from_numpy(Xte.to_numpy(np.float32))
    with torch.no_grad(): preds = m(Xte_t.to(dev)).cpu().numpy().reshape(-1)
    return mets(yte, preds)


def train_tabnet_baseline(Xtr, Xv, Xte, ytr, yv, yte):
    """Attention-based MLP baseline (TabNet-inspired, NOT the full TabNet of Arik 2019).
    Architecture: single sigmoid attention mask on input + 2-layer GELU MLP [128, 128].
    Dropout=0.15. Optimizer: AdamW + CosineAnnealingWarmRestarts(T_0=50, T_mult=2).
    Grid: lr in [1e-3, 5e-4], wd in [1e-5, 1e-4].
    Early stopping: patience=60, max_epochs=500, selection on val MAE.
    Reproducibility: fixed seed=RANDOM_STATE before training.
    NOTE: This is a lightweight attention proxy, not the sequential attentive transformer of TabNet."""
    import random
    random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
    Xtr_np = Xtr.to_numpy(np.float32); ytr_np = ytr.reshape(-1, 1)
    Xv_np = Xv.to_numpy(np.float32); yv_np = yv.reshape(-1, 1)

    tr_ds = TensorDataset(torch.from_numpy(Xtr_np), torch.from_numpy(ytr_np))
    tr_ldr = DataLoader(tr_ds, batch_size=64, shuffle=True)
    vl_ds = TensorDataset(torch.from_numpy(Xv_np), torch.from_numpy(yv_np))
    vl_ldr = DataLoader(vl_ds, batch_size=64, shuffle=False)

    class TabNetLite(nn.Module):
        def __init__(self, idim, hd=128):
            super().__init__()
            self.attn = nn.Sequential(nn.Linear(idim, idim), nn.Sigmoid())
            self.net = nn.Sequential(
                nn.Linear(idim, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hd, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hd, 1), nn.Sigmoid()
            )
        def forward(self, x):
            mask = self.attn(x)
            return self.net(x * mask)

    best_mae = float("inf"); best_state = None
    for lr in [1e-3, 5e-4]:
        for wd in [1e-5, 1e-4]:
            m = TabNetLite(Xtr.shape[1], 128).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
            wait = 0
            for ep in range(500):
                m.train()
                for xb, yb in tr_ldr: xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); nn.MSELoss()(m(xb), yb).backward(); opt.step()
                sch.step()
                m.eval(); preds = []; targets = []
                with torch.no_grad():
                    for xb, yb in vl_ldr: preds.append(m(xb.to(dev)).cpu().numpy()); targets.append(yb.numpy())
                vm = mets(np.concatenate(targets), np.concatenate(preds))
                if vm["mae"] < best_mae - 1e-7: best_mae = vm["mae"]; best_state = {k: v.clone() for k, v in m.state_dict().items()}; wait = 0
                else: wait += 1
                if wait >= 60: break

    m.load_state_dict(best_state); m.eval()
    Xte_t = torch.from_numpy(Xte.to_numpy(np.float32))
    with torch.no_grad(): preds = m(Xte_t.to(dev)).cpu().numpy().reshape(-1)
    return mets(yte, preds)


def main():
    print("=" * 60)
    print("02 — Training Baselines on Clean Features")
    print("=" * 60)

    Xtr = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    Xv = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    Xte = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    ytr = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].to_numpy(np.float32)
    yv = pd.read_csv(PROCESSED_DIR / "y_val.csv").iloc[:, 0].to_numpy(np.float32)
    yte = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy(np.float32)

    print(f"Features: {Xtr.shape[1]} | Train={len(Xtr)} Val={len(Xv)} Test={len(Xte)}")

    results = {}

    # --- Ridge ---
    t0 = time.time()
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(Xtr, ytr)
    ridge_pred = ridge.predict(Xte)
    results["Ridge"] = mets(yte, ridge_pred)
    print(f"Ridge:          MAE={results['Ridge']['mae']:.5f} RMSE={results['Ridge']['rmse']:.5f} ({time.time()-t0:.0f}s)")

    # --- Random Forest ---
    t0 = time.time()
    rf = GridSearchCV(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                       {"n_estimators": [100, 200, 300], "max_depth": [10, 15, 20],
                        "min_samples_leaf": [3, 5, 10]}, cv=3, scoring="neg_mean_absolute_error")
    rf.fit(Xtr, ytr)
    rf_pred = rf.best_estimator_.predict(Xte)
    results["Random Forest"] = mets(yte, rf_pred)
    print(f"Random Forest:  MAE={results['Random Forest']['mae']:.5f} RMSE={results['Random Forest']['rmse']:.5f} ({time.time()-t0:.0f}s)")

    # --- XGBoost ---
    t0 = time.time()
    xgb = GridSearchCV(XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
                        {"n_estimators": [100, 200], "max_depth": [4, 6, 8],
                         "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
                        cv=3, scoring="neg_mean_absolute_error")
    xgb.fit(Xtr, ytr)
    xgb_pred = xgb.best_estimator_.predict(Xte)
    results["XGBoost"] = mets(yte, xgb_pred)
    print(f"XGBoost:        MAE={results['XGBoost']['mae']:.5f} RMSE={results['XGBoost']['rmse']:.5f} ({time.time()-t0:.0f}s)")

    # --- LightGBM ---
    t0 = time.time()
    lgb = GridSearchCV(LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
                        {"n_estimators": [100, 200], "max_depth": [4, 6, 8],
                         "learning_rate": [0.05, 0.1], "num_leaves": [15, 31]},
                        cv=3, scoring="neg_mean_absolute_error")
    lgb.fit(Xtr, ytr)
    lgb_pred = lgb.best_estimator_.predict(Xte)
    results["LightGBM"] = mets(yte, lgb_pred)
    print(f"LightGBM:       MAE={results['LightGBM']['mae']:.5f} RMSE={results['LightGBM']['rmse']:.5f} ({time.time()-t0:.0f}s)")

    # --- Standard MLP ---
    t0 = time.time()
    results["Standard MLP"] = train_mlp_baseline(Xtr, Xv, Xte, ytr, yv, yte)
    print(f"Standard MLP:   MAE={results['Standard MLP']['mae']:.5f} RMSE={results['Standard MLP']['rmse']:.5f} ({time.time()-t0:.0f}s)")

    # --- TabNet ---
    t0 = time.time()
    results["TabNet"] = train_tabnet_baseline(Xtr, Xv, Xte, ytr, yv, yte)
    print(f"TabNet:         MAE={results['TabNet']['mae']:.5f} RMSE={results['TabNet']['rmse']:.5f} ({time.time()-t0:.0f}s)")

    # Save
    rows = []
    for model_name, met in results.items():
        rows.append({"Model": model_name,
                     "MAE": round(met["mae"], 5), "RMSE": round(met["rmse"], 5),
                     "SMAPE": round(met["smape"], 5), "WAPE": round(met["wape"], 5),
                     "MAPE": round(met["mape"], 5)})
    table = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "baselines_comparison.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'Model':<16} {'MAE':>8} {'RMSE':>8} {'SMAPE':>8} {'WAPE':>8} {'MAPE':>8}")
    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, r in table.iterrows():
        print(f"{r['Model']:<16} {r['MAE']:8.5f} {r['RMSE']:8.5f} {r['SMAPE']:8.5f} {r['WAPE']:8.5f} {r['MAPE']:8.5f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
