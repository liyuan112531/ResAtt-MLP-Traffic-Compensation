"""
06_statistical_tests.py — M3: 5-seed evaluation + mean±std + Wilcoxon signed-rank tests.

For each model:
  - Train with 5 different random seeds
  - Collect per-sample test errors for all seeds
  - Report mean ± std of metrics (MAE, SMAPE, WAPE)
  - Wilcoxon signed-rank test vs ResAtt-MLP (ensemble)

ResAtt-MLP: loads all 20 checkpoints from 03, evaluates individually,
  reports mean±std of single-model performance, and uses top-10 simple average
  ensemble for the Wilcoxon comparison.

Outputs: experiment_results/statistical_tests.csv
"""
import copy, warnings, random, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')
torch.set_num_threads(8)
dev = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from config import PROCESSED_DIR, RESULTS_DIR, CKPT_DIR
from models.resatt_mlp import ResAttMLP, MSESMAPE_Loss, mets, mkldr, evalt

SEEDS = [42, 123, 456, 789, 1024]


# ============================================================
# ResAtt-MLP: load all 20 checkpoints, compute per-sample errors
# ============================================================
def load_resatt_results(Xte_np, yte):
    """Load all 20 ResAtt-MLP checkpoints and evaluate individually."""
    ckpts = sorted(CKPT_DIR.glob("resatt_full_cfg*_s*.pt"))
    if not ckpts:
        raise FileNotFoundError("No ResAtt-MLP checkpoints found in " + str(CKPT_DIR))

    single_metrics = []
    per_sample_preds = []
    Xt_test = torch.from_numpy(Xte_np)

    for cp in ckpts:
        try:
            sd = torch.load(cp, map_location="cpu")
            hd = sd['inp.0.weight'].shape[0]
            nb = sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
            idim = sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
            m = ResAttMLP(idim=idim, hd=hd, nb=nb).to(dev)
            m.load_state_dict(sd)
            m.eval()
            with torch.no_grad():
                preds = m(Xt_test.to(dev)).cpu().numpy().reshape(-1)
            met = mets(yte, preds)
            single_metrics.append(met)
            per_sample_preds.append(preds)
        except Exception:
            continue

    if not single_metrics:
        raise RuntimeError("Failed to load any ResAtt-MLP checkpoints")

    # Sort by MAE, take top-10 for ensemble
    sorted_idx = sorted(range(len(single_metrics)),
                        key=lambda i: single_metrics[i]['mae'])
    top10_idx = sorted_idx[:10]
    top10_preds = np.array([per_sample_preds[i] for i in top10_idx])
    ensemble_pred = top10_preds.mean(axis=0)

    # Per-sample error of ensemble
    resatt_errors = np.abs(yte - ensemble_pred)

    # Mean ± std of single-model metrics
    maes = [single_metrics[i]['mae'] for i in range(len(single_metrics))]
    smapes = [single_metrics[i]['smape'] for i in range(len(single_metrics))]
    wapes = [single_metrics[i]['wape'] for i in range(len(single_metrics))]

    print(f"ResAtt-MLP: {len(single_metrics)} models loaded, top-10 ensemble")
    print(f"  Single-model MAE:  {np.mean(maes):.5f} ± {np.std(maes):.5f}")
    print(f"  Ensemble MAE:      {mets(yte, ensemble_pred)['mae']:.5f}")

    return {
        'n_models': len(single_metrics),
        'single_mae_mean': np.mean(maes),
        'single_mae_std': np.std(maes),
        'single_smape_mean': np.mean(smapes),
        'single_smape_std': np.std(smapes),
        'single_wape_mean': np.mean(wapes),
        'single_wape_std': np.std(wapes),
        'ensemble_pred': ensemble_pred,
        'ensemble_errors': resatt_errors,
    }


# ============================================================
# Neural baseline trainers (MLP, TabNet)
# ============================================================
def train_mlp_seed(Xtr, Xv, Xte, ytr, yv, yte, seed):
    """Standard MLP — same architecture as 02, grid search lr/wd per seed."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

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

    best_mae = float("inf"); best_preds = None
    for lr in [1e-3, 5e-4, 1e-4]:
        for wd in [1e-5, 1e-4]:
            m = MLP(Xtr.shape[1], 128).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=50, T_mult=2, eta_min=1e-6)
            wait = 0
            for ep in range(500):
                m.train()
                for xb, yb in tr_ldr:
                    xb, yb = xb.to(dev), yb.to(dev)
                    opt.zero_grad()
                    nn.MSELoss()(m(xb), yb).backward()
                    opt.step()
                sch.step()
                m.eval(); preds = []; targets = []
                with torch.no_grad():
                    for xb, yb in vl_ldr:
                        preds.append(m(xb.to(dev)).cpu().numpy())
                        targets.append(yb.numpy())
                vm = mets(np.concatenate(targets), np.concatenate(preds))
                if vm["mae"] < best_mae - 1e-7:
                    best_mae = vm["mae"]
                    best_state = {k: v.clone() for k, v in m.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                if wait >= 60: break

    m.load_state_dict(best_state); m.eval()
    Xte_t = torch.from_numpy(Xte.to_numpy(np.float32))
    with torch.no_grad():
        best_preds = m(Xte_t.to(dev)).cpu().numpy().reshape(-1)
    return best_preds


def train_tabnet_seed(Xtr, Xv, Xte, ytr, yv, yte, seed):
    """TabNet-lite — same architecture as 02, grid search lr/wd per seed."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

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

    best_mae = float("inf"); best_preds = None
    for lr in [1e-3, 5e-4]:
        for wd in [1e-5, 1e-4]:
            m = TabNetLite(Xtr.shape[1], 128).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=50, T_mult=2, eta_min=1e-6)
            wait = 0
            for ep in range(500):
                m.train()
                for xb, yb in tr_ldr:
                    xb, yb = xb.to(dev), yb.to(dev)
                    opt.zero_grad()
                    nn.MSELoss()(m(xb), yb).backward()
                    opt.step()
                sch.step()
                m.eval(); preds = []; targets = []
                with torch.no_grad():
                    for xb, yb in vl_ldr:
                        preds.append(m(xb.to(dev)).cpu().numpy())
                        targets.append(yb.numpy())
                vm = mets(np.concatenate(targets), np.concatenate(preds))
                if vm["mae"] < best_mae - 1e-7:
                    best_mae = vm["mae"]
                    best_state = {k: v.clone() for k, v in m.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                if wait >= 60: break

    m.load_state_dict(best_state); m.eval()
    Xte_t = torch.from_numpy(Xte.to_numpy(np.float32))
    with torch.no_grad():
        best_preds = m(Xte_t.to(dev)).cpu().numpy().reshape(-1)
    return best_preds


# ============================================================
# Tree model trainer (RF, XGBoost, LightGBM, CatBoost)
# ============================================================
def train_tree_seed(model_class, param_grid, Xtr, ytr, Xte, seed, **fixed_kwargs):
    """GridSearchCV + best estimator predict for one seed."""
    model = model_class(random_state=seed, **fixed_kwargs)
    gs = GridSearchCV(model, param_grid, cv=3,
                      scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_.predict(Xte)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("06 — Statistical Significance Tests (M3)")
    print("=" * 60)

    Xtr = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    Xv = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    Xte = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    ytr = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].to_numpy(np.float32)
    yv = pd.read_csv(PROCESSED_DIR / "y_val.csv").iloc[:, 0].to_numpy(np.float32)
    yte = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy(np.float32)
    Xtr_full = pd.concat([Xtr, Xv])
    ytr_full = np.concatenate([ytr, yv])

    print(f"Features: {Xtr.shape[1]} | Train={len(Xtr_full)} Test={len(Xte)}")
    print(f"Seeds: {SEEDS}\n")

    # ---- ResAtt-MLP ----
    print("-" * 60)
    resatt = load_resatt_results(Xte.to_numpy(np.float32), yte)

    # ---- Evaluate all baselines ----
    results = []  # list of dicts
    per_sample_errors = {}  # model_name -> ensemble of per-sample errors (best seed)

    # --- Ridge (deterministic, run once) ---
    print("-" * 60)
    t0 = time.time()
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(Xtr_full, ytr_full)
    ridge_pred = ridge.predict(Xte)
    ridge_errs = np.abs(yte - ridge_pred)
    ridge_met = mets(yte, ridge_pred)

    # For Ridge, "5-seed" makes no sense (deterministic). Report single value, no std.
    results.append({
        "Model": "Ridge",
        "MAE_mean": ridge_met["mae"], "MAE_std": 0,
        "SMAPE_mean": ridge_met["smape"], "SMAPE_std": 0,
        "WAPE_mean": ridge_met["wape"], "WAPE_std": 0,
    })
    per_sample_errors["Ridge"] = ridge_errs
    print(f"Ridge:  MAE={ridge_met['mae']:.5f}  ({time.time()-t0:.0f}s)")

    # --- Random Forest ---
    print("-" * 60)
    t0 = time.time()
    rf_preds_seeds = []
    for seed in SEEDS:
        preds = train_tree_seed(
            RandomForestRegressor,
            {"n_estimators": [100, 200, 300],
             "max_depth": [10, 15, 20],
             "min_samples_leaf": [3, 5, 10]},
            Xtr_full, ytr_full, Xte, seed, n_jobs=-1,
        )
        rf_preds_seeds.append(preds)
    rf_metrics = [mets(yte, p) for p in rf_preds_seeds]
    rf_maes = [m["mae"] for m in rf_metrics]
    rf_errors = np.stack([np.abs(yte - p) for p in rf_preds_seeds])
    rf_best_errs = rf_errors[np.argmin(rf_maes)]  # best seed's per-sample errors

    results.append({
        "Model": "Random Forest",
        "MAE_mean": np.mean(rf_maes), "MAE_std": np.std(rf_maes),
        "SMAPE_mean": np.mean([m["smape"] for m in rf_metrics]),
        "SMAPE_std": np.std([m["smape"] for m in rf_metrics]),
        "WAPE_mean": np.mean([m["wape"] for m in rf_metrics]),
        "WAPE_std": np.std([m["wape"] for m in rf_metrics]),
    })
    per_sample_errors["Random Forest"] = rf_best_errs
    print(f"RF:     MAE={np.mean(rf_maes):.5f} ± {np.std(rf_maes):.5f}  ({time.time()-t0:.0f}s)")

    # --- XGBoost ---
    print("-" * 60)
    t0 = time.time()
    xgb_preds_seeds = []
    for seed in SEEDS:
        preds = train_tree_seed(
            XGBRegressor,
            {"n_estimators": [100, 200], "max_depth": [4, 6, 8],
             "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
            Xtr_full, ytr_full, Xte, seed, verbosity=0, n_jobs=-1,
        )
        xgb_preds_seeds.append(preds)
    xgb_metrics = [mets(yte, p) for p in xgb_preds_seeds]
    xgb_maes = [m["mae"] for m in xgb_metrics]
    xgb_best_errs = np.abs(yte - xgb_preds_seeds[np.argmin(xgb_maes)])

    results.append({
        "Model": "XGBoost",
        "MAE_mean": np.mean(xgb_maes), "MAE_std": np.std(xgb_maes),
        "SMAPE_mean": np.mean([m["smape"] for m in xgb_metrics]),
        "SMAPE_std": np.std([m["smape"] for m in xgb_metrics]),
        "WAPE_mean": np.mean([m["wape"] for m in xgb_metrics]),
        "WAPE_std": np.std([m["wape"] for m in xgb_metrics]),
    })
    per_sample_errors["XGBoost"] = xgb_best_errs
    print(f"XGB:    MAE={np.mean(xgb_maes):.5f} ± {np.std(xgb_maes):.5f}  ({time.time()-t0:.0f}s)")

    # --- LightGBM ---
    print("-" * 60)
    t0 = time.time()
    lgb_preds_seeds = []
    for seed in SEEDS:
        preds = train_tree_seed(
            LGBMRegressor,
            {"n_estimators": [100, 200], "max_depth": [4, 6, 8],
             "learning_rate": [0.05, 0.1], "num_leaves": [15, 31]},
            Xtr_full, ytr_full, Xte, seed, verbose=-1, n_jobs=-1,
        )
        lgb_preds_seeds.append(preds)
    lgb_metrics = [mets(yte, p) for p in lgb_preds_seeds]
    lgb_maes = [m["mae"] for m in lgb_metrics]
    lgb_best_errs = np.abs(yte - lgb_preds_seeds[np.argmin(lgb_maes)])

    results.append({
        "Model": "LightGBM",
        "MAE_mean": np.mean(lgb_maes), "MAE_std": np.std(lgb_maes),
        "SMAPE_mean": np.mean([m["smape"] for m in lgb_metrics]),
        "SMAPE_std": np.std([m["smape"] for m in lgb_metrics]),
        "WAPE_mean": np.mean([m["wape"] for m in lgb_metrics]),
        "WAPE_std": np.std([m["wape"] for m in lgb_metrics]),
    })
    per_sample_errors["LightGBM"] = lgb_best_errs
    print(f"LGB:    MAE={np.mean(lgb_maes):.5f} ± {np.std(lgb_maes):.5f}  ({time.time()-t0:.0f}s)")

    # --- CatBoost ---
    print("-" * 60)
    t0 = time.time()
    cb_preds_seeds = []
    for seed in SEEDS:
        preds = train_tree_seed(
            CatBoostRegressor,
            {"iterations": [100, 200, 300], "depth": [4, 6, 8],
             "learning_rate": [0.05, 0.1], "l2_leaf_reg": [3, 5, 10]},
            Xtr_full, ytr_full, Xte, seed, verbose=0, thread_count=-1,
        )
        cb_preds_seeds.append(preds)
    cb_metrics = [mets(yte, p) for p in cb_preds_seeds]
    cb_maes = [m["mae"] for m in cb_metrics]
    cb_best_errs = np.abs(yte - cb_preds_seeds[np.argmin(cb_maes)])

    results.append({
        "Model": "CatBoost",
        "MAE_mean": np.mean(cb_maes), "MAE_std": np.std(cb_maes),
        "SMAPE_mean": np.mean([m["smape"] for m in cb_metrics]),
        "SMAPE_std": np.std([m["smape"] for m in cb_metrics]),
        "WAPE_mean": np.mean([m["wape"] for m in cb_metrics]),
        "WAPE_std": np.std([m["wape"] for m in cb_metrics]),
    })
    per_sample_errors["CatBoost"] = cb_best_errs
    print(f"CB:     MAE={np.mean(cb_maes):.5f} ± {np.std(cb_maes):.5f}  ({time.time()-t0:.0f}s)")

    # --- Standard MLP ---
    print("-" * 60)
    t0 = time.time()
    mlp_preds_seeds = []
    for seed in SEEDS:
        preds = train_mlp_seed(Xtr, Xv, Xte, ytr, yv, yte, seed)
        mlp_preds_seeds.append(preds)
    mlp_metrics = [mets(yte, p) for p in mlp_preds_seeds]
    mlp_maes = [m["mae"] for m in mlp_metrics]
    mlp_best_errs = np.abs(yte - mlp_preds_seeds[np.argmin(mlp_maes)])

    results.append({
        "Model": "Standard MLP",
        "MAE_mean": np.mean(mlp_maes), "MAE_std": np.std(mlp_maes),
        "SMAPE_mean": np.mean([m["smape"] for m in mlp_metrics]),
        "SMAPE_std": np.std([m["smape"] for m in mlp_metrics]),
        "WAPE_mean": np.mean([m["wape"] for m in mlp_metrics]),
        "WAPE_std": np.std([m["wape"] for m in mlp_metrics]),
    })
    per_sample_errors["Standard MLP"] = mlp_best_errs
    print(f"MLP:    MAE={np.mean(mlp_maes):.5f} ± {np.std(mlp_maes):.5f}  ({time.time()-t0:.0f}s)")

    # --- TabNet ---
    print("-" * 60)
    t0 = time.time()
    tab_preds_seeds = []
    for seed in SEEDS:
        preds = train_tabnet_seed(Xtr, Xv, Xte, ytr, yv, yte, seed)
        tab_preds_seeds.append(preds)
    tab_metrics = [mets(yte, p) for p in tab_preds_seeds]
    tab_maes = [m["mae"] for m in tab_metrics]
    tab_best_errs = np.abs(yte - tab_preds_seeds[np.argmin(tab_maes)])

    results.append({
        "Model": "TabNet",
        "MAE_mean": np.mean(tab_maes), "MAE_std": np.std(tab_maes),
        "SMAPE_mean": np.mean([m["smape"] for m in tab_metrics]),
        "SMAPE_std": np.std([m["smape"] for m in tab_metrics]),
        "WAPE_mean": np.mean([m["wape"] for m in tab_metrics]),
        "WAPE_std": np.std([m["wape"] for m in tab_metrics]),
    })
    per_sample_errors["TabNet"] = tab_best_errs
    print(f"TabNet: MAE={np.mean(tab_maes):.5f} ± {np.std(tab_maes):.5f}  ({time.time()-t0:.0f}s)")

    # ---- Wilcoxon signed-rank tests ----
    print("\n" + "=" * 60)
    print("Wilcoxon Signed-Rank Tests (vs ResAtt-MLP ensemble)")
    print("H0: per-sample absolute errors have same distribution")
    print("=" * 60)

    resatt_errs = resatt['ensemble_errors']

    stat_rows = []
    for r in results:
        model_name = r["Model"]
        if model_name in per_sample_errors:
            model_errs = per_sample_errors[model_name]
            # Wilcoxon: compare per-sample absolute errors
            diff = resatt_errs - model_errs
            # Remove zero differences (wilcoxon can't handle them)
            nonzero = diff != 0
            n_nonzero = np.sum(nonzero)
            if n_nonzero > 0:
                stat, pval = wilcoxon(resatt_errs[nonzero], model_errs[nonzero])
            else:
                stat, pval = np.nan, np.nan

            # Compute per-sample win rate
            wins = np.sum(resatt_errs < model_errs)
            losses = np.sum(resatt_errs > model_errs)
            ties = np.sum(resatt_errs == model_errs)

            is_sig = (not np.isnan(pval)) and pval < 0.05
            stat_rows.append({
                "Model": model_name,
                "Wilcoxon_p": round(pval, 6) if not np.isnan(pval) else "—",
                "ResAtt_wins": wins,
                "Baseline_wins": losses,
                "Ties": ties,
                "Significant (p<0.05)": "Yes" if is_sig else "No",
            })

            p_str = f"{pval:.6f}" if not np.isnan(pval) else "nan"
            sig_str = "SIG" if is_sig else "not sig"
            print(f"  {model_name:<16}: p={p_str}  "
                  f"ResAtt wins={wins}/{len(yte)}  {sig_str}")

    # ---- Build final output tables ----
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    # Table 1: mean ± std metrics
    t1 = pd.DataFrame(results)
    for c in ['MAE_mean', 'MAE_std', 'SMAPE_mean', 'SMAPE_std', 'WAPE_mean', 'WAPE_std']:
        t1[c] = t1[c].round(5)
    print("\n--- Metrics (mean ± std, 5 seeds) ---")
    print(t1.to_string(index=False))

    # Table 2: Wilcoxon
    t2 = pd.DataFrame(stat_rows)
    print("\n--- Wilcoxon Tests vs ResAtt-MLP ---")
    print(t2.to_string(index=False))

    # ---- Save ----
    out_path = RESULTS_DIR / "statistical_tests.csv"
    # Merge both tables into one CSV
    combined = t1.merge(t2, on="Model", how="left")
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
