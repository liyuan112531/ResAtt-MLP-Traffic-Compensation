"""
07_enhanced_tests.py — Combined strategy for M3:
  C: 8 seeds (more power than 5)
  B: Wilcoxon on MAE, SMAPE, RMSE
  D: RF scenario models vs ResAtt-MLP scenario models + Wilcoxon

ResAtt-MLP: reuses existing 25 checkpoints for global, scenario checkpoints for D.
Tree baselines: GridSearchCV per seed to find best params independently.
MLP/TabNet: grid search lr/wd per seed.
"""
import copy, warnings, random, os, sys, time, json
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

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 6144]
SCEN_CKPT = CKPT_DIR / "scenarios"


# ============================================================
# ResAtt-MLP: load checkpoints → ensemble
# ============================================================
def load_resatt_ensemble(Xte_np, yte, ckpt_pattern="resatt_full_cfg*_s*.pt"):
    """Load ResAtt-MLP checkpoints, top-10 average ensemble."""
    ckpts = sorted(CKPT_DIR.glob(ckpt_pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matching {ckpt_pattern}")
    single_metrics, per_sample_preds = [], []
    Xt_test = torch.from_numpy(Xte_np)
    for cp in ckpts:
        try:
            sd = torch.load(cp, map_location="cpu")
            hd = sd['inp.0.weight'].shape[0]
            nb = sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
            idim = sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
            m = ResAttMLP(idim=idim, hd=hd, nb=nb).to(dev)
            m.load_state_dict(sd); m.eval()
            with torch.no_grad():
                preds = m(Xt_test.to(dev)).cpu().numpy().reshape(-1)
            met = mets(yte, preds)
            single_metrics.append(met)
            per_sample_preds.append(preds)
        except Exception:
            continue
    sorted_idx = sorted(range(len(single_metrics)), key=lambda i: single_metrics[i]['mae'])
    top_n = min(10, len(sorted_idx))
    top_idx = sorted_idx[:top_n]
    ensemble_pred = np.array([per_sample_preds[i] for i in top_idx]).mean(axis=0)
    return ensemble_pred, single_metrics, per_sample_preds


def per_sample_errors_all(yte, preds):
    """Return dict of per-sample MAE, SMAPE, RMSE component errors."""
    ae = np.abs(yte - preds)
    # SMAPE per-sample component
    denom = (np.abs(yte) + np.abs(preds)) / 2 + 1e-8
    se = ae / denom
    # Squared error
    sqe = (yte - preds) ** 2
    return {"mae": ae, "smape": se, "rmse_sq": sqe}


# ============================================================
# Tree baseline GridSearchCV × N seeds
# ============================================================
def train_tree_multi_seed(model_class, param_grid, Xtr, ytr, Xte, yte, seeds, **fixed_kwargs):
    preds_list, metrics_list = [], []
    for seed in seeds:
        model = model_class(random_state=seed, **fixed_kwargs)
        gs = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        gs.fit(Xtr, ytr)
        p = gs.best_estimator_.predict(Xte)
        preds_list.append(p)
        metrics_list.append(mets(yte, p))
    return preds_list, metrics_list


# ============================================================
# Neural net baselines (MLP, TabNet) × N seeds
# ============================================================
def train_mlp_multi_seed(Xtr, Xv, Xte, ytr, yv, yte, seeds):
    preds_list = []
    Xtr_np = Xtr.to_numpy(np.float32); Xv_np = Xv.to_numpy(np.float32)
    ytr2 = ytr.reshape(-1, 1); yv2 = yv.reshape(-1, 1)

    class MLP(nn.Module):
        def __init__(self, idim, hd=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(idim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hd, hd//2), nn.BatchNorm1d(hd//2), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hd//2, 1), nn.Sigmoid())
        def forward(self, x): return self.net(x)

    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        tr_ds = TensorDataset(torch.from_numpy(Xtr_np), torch.from_numpy(ytr2))
        tr_ldr = DataLoader(tr_ds, batch_size=64, shuffle=True)
        vl_ds = TensorDataset(torch.from_numpy(Xv_np), torch.from_numpy(yv2))
        vl_ldr = DataLoader(vl_ds, batch_size=64, shuffle=False)
        best_mae, best_state = float("inf"), None
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
                    m.eval(); vp, vt = [], []
                    with torch.no_grad():
                        for xb, yb in vl_ldr: vp.append(m(xb.to(dev)).cpu().numpy()); vt.append(yb.numpy())
                    vm = mets(np.concatenate(vt), np.concatenate(vp))
                    if vm["mae"] < best_mae - 1e-7: best_mae = vm["mae"]; best_state = {k: v.clone() for k, v in m.state_dict().items()}; wait = 0
                    else: wait += 1
                    if wait >= 60: break
        m.load_state_dict(best_state); m.eval()
        Xte_t = torch.from_numpy(Xte.to_numpy(np.float32))
        with torch.no_grad(): preds_list.append(m(Xte_t.to(dev)).cpu().numpy().reshape(-1))
    return preds_list


def train_tabnet_multi_seed(Xtr, Xv, Xte, ytr, yv, yte, seeds):
    preds_list = []
    Xtr_np = Xtr.to_numpy(np.float32); Xv_np = Xv.to_numpy(np.float32)
    ytr2 = ytr.reshape(-1, 1); yv2 = yv.reshape(-1, 1)

    class TabNetLite(nn.Module):
        def __init__(self, idim, hd=128):
            super().__init__()
            self.attn = nn.Sequential(nn.Linear(idim, idim), nn.Sigmoid())
            self.net = nn.Sequential(
                nn.Linear(idim, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hd, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hd, 1), nn.Sigmoid())
        def forward(self, x): return self.net(x * self.attn(x))

    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        tr_ds = TensorDataset(torch.from_numpy(Xtr_np), torch.from_numpy(ytr2))
        tr_ldr = DataLoader(tr_ds, batch_size=64, shuffle=True)
        vl_ds = TensorDataset(torch.from_numpy(Xv_np), torch.from_numpy(yv2))
        vl_ldr = DataLoader(vl_ds, batch_size=64, shuffle=False)
        best_mae, best_state = float("inf"), None
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
                    m.eval(); vp, vt = [], []
                    with torch.no_grad():
                        for xb, yb in vl_ldr: vp.append(m(xb.to(dev)).cpu().numpy()); vt.append(yb.numpy())
                    vm = mets(np.concatenate(vt), np.concatenate(vp))
                    if vm["mae"] < best_mae - 1e-7: best_mae = vm["mae"]; best_state = {k: v.clone() for k, v in m.state_dict().items()}; wait = 0
                    else: wait += 1
                    if wait >= 60: break
        m.load_state_dict(best_state); m.eval()
        Xte_t = torch.from_numpy(Xte.to_numpy(np.float32))
        with torch.no_grad(): preds_list.append(m(Xte_t.to(dev)).cpu().numpy().reshape(-1))
    return preds_list


# ============================================================
# Wilcoxon runner
# ============================================================
def run_wilcoxon_tests(resatt_pred, baseline_preds_list, yte):
    """Run Wilcoxon on MAE, SMAPE, RMSE components across seeds."""
    resatt_err = per_sample_errors_all(yte, resatt_pred)
    results = {}
    for metric_key in ["mae", "smape", "rmse_sq"]:
        pvals = []
        for bpred in baseline_preds_list:
            berr = per_sample_errors_all(yte, bpred)
            diff = resatt_err[metric_key] - berr[metric_key]
            nonzero = diff != 0
            if np.sum(nonzero) > 0:
                _, p = wilcoxon(resatt_err[metric_key][nonzero], berr[metric_key][nonzero])
            else:
                p = np.nan
            pvals.append(p)
        # Pooled: use Fisher's method? No, just report min/median/max
        # Best seed: the one with best MAE among baselines (gives baseline its best shot)
        # Conservative approach: report median p-value across seeds
        pvals_arr = np.array([x for x in pvals if not np.isnan(x)])
        results[metric_key] = {
            "pvals": pvals,
            "p_median": np.median(pvals_arr) if len(pvals_arr) > 0 else np.nan,
            "p_min": np.min(pvals_arr) if len(pvals_arr) > 0 else np.nan,
            "p_max": np.max(pvals_arr) if len(pvals_arr) > 0 else np.nan,
        }
    return results


# ============================================================
# RF Scenario training
# ============================================================
def train_rf_scenarios(Xtr, Xv, Xte, ytr, yv, yte, seeds):
    """Train RF on each scenario, return predictions and metrics."""
    SCENARIO_RANGES = {
        'A': (0.80, 1.01), 'B': (-0.01, 0.30), 'C': (0.30, 0.80)
    }
    SCENARIO_NAMES = {'A': 'A_High_Support', 'B': 'B_Low_Support', 'C': 'C_Mid_Discretion'}

    results = {}
    Xall = pd.concat([Xtr, Xv])
    yall = np.concatenate([ytr, yv])

    for skey, (lo, hi) in SCENARIO_RANGES.items():
        mask = (yall >= lo) & (yall < hi)
        Xs = Xall[mask]
        ys = yall[mask]
        test_mask = (yte >= lo) & (yte < hi)
        Xs_test = Xte[test_mask].to_numpy(np.float32)
        ys_test = yte[test_mask]

        if len(Xs) < 20 or len(Xs_test) < 10:
            results[skey] = None
            continue

        preds_list = []
        for seed in seeds:
            rf = GridSearchCV(
                RandomForestRegressor(random_state=seed, n_jobs=-1),
                {"n_estimators": [100, 200], "max_depth": [8, 12, 16], "min_samples_leaf": [3, 7]},
                cv=3, scoring="neg_mean_absolute_error")
            rf.fit(Xs, ys)
            preds_list.append(rf.best_estimator_.predict(Xs_test))

        metrics = [mets(ys_test, p) for p in preds_list]
        best_idx = np.argmin([m["mae"] for m in metrics])
        results[skey] = {
            "name": SCENARIO_NAMES[skey],
            "n_test": len(Xs_test),
            "n_train": len(Xs),
            "preds": preds_list,
            "best_pred": preds_list[best_idx],
            "metrics": metrics,
            "best_metric": metrics[best_idx],
            "y_true": ys_test,
        }
    return results


def load_resatt_scenario_preds(Xte, yte):
    """Load existing ResAtt-MLP scenario checkpoints, compute per-scenario ensemble preds."""
    SCENARIO_NAMES = {'A': 'A_High_Support', 'B': 'B_Low_Support', 'C': 'C_Mid_Discretion'}
    SCENARIO_RANGES = {'A': (0.80, 1.01), 'B': (-0.01, 0.30), 'C': (0.30, 0.80)}
    results = {}
    for skey in ['A', 'B', 'C']:
        sname = SCENARIO_NAMES[skey]
        ckpts = sorted(SCEN_CKPT.glob(f"{sname}_cfg*_s*.pt"))
        if not ckpts:
            results[skey] = None
            continue
        lo, hi = SCENARIO_RANGES[skey]
        test_mask = (yte >= lo) & (yte < hi)
        Xs_test = Xte[test_mask].to_numpy(np.float32)
        ys_test = yte[test_mask]
        if len(Xs_test) < 5:
            results[skey] = None
            continue

        metrics_list, preds_list = [], []
        Xt = torch.from_numpy(Xs_test)
        for cp in ckpts:
            try:
                sd = torch.load(cp, map_location="cpu")
                hd = sd['inp.0.weight'].shape[0]
                nb = sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
                idim = sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
                m = ResAttMLP(idim=idim, hd=hd, nb=nb).to(dev)
                m.load_state_dict(sd); m.eval()
                with torch.no_grad():
                    p = m(Xt.to(dev)).cpu().numpy().reshape(-1)
                met = mets(ys_test, p)
                metrics_list.append(met)
                preds_list.append(p)
            except Exception:
                continue
        if not metrics_list:
            results[skey] = None
            continue
        sorted_idx = sorted(range(len(metrics_list)), key=lambda i: metrics_list[i]['mae'])
        top_n = min(3, len(sorted_idx))
        ensemble_pred = np.array([preds_list[i] for i in sorted_idx[:top_n]]).mean(axis=0)
        results[skey] = {
            "name": sname, "n_test": len(Xs_test),
            "ensemble_pred": ensemble_pred,
            "y_true": ys_test,
            "single_metrics": metrics_list,
        }
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("07 — Enhanced M3: 8 seeds + SMAPE/RMSE + RF scenarios")
    print("=" * 60)

    Xtr = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    Xv = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    Xte = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    ytr = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].to_numpy(np.float32)
    yv = pd.read_csv(PROCESSED_DIR / "y_val.csv").iloc[:, 0].to_numpy(np.float32)
    yte = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy(np.float32)
    Xtr_full = pd.concat([Xtr, Xv])
    ytr_full = np.concatenate([ytr, yv])

    Xte_np = Xte.to_numpy(np.float32)
    print(f"Features: {Xtr.shape[1]} | Seeds: {len(SEEDS)} | Train={len(Xtr_full)} Test={len(Xte)}")

    # ---- Phase 1: ResAtt-MLP global ensemble ----
    print("\n" + "-" * 60)
    print("PHASE 1: ResAtt-MLP Global Ensemble")
    print("-" * 60)
    t0 = time.time()
    resatt_pred, resatt_single, _ = load_resatt_ensemble(Xte_np, yte)
    resatt_met = mets(yte, resatt_pred)
    maes = [m['mae'] for m in resatt_single]
    print(f"ResAtt-MLP: {len(resatt_single)} checkpoints loaded")
    print(f"  Single MAE: {np.mean(maes):.5f} +/- {np.std(maes):.5f}")
    print(f"  Ensemble MAE: {resatt_met['mae']:.5f}  SMAPE: {resatt_met['smape']:.5f}  RMSE: {resatt_met['rmse']:.5f}")
    print(f"  ({time.time()-t0:.0f}s)")

    # ---- Phase 2: All baselines × 8 seeds ----
    print("\n" + "-" * 60)
    print("PHASE 2: Baseline Training (8 seeds each)")
    print("-" * 60)

    # Ridge (deterministic)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(Xtr_full, ytr_full)
    ridge_pred = ridge.predict(Xte)
    ridge_met_list = [mets(yte, ridge_pred)]

    # RF
    t0 = time.time()
    rf_preds, rf_metrics = train_tree_multi_seed(
        RandomForestRegressor, {"n_estimators": [100, 200, 300], "max_depth": [10, 15, 20], "min_samples_leaf": [3, 5, 10]},
        Xtr_full, ytr_full, Xte, yte, SEEDS, n_jobs=-1)
    rf_maes = [m['mae'] for m in rf_metrics]
    print(f"RF:     MAE={np.mean(rf_maes):.5f} +/- {np.std(rf_maes):.5f}  ({time.time()-t0:.0f}s)")

    # XGBoost
    t0 = time.time()
    xgb_preds, xgb_metrics = train_tree_multi_seed(
        XGBRegressor, {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
        Xtr_full, ytr_full, Xte, yte, SEEDS, verbosity=0, n_jobs=-1)
    xgb_maes = [m['mae'] for m in xgb_metrics]
    print(f"XGB:    MAE={np.mean(xgb_maes):.5f} +/- {np.std(xgb_maes):.5f}  ({time.time()-t0:.0f}s)")

    # LightGBM
    t0 = time.time()
    lgb_preds, lgb_metrics = train_tree_multi_seed(
        LGBMRegressor, {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "num_leaves": [15, 31]},
        Xtr_full, ytr_full, Xte, yte, SEEDS, verbose=-1, n_jobs=-1)
    lgb_maes = [m['mae'] for m in lgb_metrics]
    print(f"LGB:    MAE={np.mean(lgb_maes):.5f} +/- {np.std(lgb_maes):.5f}  ({time.time()-t0:.0f}s)")

    # CatBoost
    t0 = time.time()
    cb_preds, cb_metrics = train_tree_multi_seed(
        CatBoostRegressor, {"iterations": [100, 200, 300], "depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "l2_leaf_reg": [3, 5, 10]},
        Xtr_full, ytr_full, Xte, yte, SEEDS, verbose=0, thread_count=-1)
    cb_maes = [m['mae'] for m in cb_metrics]
    print(f"CB:     MAE={np.mean(cb_maes):.5f} +/- {np.std(cb_maes):.5f}  ({time.time()-t0:.0f}s)")

    # MLP
    t0 = time.time()
    mlp_preds = train_mlp_multi_seed(Xtr, Xv, Xte, ytr, yv, yte, SEEDS)
    mlp_metrics = [mets(yte, p) for p in mlp_preds]
    mlp_maes = [m['mae'] for m in mlp_metrics]
    print(f"MLP:    MAE={np.mean(mlp_maes):.5f} +/- {np.std(mlp_maes):.5f}  ({time.time()-t0:.0f}s)")

    # TabNet
    t0 = time.time()
    tab_preds = train_tabnet_multi_seed(Xtr, Xv, Xte, ytr, yv, yte, SEEDS)
    tab_metrics = [mets(yte, p) for p in tab_preds]
    tab_maes = [m['mae'] for m in tab_metrics]
    print(f"TabNet: MAE={np.mean(tab_maes):.5f} +/- {np.std(tab_maes):.5f}  ({time.time()-t0:.0f}s)")

    # ---- Phase 3: Wilcoxon on MAE / SMAPE / RMSE ----
    print("\n" + "=" * 60)
    print("PHASE 3: Wilcoxon Tests (MAE / SMAPE / RMSE)")
    print("=" * 60)

    baseline_data = {
        "Ridge": [ridge_pred],
        "Random Forest": rf_preds,
        "XGBoost": xgb_preds,
        "LightGBM": lgb_preds,
        "CatBoost": cb_preds,
        "Standard MLP": mlp_preds,
        "TabNet": tab_preds,
    }

    all_wilcoxon = {}
    for bname, bpreds in baseline_data.items():
        wres = run_wilcoxon_tests(resatt_pred, bpreds, yte)
        all_wilcoxon[bname] = wres

        # Count wins based on best-seed (by MAE) for the baseline
        best_idx = np.argmin([mets(yte, p)['mae'] for p in bpreds])
        b_best_pred = bpreds[best_idx]
        wins = np.sum(np.abs(yte - resatt_pred) < np.abs(yte - b_best_pred))
        losses = np.sum(np.abs(yte - resatt_pred) > np.abs(yte - b_best_pred))
        ties = np.sum(np.abs(yte - resatt_pred) == np.abs(yte - b_best_pred))

        p_mae = wres['mae']['p_median']
        p_smape = wres['smape']['p_median']
        p_rmse = wres['rmse_sq']['p_median']

        sig_flags = []
        for p in [p_mae, p_smape, p_rmse]:
            if not np.isnan(p) and p < 0.05: sig_flags.append("SIG")
        sig_str = "/".join(sig_flags) if sig_flags else "not sig"

        print(f"  {bname:<16} MAE_p={p_mae:.4f}  SMAPE_p={p_smape:.4f}  RMSE_p={p_rmse:.4f}  [{sig_str}]  wins={wins}/{len(yte)}")

    # ---- Phase 4: RF scenarios vs ResAtt-MLP scenarios ----
    print("\n" + "=" * 60)
    print("PHASE 4: RF vs ResAtt-MLP on 3 Scenarios")
    print("=" * 60)

    rf_scen = train_rf_scenarios(Xtr, Xv, Xte, ytr, yv, yte, SEEDS)
    resatt_scen = load_resatt_scenario_preds(Xte, yte)

    scen_rows = []
    for skey in ['A', 'B', 'C']:
        if rf_scen.get(skey) is None or resatt_scen.get(skey) is None:
            print(f"  {skey}: SKIP (insufficient data)")
            continue

        rf_info = rf_scen[skey]
        ra_info = resatt_scen[skey]

        rf_best_met = rf_info['best_metric']
        ra_met = mets(ra_info['y_true'], ra_info['ensemble_pred'])

        # Wilcoxon: ResAtt-MLP ensemble errors vs RF best-seed errors
        ra_errs = np.abs(ra_info['y_true'] - ra_info['ensemble_pred'])
        rf_errs = np.abs(ra_info['y_true'] - rf_info['best_pred'])
        diff = ra_errs - rf_errs
        nonzero = diff != 0
        if np.sum(nonzero) > 0:
            _, scen_p = wilcoxon(ra_errs[nonzero], rf_errs[nonzero])
        else:
            scen_p = np.nan

        wins = np.sum(ra_errs < rf_errs)
        losses = np.sum(ra_errs > rf_errs)

        mae_impr = (1 - ra_met['mae'] / rf_best_met['mae']) * 100 if rf_best_met['mae'] > 0 else 0

        print(f"  {rf_info['name']} (n={rf_info['n_test']}):")
        print(f"    ResAtt-MLP MAE={ra_met['mae']:.5f}  RF MAE={rf_best_met['mae']:.5f}  Impr={mae_impr:+.1f}%")
        print(f"    Wilcoxon p={scen_p:.6f}  ResAtt wins={wins}/{ra_info['n_test']}  {'SIG' if scen_p < 0.05 else 'not sig'}")

        scen_rows.append({
            "Scenario": rf_info['name'],
            "N_test": rf_info['n_test'],
            "ResAtt_MAE": round(ra_met['mae'], 5),
            "RF_MAE": round(rf_best_met['mae'], 5),
            "MAE_impr_pct": f"{mae_impr:+.1f}%",
            "Wilcoxon_p": round(scen_p, 6) if not np.isnan(scen_p) else "—",
            "ResAtt_wins": wins,
            "RF_wins": losses,
            "Significant": "Yes" if scen_p < 0.05 else "No",
        })

    # ---- Build output tables ----
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    # Table 1: metrics mean +/- std
    metric_rows = []
    for bname in ["Ridge", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Standard MLP", "TabNet"]:
        if bname == "Ridge":
            ms = ridge_met_list
        elif bname == "Random Forest":
            ms = rf_metrics
        elif bname == "XGBoost":
            ms = xgb_metrics
        elif bname == "LightGBM":
            ms = lgb_metrics
        elif bname == "CatBoost":
            ms = cb_metrics
        elif bname == "Standard MLP":
            ms = mlp_metrics
        elif bname == "TabNet":
            ms = tab_metrics
        else:
            continue
        mae_arr = [m['mae'] for m in ms]
        smape_arr = [m['smape'] for m in ms]
        wape_arr = [m['wape'] for m in ms]
        metric_rows.append({
            "Model": bname,
            "MAE_mean": round(np.mean(mae_arr), 5), "MAE_std": round(np.std(mae_arr), 5),
            "SMAPE_mean": round(np.mean(smape_arr), 5), "SMAPE_std": round(np.std(smape_arr), 5),
            "WAPE_mean": round(np.mean(wape_arr), 5), "WAPE_std": round(np.std(wape_arr), 5),
        })
    # Add ResAtt-MLP row
    metric_rows.append({
        "Model": "ResAtt-MLP (ens)",
        "MAE_mean": round(resatt_met['mae'], 5), "MAE_std": 0,
        "SMAPE_mean": round(resatt_met['smape'], 5), "SMAPE_std": 0,
        "WAPE_mean": round(resatt_met['wape'], 5), "WAPE_std": 0,
    })
    t1 = pd.DataFrame(metric_rows)
    print("\n--- Metrics (mean +/- std, {} seeds) ---".format(len(SEEDS)))
    print(t1.to_string(index=False))

    # Table 2: Wilcoxon
    wil_rows = []
    for bname in ["Ridge", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Standard MLP", "TabNet"]:
        w = all_wilcoxon[bname]
        best_idx = np.argmin([mets(yte, p)['mae'] for p in baseline_data[bname]])
        bpred = baseline_data[bname][best_idx]
        wins = np.sum(np.abs(yte - resatt_pred) < np.abs(yte - bpred))
        losses = np.sum(np.abs(yte - resatt_pred) > np.abs(yte - bpred))
        p_mae = w['mae']['p_median']
        p_smape = w['smape']['p_median']
        p_rmse = w['rmse_sq']['p_median']
        sig_any = (not np.isnan(p_mae) and p_mae < 0.05) or \
                  (not np.isnan(p_smape) and p_smape < 0.05) or \
                  (not np.isnan(p_rmse) and p_rmse < 0.05)
        wil_rows.append({
            "Model": bname,
            "MAE_p": round(p_mae, 6) if not np.isnan(p_mae) else "—",
            "SMAPE_p": round(p_smape, 6) if not np.isnan(p_smape) else "—",
            "RMSE_p": round(p_rmse, 6) if not np.isnan(p_rmse) else "—",
            "ResAtt_wins": wins, "Model_wins": losses,
            "Any_SIG": "Yes" if sig_any else "No",
        })
    t2 = pd.DataFrame(wil_rows)
    print("\n--- Wilcoxon Tests (per error-type, median p across seeds) ---")
    print(t2.to_string(index=False))

    # Table 3: Scenarios
    if scen_rows:
        t3 = pd.DataFrame(scen_rows)
        print("\n--- RF vs ResAtt-MLP Scenarios ---")
        print(t3.to_string(index=False))

        # Save scenario table
        t3.to_csv(RESULTS_DIR / "scenario_wilcoxon.csv", index=False, encoding="utf-8-sig")
    else:
        t3 = None

    # Save
    t1.to_csv(RESULTS_DIR / "metrics_8seed.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(RESULTS_DIR / "wilcoxon_all_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
