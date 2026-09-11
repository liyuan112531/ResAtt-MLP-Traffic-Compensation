"""
09_hyperparam_tuning.py — Lightweight grid search over hd and alpha.

Search space:
  - hd (hidden dim):    [96, 128, 160]
  - alpha (MSE weight): [0.2, 0.3, 0.4]  →  mse_w=alpha, smape_w=1-alpha

Fixed params: nb=3, do=0.12, tau=1.0, lr=1e-3, bs=64, wd=1e-5
Total: 3 x 3 = 9 configs x 5 seeds = 45 models
Ensemble: Top-10 by validation MAE, simple average.
"""
import copy, warnings, random, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')
torch.set_num_threads(8)
dev = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from config import PROCESSED_DIR, RESULTS_DIR, CKPT_DIR, MAIN_SEEDS
from models.resatt_mlp import (
    ResAttMLP, MaskedCompositeLoss,
    mets, per_item_metrics, format_item_metrics_table,
    mkldr_multitask, evalt_multitask_full,
)

SEEDS = MAIN_SEEDS  # [42, 123, 456, 789, 1024]

# ============================================================
# Grid search space
# ============================================================
HD_VALUES = [96, 128, 160]
ALPHA_VALUES = [0.2, 0.3, 0.4]  # alpha = mse_weight; smape_weight = 1 - alpha

# Fixed base config (all other params frozen)
BASE_CONFIG = {
    "nb": 3, "do": 0.12, "tau": 1.0, "lr": 1e-3, "bs": 64,
    "max_ep": 1500, "pat": 180, "wd": 1e-5,
}


def build_configs():
    """Generate all 9 config combinations."""
    configs = []
    for hd in HD_VALUES:
        for alpha in ALPHA_VALUES:
            cfg = dict(BASE_CONFIG)
            cfg["hd"] = hd
            cfg["alpha"] = alpha
            cfg["mse_w"] = alpha
            cfg["smape_w"] = 1.0 - alpha
            cfg["label"] = f"hd{hd}_a{alpha}"
            configs.append(cfg)
    return configs


def load_data():
    """Load all multi-task data."""
    Xtr = pd.read_csv(PROCESSED_DIR / "X_train.csv").to_numpy(np.float32)
    Xv = pd.read_csv(PROCESSED_DIR / "X_val.csv").to_numpy(np.float32)
    Xte = pd.read_csv(PROCESSED_DIR / "X_test.csv").to_numpy(np.float32)

    ytr = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].to_numpy(np.float32)
    yv = pd.read_csv(PROCESSED_DIR / "y_val.csv").iloc[:, 0].to_numpy(np.float32)
    yte = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy(np.float32)

    it_tr = pd.read_csv(PROCESSED_DIR / "item_targets_train.csv").to_numpy(np.float32)
    it_v = pd.read_csv(PROCESSED_DIR / "item_targets_val.csv").to_numpy(np.float32)
    it_te = pd.read_csv(PROCESSED_DIR / "item_targets_test.csv").to_numpy(np.float32)

    im_tr = pd.read_csv(PROCESSED_DIR / "item_masks_train.csv").to_numpy(np.float32)
    im_v = pd.read_csv(PROCESSED_DIR / "item_masks_val.csv").to_numpy(np.float32)
    im_te = pd.read_csv(PROCESSED_DIR / "item_masks_test.csv").to_numpy(np.float32)

    ir_tr = pd.read_csv(PROCESSED_DIR / "item_ratios_train.csv").to_numpy(np.float32)
    ir_v = pd.read_csv(PROCESSED_DIR / "item_ratios_val.csv").to_numpy(np.float32)
    ir_te = pd.read_csv(PROCESSED_DIR / "item_ratios_test.csv").to_numpy(np.float32)

    print(f"Data: Train={len(Xtr)} Val={len(Xv)} Test={len(Xte)}  Features={Xtr.shape[1]}")
    return Xtr, Xv, Xte, ytr, yv, yte, it_tr, it_v, it_te, im_tr, im_v, im_te, ir_tr, ir_v, ir_te


def train_one(cfg, seed, data, ckpt_path):
    """Train one model; return (val_mae, model_state, test_result)."""
    Xtr, Xv, Xte, ytr, yv, yte, it_tr, it_v, it_te, im_tr, im_v, im_te, ir_tr, ir_v, ir_te = data

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Merge train+val, reserve tail as internal validation
    X_full = np.concatenate([Xtr, Xv], axis=0)
    y_full = np.concatenate([ytr, yv], axis=0)
    it_full = np.concatenate([it_tr, it_v], axis=0)
    im_full = np.concatenate([im_tr, im_v], axis=0)
    ir_full = np.concatenate([ir_tr, ir_v], axis=0)

    vn = min(80, len(X_full) // 10)
    Xt = X_full[:-vn]; yt_tr = y_full[:-vn]
    it_t = it_full[:-vn]; im_t = im_full[:-vn]; ir_t = ir_full[:-vn]
    Xv2 = X_full[-vn:]; yv_tr = y_full[-vn:]
    it_v2 = it_full[-vn:]; im_v2 = im_full[-vn:]; ir_v2 = ir_full[-vn:]

    train_ldr = mkldr_multitask(Xt, yt_tr, it_t, im_t, ir_t, cfg["bs"], shuffle=True)
    val_ldr = mkldr_multitask(Xv2, yv_tr, it_v2, im_v2, ir_v2, cfg["bs"], shuffle=False)

    model = ResAttMLP(
        idim=Xt.shape[1], hd=cfg["hd"], nb=cfg["nb"],
        do=cfg["do"], tau=cfg["tau"],
        use_gate=True, use_skip=True, multitask=True,
    ).to(dev)

    criterion = MaskedCompositeLoss(mse_weight=cfg["mse_w"], smape_weight=cfg["smape_w"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_mae = float("inf")
    best_state = None
    wait = 0

    for ep in range(cfg["max_ep"]):
        model.train()
        for batch in train_ldr:
            xb, itb, imb, irb, _ = [b.to(dev) for b in batch]
            opt.zero_grad()
            ip, tp = model(xb, item_ratios=irb)
            loss, _ = criterion(ip, itb, imb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        vr = evalt_multitask_full(model, val_ldr, dev)
        val_mae = vr["global"]["mae"]
        if val_mae < best_val_mae - 1e-7:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= cfg["pat"]:
            break

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)

    # Test evaluation
    test_ldr = mkldr_multitask(Xte, yte, it_te, im_te, ir_te, cfg["bs"], shuffle=False)
    test_result = evalt_multitask_full(model, test_ldr, dev)

    return best_val_mae, model, test_result


def main():
    print("=" * 60)
    print("09 — Hyperparameter Tuning: hd x alpha Grid Search")
    print("=" * 60)
    print(f"Search: hd={HD_VALUES}  alpha={ALPHA_VALUES}")
    print(f"Seeds: {SEEDS}  →  {len(HD_VALUES)*len(ALPHA_VALUES)}x{len(SEEDS)} = {len(HD_VALUES)*len(ALPHA_VALUES)*len(SEEDS)} models")
    print(f"Device: {dev}")

    data = load_data()
    configs = build_configs()
    Xte, yte, it_te, im_te, ir_te = data[2], data[5], data[8], data[11], data[14]

    # Store: (val_mae, cfg_label, seed, model, test_result)
    all_results = []
    trained, cached = 0, 0
    t_start = time.time()

    for ci, cfg in enumerate(configs):
        for si, seed in enumerate(SEEDS):
            ckpt_path = CKPT_DIR / f"tune_{cfg['label']}_s{seed}.pt"
            label = f"[{ci+1}/{len(configs)}] {cfg['label']}_s{seed}"

            if ckpt_path.exists():
                try:
                    sd = torch.load(ckpt_path, map_location="cpu")
                    hd_dim = sd["inp.0.weight"].shape[0]
                    nb = sum(1 for k in sd if k.startswith("stack.") and k.endswith(".block.0.weight"))
                    idim = sd["gate.1.weight"].shape[0] if "gate.1.weight" in sd else sd["inp.0.weight"].shape[1]
                    m = ResAttMLP(idim=idim, hd=hd_dim, nb=nb, multitask=True).to(dev)
                    m.load_state_dict(sd); m.eval()

                    # Recompute val MAE for ranking
                    Xtr_np, Xv_np = data[0], data[1]
                    ytr_np, yv_np = data[3], data[4]
                    it_tr_np, it_v_np = data[6], data[7]
                    im_tr_np, im_v_np = data[9], data[10]
                    ir_tr_np, ir_v_np = data[12], data[13]
                    Xf = np.concatenate([Xtr_np, Xv_np], axis=0)
                    yf = np.concatenate([ytr_np, yv_np], axis=0)
                    itf = np.concatenate([it_tr_np, it_v_np], axis=0)
                    imf = np.concatenate([im_tr_np, im_v_np], axis=0)
                    irf = np.concatenate([ir_tr_np, ir_v_np], axis=0)
                    vn = min(80, len(Xf)//10)
                    vl = mkldr_multitask(Xf[-vn:], yf[-vn:], itf[-vn:], imf[-vn:], irf[-vn:], cfg["bs"], shuffle=False)
                    vr = evalt_multitask_full(m, vl, dev)
                    val_mae = vr["global"]["mae"]

                    test_ldr = mkldr_multitask(Xte, yte, it_te, im_te, ir_te, cfg["bs"], shuffle=False)
                    tr = evalt_multitask_full(m, test_ldr, dev)
                    all_results.append((val_mae, cfg["label"], seed, m, tr))
                    cached += 1
                    print(f"  {label}: val_mae={val_mae:.5f} test_mae={tr['global']['mae']:.5f}  (cached)")
                    continue
                except Exception as e:
                    print(f"  {label}: cache error ({e}), retraining...")
                    ckpt_path.unlink(missing_ok=True)

            t0 = time.time()
            val_mae, model, test_result = train_one(cfg, seed, data, ckpt_path)
            all_results.append((val_mae, cfg["label"], seed, model, test_result))
            trained += 1
            print(f"  {label}: val_mae={val_mae:.5f} test_mae={test_result['global']['mae']:.5f}  ({time.time()-t0:.0f}s)")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTrained: {trained} | Cached: {cached} | Time: {elapsed:.1f} min")

    # ============================================================
    # Top-10 ensemble (by validation MAE)
    # ============================================================
    all_results.sort(key=lambda x: x[0])  # sort by val_mae
    top_n = min(10, len(all_results))
    top10 = all_results[:top_n]

    print(f"\nTop-{top_n} models by validation MAE:")
    for i, (vmae, label, seed, _, tr) in enumerate(top10):
        print(f"  {i+1}. {label}_s{seed}  val_mae={vmae:.5f}  test_mae={tr['global']['mae']:.5f}")

    # ============================================================
    # Config distribution analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("Config Distribution in Top-10 Ensemble")
    print(f"{'='*60}")
    from collections import Counter
    hd_counts = Counter()
    alpha_counts = Counter()
    for _, label, _, _, _ in top10:
        for hd_v in HD_VALUES:
            if f"hd{hd_v}" in label:
                hd_counts[hd_v] += 1
        for a_v in ALPHA_VALUES:
            if f"_a{a_v}" in label:
                alpha_counts[a_v] += 1

    print(f"  hd distribution:    {dict(sorted(hd_counts.items()))}")
    print(f"  alpha distribution: {dict(sorted(alpha_counts.items()))}")

    # ============================================================
    # Ensemble prediction
    # ============================================================
    Xt_test = torch.from_numpy(Xte)
    ir_test_t = torch.from_numpy(ir_te)

    all_ip = np.zeros((len(Xte), 16, top_n), dtype=np.float32)
    all_tp = np.zeros((len(Xte), top_n), dtype=np.float32)

    for i, (_, _, _, mdl, _) in enumerate(top10):
        mdl.eval()
        with torch.no_grad():
            ip, tp = mdl(Xt_test.to(dev), item_ratios=ir_test_t.to(dev))
            all_ip[:, :, i] = ip.cpu().numpy()
            all_tp[:, i] = tp.cpu().numpy().reshape(-1)

    ens_ip = all_ip.mean(axis=2)   # [N, 16]
    ens_tp = all_tp.mean(axis=1)   # [N,]

    # ============================================================
    # Final metrics
    # ============================================================
    global_m = mets(yte, ens_tp)
    item_m = per_item_metrics(ens_ip, it_te, im_te)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — Top-{top_n} Ensemble (Grid Search)")
    print(f"{'='*60}")
    print(f"  MAE   = {global_m['mae']:.5f}")
    print(f"  SMAPE = {global_m['smape']:.5f}")
    print(f"  WAPE  = {global_m['wape']:.5f}")
    print(f"  RMSE  = {global_m['rmse']:.5f}")

    print(f"\n{'='*60}")
    print("Per-Item Support Rate Prediction Metrics")
    print(f"{'='*60}")
    print(format_item_metrics_table(item_m))

    # ============================================================
    # Save
    # ============================================================
    best_cfg = top10[0]
    print(f"\nBest single model: {best_cfg[1]}_s{best_cfg[2]}  val_mae={best_cfg[0]:.5f}")

    pd.DataFrame([{
        "Model": "ResAtt-MLP-MultiTask-Tuned",
        "MAE": round(global_m["mae"], 5),
        "SMAPE": round(global_m["smape"], 5),
        "WAPE": round(global_m["wape"], 5),
        "RMSE": round(global_m["rmse"], 5),
        "Best_HD": str(dict(hd_counts)),
        "Best_Alpha": str(dict(alpha_counts)),
        "EnsembleSize": top_n,
    }]).to_csv(RESULTS_DIR / "tuning_final.csv", index=False, encoding="utf-8-sig")

    # Per-item CSV
    from models.resatt_mlp import _SHORT_NAMES_ORDER, _ITEM_CN_NAMES
    rigid_set = {"medical", "disability_comp", "death_comp"}
    flex_set = {"transport", "accommodation", "solace", "other"}
    rows = []
    for i in range(16):
        sn = _SHORT_NAMES_ORDER[i]; cn = _ITEM_CN_NAMES[i]
        r = item_m[sn]
        if sn in rigid_set: tp = "rigid"
        elif sn in flex_set: tp = "flexible"
        else: tp = "moderate"
        rows.append({"Item_CN": cn, "Item_EN": sn, "Type": tp,
                     "N_Samples": r["n_samples"],
                     "MAE": round(r["mae"], 5) if not np.isnan(r["mae"]) else "",
                     "SMAPE": round(r["smape"], 5) if not np.isnan(r["smape"]) else ""})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tuning_per_item.csv", index=False, encoding="utf-8-sig")

    # Save top-3 models
    for j in range(min(3, top_n)):
        torch.save(top10[j][3].state_dict(), CKPT_DIR / f"tuning_ens_{j}.pt")

    print(f"\nSaved: {RESULTS_DIR / 'tuning_final.csv'}")
    print(f"       {RESULTS_DIR / 'tuning_per_item.csv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
