"""
09b_combined_ensemble.py
合并 08 (25 models) + 09 tuning (45 models) = 70 candidates,
按验证集 MAE 选 Top-10 简单平均集成。
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch

os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')
torch.set_num_threads(8)
dev = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from config import PROCESSED_DIR, RESULTS_DIR, CKPT_DIR, MAIN_SEEDS
from models.resatt_mlp import (
    ResAttMLP, mets, per_item_metrics, format_item_metrics_table,
    mkldr_multitask, evalt_multitask_full,
)

SEEDS = MAIN_SEEDS  # [42, 123, 456, 789, 1024]

def load_data():
    Xtr = pd.read_csv(PROCESSED_DIR / "X_train.csv").to_numpy(np.float32)
    Xv  = pd.read_csv(PROCESSED_DIR / "X_val.csv").to_numpy(np.float32)
    Xte = pd.read_csv(PROCESSED_DIR / "X_test.csv").to_numpy(np.float32)
    ytr = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:,0].to_numpy(np.float32)
    yv  = pd.read_csv(PROCESSED_DIR / "y_val.csv").iloc[:,0].to_numpy(np.float32)
    yte = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:,0].to_numpy(np.float32)
    it_tr = pd.read_csv(PROCESSED_DIR / "item_targets_train.csv").to_numpy(np.float32)
    it_v  = pd.read_csv(PROCESSED_DIR / "item_targets_val.csv").to_numpy(np.float32)
    it_te = pd.read_csv(PROCESSED_DIR / "item_targets_test.csv").to_numpy(np.float32)
    im_tr = pd.read_csv(PROCESSED_DIR / "item_masks_train.csv").to_numpy(np.float32)
    im_v  = pd.read_csv(PROCESSED_DIR / "item_masks_val.csv").to_numpy(np.float32)
    im_te = pd.read_csv(PROCESSED_DIR / "item_masks_test.csv").to_numpy(np.float32)
    ir_tr = pd.read_csv(PROCESSED_DIR / "item_ratios_train.csv").to_numpy(np.float32)
    ir_v  = pd.read_csv(PROCESSED_DIR / "item_ratios_val.csv").to_numpy(np.float32)
    ir_te = pd.read_csv(PROCESSED_DIR / "item_ratios_test.csv").to_numpy(np.float32)
    return Xtr, Xv, Xte, ytr, yv, yte, it_tr, it_v, it_te, im_tr, im_v, im_te, ir_tr, ir_v, ir_te

def make_val_loader(data, bs=64):
    Xtr, Xv, ytr, yv, it_tr, it_v, im_tr, im_v, ir_tr, ir_v = (
        data[0], data[1], data[3], data[4], data[6], data[7],
        data[9], data[10], data[12], data[13])
    Xf = np.concatenate([Xtr, Xv], axis=0)
    yf = np.concatenate([ytr, yv], axis=0)
    itf = np.concatenate([it_tr, it_v], axis=0)
    imf = np.concatenate([im_tr, im_v], axis=0)
    irf = np.concatenate([ir_tr, ir_v], axis=0)
    vn = min(80, len(Xf)//10)
    return mkldr_multitask(Xf[-vn:], yf[-vn:], itf[-vn:], imf[-vn:], irf[-vn:], bs, shuffle=False)

def try_load(ckpt_path):
    """Try loading a checkpoint; return (model, hd, nb) or (None, None, None)."""
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        hd = sd["inp.0.weight"].shape[0]
        nb = sum(1 for k in sd if k.startswith("stack.") and k.endswith(".block.0.weight"))
        idim = sd["gate.1.weight"].shape[0] if "gate.1.weight" in sd else sd["inp.0.weight"].shape[1]
        m = ResAttMLP(idim=idim, hd=hd, nb=nb, multitask=True).to(dev)
        m.load_state_dict(sd); m.eval()
        return m, hd, nb
    except Exception:
        return None, None, None

def main():
    print("=" * 60)
    print("09b — Combined Ensemble (08 + 09 tuning)")
    print("=" * 60)

    data = load_data()
    Xte, yte, it_te, im_te, ir_te = data[2], data[5], data[8], data[11], data[14]
    val_loader = make_val_loader(data)

    # Collect all checkpoint paths
    ckpt_dir = CKPT_DIR
    all_ckpts = []

    # 08 checkpoints: multitask_cfg*_s*.pt
    for p in sorted(ckpt_dir.glob("multitask_cfg*_s*.pt")):
        all_ckpts.append(("08_multitask", p))
    # 09 tuning checkpoints: tune_hd*_a*_s*.pt
    for p in sorted(ckpt_dir.glob("tune_hd*_a*_s*.pt")):
        all_ckpts.append(("09_tuning", p))

    print(f"Total checkpoints found: {len(all_ckpts)}")
    if len(all_ckpts) == 0:
        print("ERROR: No checkpoints found!")
        return

    # Evaluate all models on validation set
    candidates = []
    for source, ckpt_path in all_ckpts:
        model, hd, nb = try_load(ckpt_path)
        if model is None:
            print(f"  SKIP (load err): {ckpt_path.name}")
            continue
        vr = evalt_multitask_full(model, val_loader, dev)
        val_mae = vr["global"]["mae"]
        candidates.append((val_mae, source, ckpt_path.name, model, hd, nb))

    candidates.sort(key=lambda x: x[0])
    print(f"Successfully evaluated: {len(candidates)} models")

    # Top-10
    top_n = min(10, len(candidates))
    top10 = candidates[:top_n]

    print(f"\nTop-{top_n} by validation MAE:")
    for i, (vmae, src, name, _, hd, nb) in enumerate(top10):
        print(f"  {i+1}. [{src}] {name}  hd={hd} nb={nb}  val_mae={vmae:.5f}")

    # Source distribution
    from collections import Counter
    src_counts = Counter(s for _, s, _, _, _, _ in top10)
    hd_counts = Counter(hd for _, _, _, _, hd, _ in top10)
    print(f"\n  Source distribution: {dict(src_counts)}")
    print(f"  hd distribution:     {dict(sorted(hd_counts.items()))}")

    # ---- Ensemble prediction on test set ----
    Xt_test = torch.from_numpy(Xte)
    ir_test_t = torch.from_numpy(ir_te)

    all_ip = np.zeros((len(Xte), 16, top_n), dtype=np.float32)
    all_tp = np.zeros((len(Xte), top_n), dtype=np.float32)

    for i, (_, _, _, mdl, _, _) in enumerate(top10):
        mdl.eval()
        with torch.no_grad():
            ip, tp = mdl(Xt_test.to(dev), item_ratios=ir_test_t.to(dev))
            all_ip[:, :, i] = ip.cpu().numpy()
            all_tp[:, i] = tp.cpu().numpy().reshape(-1)

    ens_ip = all_ip.mean(axis=2)
    ens_tp = all_tp.mean(axis=1)

    global_m = mets(yte, ens_tp)
    item_m = per_item_metrics(ens_ip, it_te, im_te)

    print(f"\n{'='*60}")
    print(f"COMBINED ENSEMBLE — Global Metrics (Top-{top_n} of {len(candidates)})")
    print(f"{'='*60}")
    print(f"  MAE   = {global_m['mae']:.5f}")
    print(f"  SMAPE = {global_m['smape']:.5f}")
    print(f"  WAPE  = {global_m['wape']:.5f}")
    print(f"  RMSE  = {global_m['rmse']:.5f}")

    print(f"\n{'='*60}")
    print("Per-Item Support Rate Prediction Metrics")
    print(f"{'='*60}")
    print(format_item_metrics_table(item_m))

    # Save
    pd.DataFrame([{
        "Model": "ResAtt-MLP-MultiTask-Combined",
        "MAE": round(global_m["mae"], 5),
        "SMAPE": round(global_m["smape"], 5),
        "WAPE": round(global_m["wape"], 5),
        "RMSE": round(global_m["rmse"], 5),
        "CandidatePool": len(candidates),
        "EnsembleSize": top_n,
        "Source08": src_counts.get("08_multitask", 0),
        "Source09": src_counts.get("09_tuning", 0),
    }]).to_csv(RESULTS_DIR / "combined_ensemble.csv", index=False, encoding="utf-8-sig")

    from models.resatt_mlp import _SHORT_NAMES_ORDER, _ITEM_CN_NAMES
    rigid_set = {"medical","disability_comp","death_comp"}
    flex_set = {"transport","accommodation","solace","other"}
    rows = []
    for i in range(16):
        sn = _SHORT_NAMES_ORDER[i]; cn = _ITEM_CN_NAMES[i]; r = item_m[sn]
        tp = "rigid" if sn in rigid_set else ("flexible" if sn in flex_set else "moderate")
        rows.append({"Item_CN":cn,"Item_EN":sn,"Type":tp,"N_Samples":r["n_samples"],
                     "MAE":round(r["mae"],5) if not np.isnan(r["mae"]) else "",
                     "SMAPE":round(r["smape"],5) if not np.isnan(r["smape"]) else ""})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "combined_per_item.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved: {RESULTS_DIR / 'combined_ensemble.csv'}")
    print(f"       {RESULTS_DIR / 'combined_per_item.csv'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
