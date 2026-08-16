"""
09c_diverse_tuning.py
结合 08 的架构多样性 + 09 调参发现的 alpha=0.4 偏好。

搜索空间 (3 x 3 = 9 configs):
  - hd: [96, 128, 160]
  - nb (残差块数): [2, 3, 4]

固定: do=0.12, tau=1.0, alpha=0.4 (mse_w=0.4, smape_w=0.6)
       lr=1e-3, bs=64, wd=1e-5, max_ep=1500, pat=180

9 configs x 5 seeds = 45 models, Top-10 简单平均集成。
"""
import copy, warnings, random, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from collections import Counter

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

SEEDS = MAIN_SEEDS
HD_VALUES = [96, 128, 160]
NB_VALUES = [2, 3, 4]
ALPHA = 0.4  # 从 09 调参确定的最优值

BASE_CFG = {"do": 0.12, "tau": 1.0, "lr": 1e-3, "bs": 64,
            "max_ep": 1500, "pat": 180, "wd": 1e-5,
            "mse_w": ALPHA, "smape_w": 1.0 - ALPHA}


def build_configs():
    configs = []
    for hd in HD_VALUES:
        for nb in NB_VALUES:
            cfg = dict(BASE_CFG); cfg["hd"] = hd; cfg["nb"] = nb
            cfg["label"] = f"hd{hd}_nb{nb}"
            configs.append(cfg)
    return configs


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


def train_one(cfg, seed, data, ckpt_path):
    Xtr, Xv, Xte, ytr, yv, yte, it_tr, it_v, it_te, im_tr, im_v, im_te, ir_tr, ir_v, ir_te = data
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    Xf = np.concatenate([Xtr, Xv], axis=0); yf = np.concatenate([ytr, yv], axis=0)
    itf = np.concatenate([it_tr, it_v], axis=0); imf = np.concatenate([im_tr, im_v], axis=0)
    irf = np.concatenate([ir_tr, ir_v], axis=0)
    vn = min(80, len(Xf)//10)
    Xt=Xf[:-vn]; yt_tr=yf[:-vn]; it_t=itf[:-vn]; im_t=imf[:-vn]; ir_t=irf[:-vn]
    Xv2=Xf[-vn:]; yv_tr=yf[-vn:]; it_v2=itf[-vn:]; im_v2=imf[-vn:]; ir_v2=irf[-vn:]

    tl = mkldr_multitask(Xt, yt_tr, it_t, im_t, ir_t, cfg["bs"], shuffle=True)
    vl = mkldr_multitask(Xv2, yv_tr, it_v2, im_v2, ir_v2, cfg["bs"], shuffle=False)

    model = ResAttMLP(idim=Xt.shape[1], hd=cfg["hd"], nb=cfg["nb"],
                       do=cfg["do"], tau=cfg["tau"],
                       use_gate=True, use_skip=True, multitask=True).to(dev)
    crit = MaskedCompositeLoss(mse_weight=cfg["mse_w"], smape_weight=cfg["smape_w"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)

    bv, bs_state, wait = float("inf"), None, 0
    for ep in range(cfg["max_ep"]):
        model.train()
        for batch in tl:
            xb, itb, imb, irb, _ = [b.to(dev) for b in batch]
            opt.zero_grad()
            ip, tp = model(xb, item_ratios=irb)
            loss, _ = crit(ip, itb, imb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        vr = evalt_multitask_full(model, vl, dev)
        if vr["global"]["mae"] < bv - 1e-7:
            bv = vr["global"]["mae"]; bs_state = copy.deepcopy(model.state_dict()); wait = 0
        else: wait += 1
        if wait >= cfg["pat"]: break

    model.load_state_dict(bs_state); torch.save(bs_state, ckpt_path)
    tel = mkldr_multitask(Xte, yte, it_te, im_te, ir_te, cfg["bs"], shuffle=False)
    return bv, model, evalt_multitask_full(model, tel, dev)


def main():
    print("=" * 60)
    print("09c — Diverse Tuning: hd x nb (alpha=0.4 fixed)")
    print("=" * 60)
    print(f"Search: hd={HD_VALUES} nb={NB_VALUES} alpha={ALPHA}")
    print(f"Total: {len(HD_VALUES)*len(NB_VALUES)}x{len(SEEDS)} = {len(HD_VALUES)*len(NB_VALUES)*len(SEEDS)} models")

    data = load_data()
    Xte, yte, it_te, im_te, ir_te = data[2], data[5], data[8], data[11], data[14]
    configs = build_configs()
    all_res = []; trained = 0; cached = 0; t0 = time.time()

    for ci, cfg in enumerate(configs):
        for si, seed in enumerate(SEEDS):
            ckpt_path = CKPT_DIR / f"div_{cfg['label']}_s{seed}.pt"
            label = f"[{ci+1}/{len(configs)}] {cfg['label']}_s{seed}"

            if ckpt_path.exists():
                try:
                    sd = torch.load(ckpt_path, map_location="cpu")
                    hd = sd["inp.0.weight"].shape[0]
                    nb = sum(1 for k in sd if k.startswith("stack.") and k.endswith(".block.0.weight"))
                    idim = sd["gate.1.weight"].shape[0] if "gate.1.weight" in sd else sd["inp.0.weight"].shape[1]
                    m = ResAttMLP(idim=idim, hd=hd, nb=nb, multitask=True).to(dev)
                    m.load_state_dict(sd); m.eval()

                    Xtr_np, Xv_np = data[0], data[1]; yt_np, yv_np = data[3], data[4]
                    it_tn, it_vn = data[6], data[7]; im_tn, im_vn = data[9], data[10]
                    ir_tn, ir_vn = data[12], data[13]
                    Xf = np.concatenate([Xtr_np, Xv_np], axis=0); yf = np.concatenate([yt_np, yv_np], axis=0)
                    itf = np.concatenate([it_tn, it_vn], axis=0); imf = np.concatenate([im_tn, im_vn], axis=0)
                    irf = np.concatenate([ir_tn, ir_vn], axis=0)
                    vn = min(80, len(Xf)//10)
                    vl = mkldr_multitask(Xf[-vn:], yf[-vn:], itf[-vn:], imf[-vn:], irf[-vn:], cfg["bs"], shuffle=False)
                    vr = evalt_multitask_full(m, vl, dev); vmae = vr["global"]["mae"]

                    tel = mkldr_multitask(Xte, yte, it_te, im_te, ir_te, cfg["bs"], shuffle=False)
                    tr = evalt_multitask_full(m, tel, dev)
                    all_res.append((vmae, cfg["label"], seed, m, tr, hd, nb))
                    cached += 1
                    print(f"  {label}: v_mae={vmae:.5f} t_mae={tr['global']['mae']:.5f}  (cached)")
                    continue
                except Exception as e:
                    ckpt_path.unlink(missing_ok=True)

            tt = time.time()
            vmae, model, test_res = train_one(cfg, seed, data, ckpt_path)
            all_res.append((vmae, cfg["label"], seed, model, test_res, cfg["hd"], cfg["nb"]))
            trained += 1
            print(f"  {label}: v_mae={vmae:.5f} t_mae={test_res['global']['mae']:.5f}  ({time.time()-tt:.0f}s)")

    elapsed = (time.time()-t0)/60
    print(f"\nTrained: {trained} | Cached: {cached} | Time: {elapsed:.1f} min")

    # Top-10 ensemble
    all_res.sort(key=lambda x: x[0])
    top_n = min(10, len(all_res)); top10 = all_res[:top_n]

    print(f"\nTop-{top_n} by validation MAE:")
    hd_c = Counter(); nb_c = Counter()
    for i, (vmae, label, seed, _, tr, hd, nb) in enumerate(top10):
        hd_c[hd] += 1; nb_c[nb] += 1
        print(f"  {i+1}. {label}_s{seed}  hd={hd} nb={nb}  v_mae={vmae:.5f}  t_mae={tr['global']['mae']:.5f}")
    print(f"  hd dist: {dict(sorted(hd_c.items()))}  nb dist: {dict(sorted(nb_c.items()))}")

    # Ensemble
    Xtt = torch.from_numpy(Xte); irtt = torch.from_numpy(ir_te)
    aip = np.zeros((len(Xte), 16, top_n), dtype=np.float32)
    atp = np.zeros((len(Xte), top_n), dtype=np.float32)
    for i, (_, _, _, mdl, _, _, _) in enumerate(top10):
        mdl.eval()
        with torch.no_grad():
            ip, tp = mdl(Xtt.to(dev), item_ratios=irtt.to(dev))
            aip[:,:,i] = ip.cpu().numpy(); atp[:,i] = tp.cpu().numpy().reshape(-1)

    eip = aip.mean(axis=2); etp = atp.mean(axis=1)
    gm = mets(yte, etp); im = per_item_metrics(eip, it_te, im_te)

    print(f"\n{'='*60}")
    print(f"FINAL — Diverse Tuning Ensemble (Top-{top_n})")
    print(f"{'='*60}")
    print(f"  MAE   = {gm['mae']:.5f}")
    print(f"  SMAPE = {gm['smape']:.5f}")
    print(f"  WAPE  = {gm['wape']:.5f}")
    print(f"  RMSE  = {gm['rmse']:.5f}")
    print(f"\n{'='*60}")
    print("Per-Item Metrics")
    print(f"{'='*60}")
    print(format_item_metrics_table(im))

    pd.DataFrame([{"Model":"ResAtt-MLP-MultiTask-Diverse",
        "MAE":round(gm["mae"],5),"SMAPE":round(gm["smape"],5),
        "WAPE":round(gm["wape"],5),"RMSE":round(gm["rmse"],5),
        "BestHD":str(dict(hd_c)),"BestNB":str(dict(nb_c)),
        "EnsembleSize":top_n}
    ]).to_csv(RESULTS_DIR / "diverse_tuning.csv", index=False, encoding="utf-8-sig")

    from models.resatt_mlp import _SHORT_NAMES_ORDER, _ITEM_CN_NAMES
    rigid_set={"medical","disability_comp","death_comp"}
    flex_set={"transport","accommodation","solace","other"}
    rows=[]
    for i in range(16):
        sn=_SHORT_NAMES_ORDER[i]; cn=_ITEM_CN_NAMES[i]; r=im[sn]
        tp="rigid" if sn in rigid_set else ("flexible" if sn in flex_set else "moderate")
        rows.append({"Item_CN":cn,"Item_EN":sn,"Type":tp,"N_Samples":r["n_samples"],
            "MAE":round(r["mae"],5) if not np.isnan(r["mae"]) else "",
            "SMAPE":round(r["smape"],5) if not np.isnan(r["smape"]) else ""})
    pd.DataFrame(rows).to_csv(RESULTS_DIR/"diverse_per_item.csv", index=False, encoding="utf-8-sig")
    for j in range(min(3,top_n)): torch.save(top10[j][3].state_dict(), CKPT_DIR/f"diverse_ens_{j}.pt")
    print(f"\nSaved to results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
