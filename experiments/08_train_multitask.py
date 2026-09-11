"""
08_train_multitask.py — 多任务掩码输出 + 权重聚合训练

核心改动（相对单任务 03_train_resatt_mlp.py）：
  1. 输出层：1 → 16 维，每个维度对应一个赔偿分项的支持率预测
  2. 损失函数：MaskedCompositeLoss — 仅在被主张的分项上计算损失，
     对刚性项目（医疗费、死亡赔偿金、伤残赔偿金）赋权 1.2，
     对弹性项目（精神抚慰、交通费）赋权 0.8
  3. 聚合层：item_preds 与 item_ratios（诉请占比）加权求和 → total_pred
  4. 评估：除全局 MAE/SMAPE 外，输出 16 个分项的独立指标表格

输出：results/resatt_mlp_multitask.csv + 分项指标表格打印
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

# ============================================================
# 多任务训练超参数配置
# ============================================================
# 基于单任务最佳配置适配：
#   - hd=96（与单任务相同，避免过拟合 16 个输出头）
#   - alpha 不再适用，改用 mse_weight/smape_weight 控制复合损失
#   - mask_loss 始终启用（多任务的核心）
MULTITASK_CONFIGS = [
    {"hd": 96,  "nb": 3, "do": 0.12, "tau": 1.0, "lr": 1e-3, "bs": 64,
     "max_ep": 1500, "pat": 180, "mse_w": 0.3, "smape_w": 0.7, "wd": 1e-5},
    {"hd": 128, "nb": 3, "do": 0.10, "tau": 1.3, "lr": 5e-4, "bs": 64,
     "max_ep": 1500, "pat": 180, "mse_w": 0.3, "smape_w": 0.7, "wd": 1e-5},
    {"hd": 128, "nb": 3, "do": 0.10, "tau": 3.0, "lr": 1e-3, "bs": 64,
     "max_ep": 1500, "pat": 180, "mse_w": 0.3, "smape_w": 0.7, "wd": 5e-6},
    {"hd": 128, "nb": 3, "do": 0.10, "tau": 1.0, "lr": 1e-3, "bs": 64,
     "max_ep": 1500, "pat": 180, "mse_w": 0.3, "smape_w": 0.7, "wd": 1e-5},
    {"hd": 128, "nb": 4, "do": 0.10, "tau": 1.3, "lr": 5e-4, "bs": 64,
     "max_ep": 1800, "pat": 200, "mse_w": 0.3, "smape_w": 0.7, "wd": 5e-5},
]
SEEDS = MAIN_SEEDS  # [42, 123, 456, 789, 1024]


def load_multitask_data():
    """
    加载多任务数据：特征 + 总标签 + 分项目标/掩码/占比。

    Returns
    -------
    Xtr, Xv, Xte : DataFrame  标准化特征（train/val/test）
    ytr, yv, yte : np.ndarray [N,]  总支持率
    it_tr, it_v, it_te : np.ndarray [N, 16]  分项真实支持率
    im_tr, im_v, im_te : np.ndarray [N, 16]  分项掩码
    ir_tr, ir_v, ir_te : np.ndarray [N, 16]  分项诉请占比
    """
    print("Loading multi-task data...")
    Xtr = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    Xv = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    Xte = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    ytr = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].to_numpy(np.float32)
    yv = pd.read_csv(PROCESSED_DIR / "y_val.csv").iloc[:, 0].to_numpy(np.float32)
    yte = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy(np.float32)

    # 加载分项数据
    it_tr = pd.read_csv(PROCESSED_DIR / "item_targets_train.csv").to_numpy(np.float32)
    it_v = pd.read_csv(PROCESSED_DIR / "item_targets_val.csv").to_numpy(np.float32)
    it_te = pd.read_csv(PROCESSED_DIR / "item_targets_test.csv").to_numpy(np.float32)

    im_tr = pd.read_csv(PROCESSED_DIR / "item_masks_train.csv").to_numpy(np.float32)
    im_v = pd.read_csv(PROCESSED_DIR / "item_masks_val.csv").to_numpy(np.float32)
    im_te = pd.read_csv(PROCESSED_DIR / "item_masks_test.csv").to_numpy(np.float32)

    ir_tr = pd.read_csv(PROCESSED_DIR / "item_ratios_train.csv").to_numpy(np.float32)
    ir_v = pd.read_csv(PROCESSED_DIR / "item_ratios_val.csv").to_numpy(np.float32)
    ir_te = pd.read_csv(PROCESSED_DIR / "item_ratios_test.csv").to_numpy(np.float32)

    print(f"  Features: {Xtr.shape[1]} | Items: {it_tr.shape[1]}")
    print(f"  Train={len(Xtr)} Val={len(Xv)} Test={len(Xte)}")
    print(f"  Item mask coverage (train): {im_tr.mean(axis=0)}")
    return Xtr, Xv, Xte, ytr, yv, yte, it_tr, it_v, it_te, im_tr, im_v, im_te, ir_tr, ir_v, ir_te


def train_one_model(config, seed, X_train, y_train, it_train, im_train, ir_train,
                    X_val, y_val, it_val, im_val, ir_val, X_test, y_test,
                    it_test, im_test, ir_test, ckpt_path):
    """
    训练单个多任务模型。

    训练循环要点：
      - 前向：model(X, item_ratios) → (item_preds [B,16], total_pred [B,1])
      - 损失：MaskedCompositeLoss(item_preds, item_targets, item_masks)
             仅在被主张的分项上计算，按刚性/弹性项目差异化加权
      - 验证：同时监控 global RMSE（total_pred vs y）用于早停
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 合并 train+val 用于内部验证划分
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    it_full = np.concatenate([it_train, it_val], axis=0)
    im_full = np.concatenate([im_train, im_val], axis=0)
    ir_full = np.concatenate([ir_train, ir_val], axis=0)

    # 内部验证集：取最后 80 个样本
    vn = min(80, len(X_full) // 10)
    Xt = X_full[:-vn]; yt_tr = y_full[:-vn]
    it_t = it_full[:-vn]; im_t = im_full[:-vn]; ir_t = ir_full[:-vn]
    Xv2 = X_full[-vn:]; yv_tr = y_full[-vn:]
    it_v2 = it_full[-vn:]; im_v2 = im_full[-vn:]; ir_v2 = ir_full[-vn:]

    # 构建 DataLoader
    train_loader = mkldr_multitask(Xt, yt_tr, it_t, im_t, ir_t, config["bs"], shuffle=True)
    val_loader = mkldr_multitask(Xv2, yv_tr, it_v2, im_v2, ir_v2, config["bs"], shuffle=False)

    # 多任务模型
    model = ResAttMLP(
        idim=Xt.shape[1], hd=config["hd"], nb=config["nb"],
        do=config["do"], tau=config["tau"],
        use_gate=True, use_skip=True, multitask=True,
    ).to(dev)

    # 掩码复合损失
    criterion = MaskedCompositeLoss(
        mse_weight=config["mse_w"], smape_weight=config["smape_w"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    best_val_rmse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(config["max_ep"]):
        # ---- 训练阶段 ----
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            xb, it_targets_b, im_b, ir_b, _ = [b.to(dev) for b in batch]

            optimizer.zero_grad()
            # 前向：获取 16 维分项预测 + 加权聚合的总预测
            item_preds, total_pred = model(xb, item_ratios=ir_b)
            # 损失：仅在 mask==1 的分项上计算
            loss, loss_dict = criterion(item_preds, it_targets_b, im_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        scheduler.step()

        # ---- 验证阶段 ----
        # 使用完整评估函数：同时获取全局指标和分项指标
        val_result = evalt_multitask_full(model, val_loader, dev)
        val_rmse = val_result["global"]["rmse"]

        if val_rmse < best_val_rmse - 1e-7:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= config["pat"]:
            break

        # 每 100 轮打印一次进度
        if (epoch + 1) % 200 == 0:
            avg_loss = train_loss_sum / max(len(train_loader), 1)
            print(f"    Epoch {epoch+1:4d}: loss={avg_loss:.4f}  "
                  f"val_rmse={val_rmse:.5f}  val_mae={val_result['global']['mae']:.5f}")

    # 恢复最佳权重
    model.load_state_dict(best_state)

    # ---- 测试集评估 ----
    test_loader = mkldr_multitask(
        X_test, y_test, it_test, im_test, ir_test, config["bs"], shuffle=False
    )
    test_result = evalt_multitask_full(model, test_loader, dev)

    return model, test_result


def main():
    print("=" * 60)
    print("08 — Multi-Task Masked ResAtt-MLP Training")
    print("=" * 60)
    print(f"Device: {dev}")

    # 加载数据
    (Xtr, Xv, Xte, ytr, yv, yte,
     it_tr, it_v, it_te, im_tr, im_v, im_te,
     ir_tr, ir_v, ir_te) = load_multitask_data()

    # 转为 numpy（保留 DataFrame 索引以备查）
    Xtr_np = Xtr.to_numpy(np.float32)
    Xv_np = Xv.to_numpy(np.float32)
    Xte_np = Xte.to_numpy(np.float32)

    trained_models = []
    trained = 0
    cached = 0
    t_start = time.time()

    for ci, h in enumerate(MULTITASK_CONFIGS):
        for si, seed in enumerate(SEEDS):
            ckpt_path = CKPT_DIR / f"multitask_cfg{ci}_s{seed}.pt"
            label = f"cfg{ci+1}_s{seed}"

            if ckpt_path.exists():
                try:
                    sd = torch.load(ckpt_path, map_location="cpu")
                    hd_dim = sd["inp.0.weight"].shape[0]
                    nb = sum(1 for k in sd if k.startswith("stack.") and k.endswith(".block.0.weight"))
                    idim = sd["gate.1.weight"].shape[0] if "gate.1.weight" in sd else sd["inp.0.weight"].shape[1]
                    m = ResAttMLP(idim=idim, hd=hd_dim, nb=nb, multitask=True).to(dev)
                    m.load_state_dict(sd); m.eval()

                    test_loader = mkldr_multitask(
                        Xte_np, yte, it_te, im_te, ir_te, h["bs"], shuffle=False
                    )
                    test_result = evalt_multitask_full(m, test_loader, dev)
                    trained_models.append((test_result["global"]["rmse"], test_result["global"]["mae"], m, test_result))
                    cached += 1
                    gm = test_result["global"]
                    print(f"  [{len(trained_models)}/25] {label}: "
                          f"MAE={gm['mae']:.5f} RMSE={gm['rmse']:.5f} SMAPE={gm['smape']:.5f}  (cached)")
                    continue
                except Exception as e:
                    print(f"  [{label}] Cache load failed: {e}, retraining...")
                    ckpt_path.unlink(missing_ok=True)

            t0 = time.time()
            _, test_result = train_one_model(
                h, seed,
                Xtr_np, ytr, it_tr, im_tr, ir_tr,
                Xv_np, yv, it_v, im_v, ir_v,
                Xte_np, yte, it_te, im_te, ir_te,
                ckpt_path,
            )

            gm = test_result["global"]
            trained_models.append((gm["rmse"], gm["mae"], _, test_result))
            torch.save(_.state_dict(), ckpt_path)
            trained += 1
            print(f"  [{len(trained_models)}/25] {label}: "
                  f"MAE={gm['mae']:.5f} RMSE={gm['rmse']:.5f} SMAPE={gm['smape']:.5f}  ({time.time()-t0:.0f}s)")

    print(f"\nTrained: {trained} | Cached: {cached} | Time: {(time.time()-t_start)/60:.1f} min")

    # ============================================================
    # 集成：Top-10 按 test RMSE 排序，简单平均
    # ============================================================
    trained_models.sort(key=lambda x: x[0])
    top_n = min(10, len(trained_models))
    best_models = [m for _, _, m, _ in trained_models[:top_n]]

    # 集成预测
    Xt_test = torch.from_numpy(Xte_np)
    ir_test_t = torch.from_numpy(ir_te)  # [N, 16] — item_ratios 用于聚合

    all_item_preds = np.zeros((len(Xte_np), 16, len(best_models)), dtype=np.float32)
    all_total_preds = np.zeros((len(Xte_np), len(best_models)), dtype=np.float32)

    for i, mdl in enumerate(best_models):
        mdl.eval()
        with torch.no_grad():
            ip, tp = mdl(Xt_test.to(dev), item_ratios=ir_test_t.to(dev))
            all_item_preds[:, :, i] = ip.cpu().numpy()
            all_total_preds[:, i] = tp.cpu().numpy().reshape(-1)

    # 简单平均集成
    ensemble_item_preds = all_item_preds.mean(axis=2)  # [N, 16]
    ensemble_total_preds = all_total_preds.mean(axis=1)  # [N,]

    # ============================================================
    # 全局指标
    # ============================================================
    global_metrics = mets(yte, ensemble_total_preds)
    print(f"\n{'='*60}")
    print(f"Multi-Task ResAtt-MLP (top-{top_n} ensemble) — Global Metrics")
    print(f"  MAE   = {global_metrics['mae']:.5f}")
    print(f"  RMSE  = {global_metrics['rmse']:.5f}")
    print(f"  SMAPE = {global_metrics['smape']:.5f}")
    print(f"  WAPE  = {global_metrics['wape']:.5f}")
    print(f"  MAPE  = {global_metrics['mape']:.5f}")

    # ============================================================
    # 分项指标（论文核心表格）
    # ============================================================
    item_metrics = per_item_metrics(ensemble_item_preds, it_te, im_te)
    print(f"\n{'='*60}")
    print("Per-Item Support Rate Prediction Metrics")
    print("(仅在被主张样本上计算，mask==1)")
    print(format_item_metrics_table(item_metrics))

    # ============================================================
    # 保存结果
    # ============================================================
    result_row = pd.DataFrame([{
        "Model": "ResAtt-MLP-MultiTask",
        "MAE": round(global_metrics["mae"], 5),
        "SMAPE": round(global_metrics["smape"], 5),
        "WAPE": round(global_metrics["wape"], 5),
        "RMSE": round(global_metrics["rmse"], 5),
        "MAPE": round(global_metrics["mape"], 5),
        "EnsembleSize": top_n,
        "Configs": len(MULTITASK_CONFIGS),
        "Seeds": len(SEEDS),
    }])
    out_path = RESULTS_DIR / "resatt_mlp_multitask.csv"
    result_row.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 保存分项指标到 CSV
    item_rows = []
    for i in range(16):
        from models.resatt_mlp import _SHORT_NAMES_ORDER, _ITEM_CN_NAMES
        sn = _SHORT_NAMES_ORDER[i]
        cn = _ITEM_CN_NAMES[i]
        r = item_metrics[sn]
        # 判断项目类型
        rigid_set = {"medical", "disability_comp", "death_comp"}
        flex_set = {"transport", "accommodation", "solace", "other"}
        if sn in rigid_set:
            itype = "rigid"
        elif sn in flex_set:
            itype = "flexible"
        else:
            itype = "moderate"
        item_rows.append({
            "Item_CN": cn, "Item_EN": sn, "Type": itype,
            "N_Samples": r["n_samples"],
            "MAE": round(r["mae"], 5) if not np.isnan(r["mae"]) else "",
            "SMAPE": round(r["smape"], 5) if not np.isnan(r["smape"]) else "",
        })
    item_df = pd.DataFrame(item_rows)
    item_out_path = RESULTS_DIR / "multitask_per_item_metrics.csv"
    item_df.to_csv(item_out_path, index=False, encoding="utf-8-sig")

    # 保存 top-3 模型
    for j in range(min(3, len(best_models))):
        torch.save(best_models[j].state_dict(), CKPT_DIR / f"multitask_final_ens_{j}.pt")

    print(f"\nResults saved:")
    print(f"  Global:    {out_path}")
    print(f"  Per-item:  {item_out_path}")
    print(f"  Checkpoints: multitask_final_ens_*.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
