"""
Shared ResAtt-MLP model definition and training utilities.
Single source of truth — imported by all experiment scripts.

v2.0 新增：
  - MultiTaskDataset：多任务数据集，产出 (X, item_targets, item_masks, item_ratios, y)
  - ResAttMLP 多任务模式：输出 16 维分项预测 + 诉请占比加权聚合
  - MaskedCompositeLoss：差异化掩码损失（刚性/弹性项目不同权重）
  - 分项评估函数：逐项计算 MAE / SMAPE，输出格式化表格
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 16 个赔偿分项定义（与 config.py 保持一致）
# ============================================================
_CLAIM_ITEMS_ORDER = [
    "医疗费", "后续治疗费", "住院伙食补助费", "营养费", "护理费",
    "误工费/停运损失", "交通费", "住宿费",
    "残疾赔偿金(含被扶养人生活费)", "残疾辅助器具费",
    "死亡赔偿金", "丧葬费", "精神损害抚慰金",
    "财产损失", "鉴定/评估费", "其他费用",
]

_SHORT_NAMES_ORDER = [
    "medical", "followup_treatment", "meal_subsidy", "nutrition", "nursing",
    "lost_wage", "transport", "accommodation",
    "disability_comp", "disability_device",
    "death_comp", "funeral", "solace",
    "property_loss", "appraisal", "other",
]

# 分项中文简称（用于表格打印）
_ITEM_CN_NAMES = [
    "医疗费", "后续治疗费", "住院伙食", "营养费", "护理费",
    "误工/停运", "交通费", "住宿费",
    "残疾赔偿", "残疾辅具",
    "死亡赔偿", "丧葬费", "精神抚慰",
    "财产损失", "鉴定评估", "其他费用",
]

# ============================================================
# ResAtt-MLP Model（新增多任务模式）
# ============================================================

class ResAttMLP(nn.Module):
    """
    Residual Attention MLP with feature gating, stacked residual blocks.

    支持两种模式：
      - 单任务模式（默认）：输出标量 total_pred [B, 1]，向后兼容
      - 多任务模式（multitask=True）：输出 16 维 item_preds [B, 16]
        并在 forward 中通过 item_ratios 加权聚合得到 total_pred

    Parameters
    ----------
    idim : int        Input feature dimension
    hd : int          Hidden dimension
    nb : int          Number of residual blocks
    do : float        Dropout rate
    tau : float       Temperature for gate sigmoid (higher = softer)
    use_gate : bool   Enable feature attention gate
    use_skip : bool   Enable skip connections in residual blocks
    multitask : bool  启用多任务输出（16 维分项预测 + 加权聚合）
    """

    class _ResBlock(nn.Module):
        """Residual block with skip connection."""
        def __init__(self, hd, do):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(hd, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(do)
            )

        def forward(self, x):
            return x + self.block(x)

    class _NoSkipBlock(nn.Module):
        """Plain block without skip connection (for ablation)."""
        def __init__(self, hd, do):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(hd, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(do)
            )

        def forward(self, x):
            return self.block(x)

    def __init__(self, idim, hd=256, nb=3, do=0.1, tau=1.0,
                 use_gate=True, use_skip=True, multitask=False):
        super().__init__()
        self.tau = tau
        self.use_gate = use_gate
        self.multitask = multitask
        self.output_dim = 16 if multitask else 1

        # ---- 特征注意力门控（Feature Attention Gate）----
        # 原理：对每个输入特征学习一个 [0,1] 的软掩码，通过 (0.5+mask) 实现
        # 有偏门控——最低保留 50% 原始信号，避免关键特征被完全抑制
        if use_gate:
            self.gate = nn.Sequential(
                nn.LayerNorm(idim),
                nn.Linear(idim, idim),
            )

        # ---- 输入标准化投影（Input Standardization + Projection）----
        self.inp = nn.Sequential(
            nn.Linear(idim, hd),
            nn.BatchNorm1d(hd),
            nn.GELU(),
            nn.Dropout(do),
        )

        # ---- 堆叠残差块（Stacked Residual Blocks）----
        Block = self._ResBlock if use_skip else self._NoSkipBlock
        self.stack = nn.ModuleList([Block(hd, do) for _ in range(nb)])

        # ---- 输出层 ----
        # 多任务模式：16 维输出，每个维度对应一个赔偿分项的支持率预测
        # 后接 Sigmoid 将输出压缩到 [0, 1]
        self.out = nn.Linear(hd, self.output_dim)

    def forward(self, x, item_ratios=None, return_mask=False):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [B, idim]           输入特征
        item_ratios : Tensor [B, 16]   各分项诉请占比（仅多任务模式需要）
        return_mask : bool             是否返回特征门控掩码

        Returns
        -------
        单任务模式：y [B, 1] 或 (y, mask)
        多任务模式：(item_preds [B, 16], total_pred [B, 1]) 或三元组含 mask
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            if item_ratios is not None and item_ratios.dim() == 1:
                item_ratios = item_ratios.unsqueeze(0)

        mask = None
        if self.use_gate:
            mask = torch.sigmoid(self.gate(x) / self.tau)
            # 有偏门控：特征被缩放至 [0.5, 1.5] 倍，而非 [0, 1]
            # 动机——即便门控输出为 0，原始信号仍保留一半强度
            x = x * (0.5 + mask)

        h = self.inp(x)
        for block in self.stack:
            h = block(h)

        if self.multitask:
            # ---- 多任务输出分支 ----
            # item_preds: [B, 16] — 16 个赔偿分项的独立支持率预测
            item_preds = torch.sigmoid(self.out(h))

            # ---- 无参数聚合层（Non-parametric Aggregation）----
            # 利用 item_ratios（诉请金额占比）作为先验权重进行加权求和
            # 公式：total_pred = Σ(item_pred_i × ratio_i)
            # 物理含义：总支持率 = 各分项支持率按其诉请金额占比加权
            if item_ratios is not None:
                total_pred = torch.sum(item_preds * item_ratios, dim=1, keepdim=True)
            else:
                # 无 ratio 时的退化方案：等权平均
                total_pred = item_preds.mean(dim=1, keepdim=True)

            return (item_preds, total_pred, mask) if return_mask else (item_preds, total_pred)
        else:
            # ---- 单任务输出分支（向后兼容）----
            y = torch.sigmoid(self.out(h))
            return (y, mask) if return_mask else y


# ============================================================
# 多任务数据集（MultiTaskDataset）
# ============================================================

class MultiTaskDataset(Dataset):
    """
    多任务表格数据集。

    每次迭代产出 5 个张量：
      X              : [n_features]     标准化特征
      item_targets   : [16]            各分项真实支持率 (awarded_i / claim_i)
      item_masks     : [16]            0/1 掩码——该分项是否被原告主张
      item_ratios    : [16]            各分项诉请占比 (claim_i / total_claim)
      y              : [1]             总支持率（仅用于全局验证，不参与训练）

    掩码机制说明：
      交通事故案件中，并非所有 16 项赔偿都会被主张（例如无死亡则不主张丧葬费）。
      item_masks 标记了哪些分项实际存在，损失函数仅在 mask==1 的位置计算梯度，
      避免模型被迫在无意义的维度上拟合。
    """

    def __init__(self, X, y, item_targets, item_masks, item_ratios):
        """
        Parameters
        ----------
        X : np.ndarray [N, D]             标准化特征矩阵
        y : np.ndarray [N,] 或 [N, 1]     总支持率
        item_targets : np.ndarray [N, 16]  各分项真实支持率
        item_masks : np.ndarray [N, 16]    分项掩码（0/1）
        item_ratios : np.ndarray [N, 16]   分项诉请占比
        """
        self.X = torch.from_numpy(np.asarray(X, np.float32).copy())
        self.y = torch.from_numpy(np.asarray(y, np.float32).reshape(-1, 1).copy())
        self.item_targets = torch.from_numpy(np.asarray(item_targets, np.float32).copy())
        self.item_masks = torch.from_numpy(np.asarray(item_masks, np.float32).copy())
        self.item_ratios = torch.from_numpy(np.asarray(item_ratios, np.float32).copy())

        # 校验维度一致性
        N = self.X.shape[0]
        assert self.item_targets.shape == (N, 16), \
            f"item_targets shape mismatch: {self.item_targets.shape} vs ({N}, 16)"
        assert self.item_masks.shape == (N, 16), \
            f"item_masks shape mismatch: {self.item_masks.shape} vs ({N}, 16)"
        assert self.item_ratios.shape == (N, 16), \
            f"item_ratios shape mismatch: {self.item_ratios.shape} vs ({N}, 16)"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],            # [D]
            self.item_targets[idx], # [16]
            self.item_masks[idx],   # [16]
            self.item_ratios[idx],  # [16]
            self.y[idx],            # [1]
        )


# ============================================================
# 差异化掩码复合损失函数（Masked Composite Loss）
# ============================================================

class MaskedCompositeLoss(nn.Module):
    """
    带项目差异化权重的掩码复合损失。

    设计思路：
      1. 掩码机制：仅对 item_masks==1 的分项计算损失，避免无意义维度的梯度污染
      2. 差异化权重：刚性项目（医疗费、死亡/伤残赔偿金等）享有更高权重，
         因为其法律标准明确、判决稳定性高；弹性项目（精神抚慰、交通费等）
         自由裁量空间大，权重适当降低
      3. 复合损失：MSE 关注大误差，SMAPE 关注相对误差，组合兼顾两者

    损失公式：
      L = (1/N_eff) * Σ_i [mask_i * w_i * (0.3*MSE_i + 0.7*SMAPE_i)]
      其中 N_eff = Σ(mask) + ε，即有效掩码总数
    """

    # 16 个分项的差异化权重（索引与 CLAIM_ITEMS 对齐）
    # 刚性项目（法律标准明确、判决稳定）：weight = 1.2
    # 弹性项目（法官自由裁量空间大）：weight = 0.8
    # 其余中等项目：weight = 1.0
    _DEFAULT_ITEM_WEIGHTS = torch.tensor([
        1.2,  # 0  医疗费          — 刚性（凭票实报实销）
        1.0,  # 1  后续治疗费      — 中等（有鉴定意见但金额可调）
        1.0,  # 2  住院伙食补助费  — 中等（有固定标准但天数可争）
        1.0,  # 3  营养费          — 中等
        1.0,  # 4  护理费          — 中等
        1.0,  # 5  误工费/停运损失 — 中等
        0.8,  # 6  交通费          — 弹性（酌定空间大）
        0.8,  # 7  住宿费          — 弹性
        1.2,  # 8  残疾赔偿金      — 刚性（法定公式计算）
        1.0,  # 9  残疾辅助器具费  — 中等
        1.2,  # 10 死亡赔偿金      — 刚性（法定公式计算）
        1.0,  # 11 丧葬费          — 中等（有法定标准）
        0.8,  # 12 精神损害抚慰金  — 弹性（高度酌定）
        1.0,  # 13 财产损失        — 中等
        1.0,  # 14 鉴定/评估费     — 中等
        0.8,  # 15 其他费用        — 弹性
    ], dtype=torch.float32)

    def __init__(self, item_weights=None, mse_weight=0.3, smape_weight=0.7, eps=1e-8):
        """
        Parameters
        ----------
        item_weights : Tensor [16] or None
            各分项权重。若为 None 则使用默认差异化权重
        mse_weight : float  MSE 在复合损失中的权重（默认 0.3）
        smape_weight : float SMAPE 在复合损失中的权重（默认 0.7）
        eps : float  防除零常数
        """
        super().__init__()
        if item_weights is not None:
            self.register_buffer("item_weights", torch.as_tensor(item_weights, dtype=torch.float32))
        else:
            self.register_buffer("item_weights", self._DEFAULT_ITEM_WEIGHTS.clone())
        self.mse_weight = mse_weight
        self.smape_weight = smape_weight
        self.eps = eps

    def forward(self, item_preds, item_targets, item_masks):
        """
        Parameters
        ----------
        item_preds : Tensor [B, 16]   模型预测的各分项支持率
        item_targets : Tensor [B, 16]  真实分项支持率
        item_masks : Tensor [B, 16]    0/1 掩码

        Returns
        -------
        total_loss : 标量损失
        loss_dict : dict  包含 mse_loss, smape_loss, total_loss 的分离值（用于日志）
        """
        item_preds = item_preds.float()
        item_targets = item_targets.float()
        item_masks = item_masks.float()

        # ---- 掩码机制：仅对有效分项计算损失 ----
        # 有效分项总数 N_eff = Σ(mask)，即 batch 内所有被主张的分项数量
        N_eff = item_masks.sum() + self.eps

        # 将 item_weights 扩展到 batch 维度：[16] → [1, 16]
        w = self.item_weights.to(item_preds.device).view(1, 16)

        # ---- MSE 分支 ----
        # [B, 16] 逐元素 MSE，再与 mask 和 weight 相乘
        mse_per_item = (item_preds - item_targets) ** 2
        masked_mse = (mse_per_item * item_masks * w).sum() / N_eff

        # ---- SMAPE 分支 ----
        # 对称平均绝对百分比误差：2|pred-target| / (|pred|+|target|+ε)
        abs_diff = torch.abs(item_preds - item_targets)
        smape_per_item = 2.0 * abs_diff / (torch.abs(item_targets) + torch.abs(item_preds) + self.eps)
        masked_smape = (smape_per_item * item_masks * w).sum() / N_eff

        # ---- 复合损失：0.3 * MSE + 0.7 * SMAPE ----
        # SMAPE 天然在 [0, 2] 范围，MSE 在 [0, 1] 范围——SMAPE 占主导权重
        total_loss = self.mse_weight * masked_mse + self.smape_weight * masked_smape

        return total_loss, {
            "mse_loss": masked_mse.detach().item(),
            "smape_loss": masked_smape.detach().item(),
            "total_loss": total_loss.detach().item(),
            "N_eff": int(N_eff.item()),
        }


# ============================================================
# 原有损失函数（向后兼容，单任务模式使用）
# ============================================================

class MSESMAPE_Loss(nn.Module):
    """MSE + SMAPE combined loss for single-task training.
    alpha controls the MSE weight."""

    def __init__(self, alpha=0.7, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        mse_loss = self.mse(pred, target)
        smape_loss = (2.0 * torch.abs(pred - target) /
                      (torch.abs(target) + torch.abs(pred) + self.eps))
        return (self.alpha * mse_loss + (1.0 - self.alpha) * smape_loss).mean()


# 向后兼容别名
Huber_SMAPE_Loss = MSESMAPE_Loss


# ============================================================
# 全局指标（向后兼容）
# ============================================================

def mets(yt, yp, eps=1e-8):
    """Compute MAE, RMSE, SMAPE, WAPE, MAPE for global predictions."""
    yt = np.asarray(yt, float).reshape(-1)
    yp = np.clip(np.asarray(yp, float).reshape(-1), 0, 1)
    ae = np.abs(yp - yt)
    at = np.abs(yt)
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mape": float(np.mean(ae / np.clip(at, eps, None))),
        "smape": float(np.mean(2 * ae / np.clip(at + np.abs(yp), eps, None))),
        "wape": float(ae.sum() / max(at.sum(), eps)),
    }


# ============================================================
# 分项级别指标（Per-Item Metrics）
# ============================================================

def per_item_metrics(item_preds, item_targets, item_masks, eps=1e-8):
    """
    计算每个分项（1~16）独立的 MAE 和 SMAPE（仅在 mask==1 的样本上计算）。

    这是论文的核心评估函数——用于对比刚性项目和弹性项目的预测精度差异。

    Parameters
    ----------
    item_preds : np.ndarray [N, 16]    模型预测的各分项支持率
    item_targets : np.ndarray [N, 16]   真实分项支持率
    item_masks : np.ndarray [N, 16]     掩码（0/1）

    Returns
    -------
    metrics : dict
        key: item_name_en, value: {"mae": float, "smape": float, "n_samples": int}
    """
    item_preds = np.clip(np.asarray(item_preds, float), 0, 1)
    item_targets = np.asarray(item_targets, float)
    item_masks = np.asarray(item_masks, float)

    results = {}
    for i in range(16):
        sn = _SHORT_NAMES_ORDER[i]
        m = item_masks[:, i] > 0.5  # 布尔掩码：该分项被主张的样本
        n_valid = int(m.sum())

        if n_valid == 0:
            results[sn] = {"mae": float("nan"), "smape": float("nan"), "n_samples": 0}
            continue

        p = item_preds[m, i]
        t = item_targets[m, i]
        ae = np.abs(p - t)

        mae_i = float(np.mean(ae))
        smape_i = float(np.mean(2.0 * ae / np.clip(np.abs(t) + np.abs(p), eps, None)))
        results[sn] = {"mae": mae_i, "smape": smape_i, "n_samples": n_valid}

    return results


def format_item_metrics_table(per_item_results):
    """
    格式化打印分项指标表格，便于论文汇报。

    输出格式：
    ┌──────────────────┬────────┬────────┬────────┬──────────┐
    │ 分项名称         │   N    │  MAE   │ SMAPE  │ 类型     │
    ├──────────────────┼────────┼────────┼────────┼──────────┤
    │ 医疗费 (刚性)    │  120   │ 0.1234 │ 0.2345 │ 刚性     │
    │ 交通费 (弹性)    │  110   │ 0.3456 │ 0.5678 │ 弹性     │
    │ ...              │        │        │        │          │
    └──────────────────┴────────┴────────┴────────┴──────────┘
    │ 刚性项目均值     │   --   │ 0.xxxx │ 0.xxxx │          │
    │ 弹性项目均值     │   --   │ 0.xxxx │ 0.xxxx │          │
    └──────────────────┴────────┴────────┴────────┴──────────┘

    Parameters
    ----------
    per_item_results : dict   per_item_metrics() 的返回值

    Returns
    -------
    table_str : str  格式化后的表格字符串
    """
    # 项目分类
    rigid_items = ["medical", "disability_comp", "death_comp"]     # 刚性项目
    flexible_items = ["transport", "accommodation", "solace", "other"]  # 弹性项目

    def item_type(sn):
        if sn in rigid_items:
            return "刚性"
        elif sn in flexible_items:
            return "弹性"
        return "中等"

    # 表头
    header = f"{'分项名称':<18s} {'N':>6s} {'MAE':>8s} {'SMAPE':>8s} {'类型':>6s}"
    sep = "-" * 52

    lines = [sep, header, sep]

    rigid_mae, rigid_smape = [], []
    flex_mae, flex_smape = [], []

    for i in range(16):
        sn = _SHORT_NAMES_ORDER[i]
        cn = _ITEM_CN_NAMES[i]
        r = per_item_results.get(sn, {"mae": float("nan"), "smape": float("nan"), "n_samples": 0})
        it = item_type(sn)

        mae_str = f"{r['mae']:.4f}" if not np.isnan(r['mae']) else "   N/A"
        smape_str = f"{r['smape']:.4f}" if not np.isnan(r['smape']) else "   N/A"

        lines.append(f"{cn + ' (' + it + ')':<16s} {r['n_samples']:>6d} {mae_str:>8s} {smape_str:>8s} {it:>6s}")

        if it == "刚性" and not np.isnan(r['mae']):
            rigid_mae.append(r['mae'])
            rigid_smape.append(r['smape'])
        elif it == "弹性" and not np.isnan(r['mae']):
            flex_mae.append(r['mae'])
            flex_smape.append(r['smape'])

    lines.append(sep)

    # 分组汇总
    if rigid_mae:
        r_mae_mean = np.mean(rigid_mae)
        r_smape_mean = np.mean(rigid_smape)
        lines.append(f"{'刚性项目均值':<18s} {'--':>6s} {r_mae_mean:>8.4f} {r_smape_mean:>8.4f} {'':>6s}")
    if flex_mae:
        f_mae_mean = np.mean(flex_mae)
        f_smape_mean = np.mean(flex_smape)
        lines.append(f"{'弹性项目均值':<18s} {'--':>6s} {f_mae_mean:>8.4f} {f_smape_mean:>8.4f} {'':>6s}")
    lines.append(sep)

    return "\n".join(lines)


def print_item_metrics(per_item_results):
    """便捷函数：直接 print 分项指标表格。"""
    print(format_item_metrics_table(per_item_results))


# ============================================================
# DataLoader 工具函数
# ============================================================

def mkldr(x, y, bs, shuffle):
    """
    构建标准（单任务）DataLoader。
    向后兼容——所有现有实验脚本均使用此函数。
    """
    x = np.asarray(x, np.float32).copy()
    y = np.asarray(y, np.float32).reshape(-1, 1).copy()
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=bs, shuffle=shuffle,
    )


def mkldr_multitask(X, y, item_targets, item_masks, item_ratios, bs, shuffle):
    """
    构建多任务 DataLoader。

    每次迭代产出 (X_batch, item_targets, item_masks, item_ratios, y_batch)。

    Parameters
    ----------
    X : np.ndarray [N, D]
    y : np.ndarray [N,] or [N, 1]
    item_targets : np.ndarray [N, 16]
    item_masks : np.ndarray [N, 16]
    item_ratios : np.ndarray [N, 16]
    bs : int   batch size
    shuffle : bool

    Returns
    -------
    DataLoader  每次 yield 5 个 Tensor
    """
    ds = MultiTaskDataset(X, y, item_targets, item_masks, item_ratios)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


# ============================================================
# 评估工具函数
# ============================================================

def evalt(model, loader, device):
    """
    评估单任务模型（向后兼容）。
    返回全局 mets 指标字典。
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy().reshape(-1))
            targets.append(yb.numpy().reshape(-1))
    return mets(np.concatenate(targets), np.concatenate(preds))


def evalt_multitask(model, loader, device):
    """
    评估多任务模型。

    返回：
      - global_metrics: dict  全局 total_pred vs y 的 MAE/SMAPE 等
      - item_metrics: dict    per_item_metrics() 返回值
      - all_item_preds: np.ndarray [N, 16]  全部 item_preds
      - all_total_preds: np.ndarray [N,]    全部 total_pred
      - all_targets: np.ndarray [N,]        全部真实 y
    """
    model.eval()
    all_total_preds = []
    all_targets = []
    all_item_preds = []
    all_item_masks = []

    with torch.no_grad():
        for batch in loader:
            xb, _, item_masks_b, item_ratios_b, yb = [b.to(device) for b in batch]
            item_preds, total_pred = model(xb, item_ratios=item_ratios_b)

            all_total_preds.append(total_pred.cpu().numpy().reshape(-1))
            all_targets.append(yb.cpu().numpy().reshape(-1))
            all_item_preds.append(item_preds.cpu().numpy())
            all_item_masks.append(item_masks_b.cpu().numpy())

    total_preds = np.concatenate(all_total_preds)
    targets = np.concatenate(all_targets)
    item_preds_all = np.concatenate(all_item_preds, axis=0)
    item_masks_all = np.concatenate(all_item_masks, axis=0)

    global_metrics = mets(targets, total_preds)

    # 还需要真实分项目标来计算分项指标
    # loader 中包含了 item_targets，需要额外收集
    return global_metrics, item_preds_all, item_masks_all, total_preds, targets


def evalt_multitask_full(model, loader, device):
    """
    完整多任务评估：同时收集 item_targets 以计算分项指标。

    返回 global_metrics + per_item_metrics dict。
    """
    model.eval()
    all_total_preds = []
    all_targets = []
    all_item_preds = []
    all_item_targets = []
    all_item_masks = []

    with torch.no_grad():
        for batch in loader:
            xb, it_targets_b, item_masks_b, item_ratios_b, yb = [b.to(device) for b in batch]
            item_preds, total_pred = model(xb, item_ratios=item_ratios_b)

            all_total_preds.append(total_pred.cpu().numpy().reshape(-1))
            all_targets.append(yb.cpu().numpy().reshape(-1))
            all_item_preds.append(item_preds.cpu().numpy())
            all_item_targets.append(it_targets_b.cpu().numpy())
            all_item_masks.append(item_masks_b.cpu().numpy())

    total_preds = np.concatenate(all_total_preds)
    targets = np.concatenate(all_targets)
    item_preds_all = np.concatenate(all_item_preds, axis=0)
    item_targets_all = np.concatenate(all_item_targets, axis=0)
    item_masks_all = np.concatenate(all_item_masks, axis=0)

    global_metrics = mets(targets, total_preds)
    item_metrics = per_item_metrics(item_preds_all, item_targets_all, item_masks_all)

    return {
        "global": global_metrics,
        "per_item": item_metrics,
        "item_preds": item_preds_all,
        "total_preds": total_preds,
        "y_true": targets,
        "item_targets": item_targets_all,
        "item_masks": item_masks_all,
    }
