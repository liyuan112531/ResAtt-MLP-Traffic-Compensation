from __future__ import annotations

"""损失函数导出模块。"""

from models.resatt_mlp import MSESMAPE_Loss, Huber_SMAPE_Loss, MaskedCompositeLoss

# 论文命名别名，便于脚本兼容。
HuberMAPELoss = Huber_SMAPE_Loss

__all__ = ["MSESMAPE_Loss", "Huber_SMAPE_Loss", "HuberMAPELoss", "MaskedCompositeLoss"]
