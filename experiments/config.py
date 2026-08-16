"""
Clean experiments — shared configuration.
All scripts import from here to ensure consistent seeds, paths, and hyperparameters.
"""
from pathlib import Path

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Data sources — de-identified release dataset (single file, PII removed)
SRC1 = PROJECT_ROOT / "data" / "sample_data.xlsx"
SRC2 = PROJECT_ROOT / "data" / "sample_data.xlsx"

# Outputs
PROCESSED_DIR = BASE_DIR / "processed_data"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "checkpoints"
FIG_DIR = BASE_DIR / "figures"

for d in [PROCESSED_DIR, RESULTS_DIR, CKPT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model imports
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Random seeds
# ============================================================
RANDOM_STATE = 42
MAIN_SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds as reported in paper

# ============================================================
# Compensation items (16 items from the original preprocessing)
# ============================================================
CLAIM_ITEMS = [
    "医疗费", "后续治疗费", "住院伙食补助费", "营养费", "护理费",
    "误工费/停运损失", "交通费", "住宿费",
    "残疾赔偿金(含被扶养人生活费)", "残疾辅助器具费",
    "死亡赔偿金", "丧葬费", "精神损害抚慰金",
    "财产损失", "鉴定/评估费", "其他费用",
]

SHORT_NAMES = {
    "医疗费": "medical", "后续治疗费": "followup_treatment",
    "住院伙食补助费": "meal_subsidy", "营养费": "nutrition",
    "护理费": "nursing", "误工费/停运损失": "lost_wage",
    "交通费": "transport", "住宿费": "accommodation",
    "残疾赔偿金(含被扶养人生活费)": "disability_comp",
    "残疾辅助器具费": "disability_device",
    "死亡赔偿金": "death_comp", "丧葬费": "funeral",
    "精神损害抚慰金": "solace", "财产损失": "property_loss",
    "鉴定/评估费": "appraisal", "其他费用": "other",
}

# Key items for log transforms
LOG_ITEMS = ["医疗费", "护理费", "误工费/停运损失",
             "残疾赔偿金(含被扶养人生活费)", "精神损害抚慰金",
             "交通费", "财产损失"]

# ============================================================
# ResAtt-MLP main experiment configs (from 02_train_resatt_mlp.py)
# ============================================================
MAIN_CONFIGS = [
    {"hd": 128, "nb": 3, "do": 0.08, "tau": 1.5, "lr": 8e-4,
     "bs": 64, "max_ep": 1200, "pat": 150, "alpha": 0.75, "wd": 5e-6},
    {"hd": 128, "nb": 3, "do": 0.10, "tau": 1.3, "lr": 1e-3,
     "bs": 64, "max_ep": 1200, "pat": 150, "alpha": 0.70, "wd": 3e-6},
    {"hd": 128, "nb": 3, "do": 0.12, "tau": 1.0, "lr": 1.2e-3,
     "bs": 64, "max_ep": 1200, "pat": 150, "alpha": 0.65, "wd": 1e-5},
    {"hd": 128, "nb": 4, "do": 0.10, "tau": 1.3, "lr": 5e-4,
     "bs": 64, "max_ep": 1500, "pat": 180, "alpha": 0.70, "wd": 5e-5},
    {"hd": 140, "nb": 3, "do": 0.10, "tau": 1.4, "lr": 8e-4,
     "bs": 56, "max_ep": 1200, "pat": 150, "alpha": 0.72, "wd": 3e-6},
]

# ============================================================
# Ablation variants
# ============================================================
ABLATION_VARIANTS = [
    ("full",        True,  True,  True),
    ("wo_gate",     False, True,  True),
    ("wo_residual", True,  False, True),
    ("wo_smape",    True,  True,  False),
]

# ============================================================
# Baselines
# ============================================================
BASELINES = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "MLP", "TabNet"]
