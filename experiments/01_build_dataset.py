"""
01_build_dataset.py — Build clean, leakage-free dataset.

Merges compensation item data with legal context features.
Removes 19 leaked features (item-level award/claim rates).
Outputs standardized 70/15/15 train/val/test splits.
"""
import json, re, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from config import (
    SRC1, SRC2, PROCESSED_DIR, RANDOM_STATE,
    CLAIM_ITEMS, SHORT_NAMES, LOG_ITEMS,
)

# ============================================================
# Legal feature parsers (from 01_feature_pipeline.py)
# ============================================================
CN_NUM = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
          "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def parse_liability_ratio(text):
    """Parse liability ratio to [0,1]."""
    if pd.isna(text): return np.nan
    s = str(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        v = float(m.group(1))
        return v / 100.0 if v > 1 else v
    m2 = re.search(r"百分之\s*(\d+(?:\.\d+)?)", s)
    if m2: return float(m2.group(1)) / 100.0
    if "全责" in s: return 1.0
    if "主责" in s: return 0.7
    if "同责" in s: return 0.5
    if "次责" in s: return 0.3
    if "无责" in s: return 0.0
    return np.nan


def parse_injury_grade(v):
    """Parse injury grade as numeric level (1=most severe, 10=least)."""
    if pd.isna(v): return 0.0
    s = str(v).strip()
    if s == "": return 0.0
    m = re.search(r"(\d+)", s)
    if m: return float(m.group(1))
    for k, n in CN_NUM.items():
        if k in s and "级" in s: return float(n)
    return 0.0


def derive_accident_type(text):
    """Classify accident type from text."""
    s = "" if pd.isna(text) else str(text)
    if "行人" in s: return "motor_vs_pedestrian"
    if "非机动车" in s: return "motor_vs_nonmotor"
    if "机动车" in s: return "motor_vs_motor"
    return "unknown"


def derive_insurance_flags(text):
    """Extract insurance coverage indicators."""
    s = "" if pd.isna(text) else str(text)
    has_c = 1 if ("交强险" in s or "强制险" in s) else 0
    has_com = 1 if any(kw in s for kw in ["商业三者", "财产保险",
                        "保险股份有限公司", "保险公司"]) else 0
    if has_c and has_com: coverage = "both"
    elif has_c: coverage = "compulsory_only"
    elif has_com: coverage = "commercial_only"
    else: coverage = "none_or_unknown"
    return has_c, has_com, coverage


def to_float(v):
    """Robust float conversion."""
    if pd.isna(v): return np.nan
    s = str(v).strip().replace(",", "")
    if s == "": return np.nan
    try: return float(s)
    except Exception: return np.nan


def main():
    print("=" * 60)
    print("01 — Building Clean Dataset (no target leakage)")
    print("=" * 60)

    # ============================================================
    # Step 1: Load compensation items (src1)
    # ============================================================
    src1 = pd.read_excel(SRC1, sheet_name="判决书赔偿项目汇总")
    print(f"\nSrc1 raw rows: {len(src1)}")

    # Filter normal samples (col -1 is 数据状态)
    status_col = src1.columns[-1]
    normal = src1[src1[status_col].astype(str).str.contains("正常")].copy()
    print(f"Normal samples: {len(normal)}")

    # ============================================================
    # Step 2: Extract item-level features from src1
    # ============================================================
    feat = {}

    for item in CLAIM_ITEMS:
        sn = SHORT_NAMES[item]
        c_col = f"{item}_原告主张"
        a_col = f"{item}_法院判决"

        claim_vals = pd.to_numeric(normal[c_col], errors="coerce").fillna(0).values
        award_vals = pd.to_numeric(normal[a_col], errors="coerce").fillna(0).values

        # Pre-judgment features (NO leakage)
        feat[f"{sn}_has"] = (claim_vals > 0).astype(float)
        # Raw amounts saved for ratio/log computation, NOT included as features

    # Totals (pre-judgment)
    total_claim = pd.to_numeric(normal["原告主张合计"], errors="coerce").fillna(0).values
    total_award = pd.to_numeric(normal["法院判决合计"], errors="coerce").fillna(0).values

    # Target variable
    support_rate = np.where(total_claim > 0, total_award / total_claim, 0)
    support_rate = np.clip(support_rate, 0, 1)

    print(f"Target: mean={support_rate.mean():.4f} std={support_rate.std():.4f}")

    # ============================================================
    # Step 3: Item ratios and derived features
    # ============================================================
    features_df = pd.DataFrame(feat)

    # Item ratios (claim_i / total_claim, pre-judgment)
    for item in CLAIM_ITEMS:
        sn = SHORT_NAMES[item]
        c_col = f"{item}_原告主张"
        cv = pd.to_numeric(normal[c_col], errors="coerce").fillna(0).values
        features_df[f"{sn}_ratio"] = np.where(total_claim > 0, cv / total_claim, 0)

    # Number of items claimed
    n_items = np.zeros(len(normal))
    for item in CLAIM_ITEMS:
        sn = SHORT_NAMES[item]
        n_items += feat[f"{sn}_has"]
    features_df["n_items_claimed"] = n_items

    # Log transforms
    features_df["log_total_claim"] = np.log1p(total_claim)
    for item in LOG_ITEMS:
        sn = SHORT_NAMES[item]
        cv = pd.to_numeric(normal[f"{item}_原告主张"], errors="coerce").fillna(0).values
        features_df[f"log_{sn}_claim"] = np.log1p(cv)

    # Damage type indicators
    features_df["is_mainly_medical"] = (features_df["medical_ratio"] > 0.4).astype(float)
    features_df["is_mainly_disability"] = (features_df["disability_comp_ratio"] > 0.3).astype(float)
    features_df["is_mainly_lost_wage"] = (features_df["lost_wage_ratio"] > 0.3).astype(float)
    features_df["is_mainly_property"] = (features_df["property_loss_ratio"] > 0.3).astype(float)

    # ============================================================
    # Step 4: Load legal features from src2 (sample_data.xlsx, de-identified)
    # ============================================================
    xls2 = pd.ExcelFile(SRC2)
    liability_df = xls2.parse("原告责任信息")
    injury_df = xls2.parse("伤残等级提取")
    reason_df = xls2.parse("赔偿差距原因")

    # Use "file_name" (English) as the merge key consistently
    fn_col_1 = normal.columns[0]  # Chinese filename column in src1

    # Build legal DataFrame with "file_name" key
    # Start from liability_df which has file_name
    legal = pd.DataFrame()
    legal["file_name"] = liability_df["file_name"].astype(str)

    # Liability ratio
    if "plaintiff_responsibility" in liability_df.columns:
        legal["liability_ratio"] = liability_df["plaintiff_responsibility"].apply(parse_liability_ratio)
    else:
        legal["liability_ratio"] = 0.0

    # Injury grade — injury_df has Chinese filename col as col 0
    inj_fn_col = injury_df.columns[0]
    inj_val_col = None
    for c in injury_df.columns:
        if "标准化" in str(c) or "等级" in str(c):
            inj_val_col = c
            break
    if inj_val_col:
        injury_df = injury_df.copy()
        injury_df["injury_grade"] = injury_df[inj_val_col].apply(parse_injury_grade)
        inj_map = dict(zip(injury_df[inj_fn_col].astype(str), injury_df["injury_grade"]))
        legal["injury_grade"] = legal["file_name"].map(inj_map).fillna(0)

    # Accident type and insurance — pre-computed in the de-identified release
    # (original free-text fields carried litigant names and were collapsed into
    #  these structured columns during de-identification).
    if "file_name" in reason_df.columns and "accident_type" in reason_df.columns:
        for col in ["accident_type", "has_compulsory_insurance",
                     "has_commercial_insurance", "insurance_coverage"]:
            rmap = dict(zip(reason_df["file_name"].astype(str), reason_df[col]))
            legal[col] = legal["file_name"].map(rmap)

    # ============================================================
    # Step 5: Merge legal features with compensation features
    # ============================================================
    features_df["_filename"] = normal[fn_col_1].astype(str)
    legal["_filename"] = legal["file_name"].astype(str)
    merge_cols_legal = [c for c in legal.columns if c not in ["file_name", "_filename"]]
    features_df = features_df.merge(
        legal[["_filename"] + merge_cols_legal], on="_filename", how="left")

    # One-hot encode categoricals
    for cat_col in ["accident_type", "insurance_coverage"]:
        if cat_col in features_df.columns:
            dummies = pd.get_dummies(features_df[cat_col], prefix=cat_col)
            features_df = pd.concat([features_df, dummies], axis=1)
            features_df.drop(columns=[cat_col], inplace=True)

    # Fill missing legal features
    for c in ["liability_ratio", "injury_grade", "has_compulsory_insurance",
              "has_commercial_insurance"]:
        if c in features_df.columns:
            features_df[c] = features_df[c].fillna(0)

    # ============================================================
    # Step 6: Engineered interaction features
    # ============================================================
    features_df["has_injury"] = (features_df["injury_grade"].fillna(0) > 0).astype(float)
    features_df["liability_x_log_claim"] = (
        features_df["liability_ratio"].fillna(0) * features_df["log_total_claim"]
    )
    features_df["liability_x_injury"] = (
        features_df["liability_ratio"].fillna(0) * features_df["injury_grade"].fillna(0)
    )

    # ============================================================
    # Step 7: Select final model features (EXCLUDE leaked features)
    # ============================================================
    # The original pipeline included {sn}_rate, weighted_item_rate, max/min_item_rate
    # These all use post-judgment award amounts → TARGET LEAKAGE → EXCLUDED
    # Explicitly leaked: item-level award/claim RATES and their aggregates
    # (NOT log transforms, NOT ratios, NOT _has flags)
    explicit_leak = set()
    for item in CLAIM_ITEMS:
        sn = SHORT_NAMES[item]
        explicit_leak.add(f"{sn}_rate")  # item-level support rate (uses award)
    explicit_leak.add("weighted_item_rate")
    explicit_leak.add("max_item_rate")
    explicit_leak.add("min_item_rate")

    # Also exclude metadata columns, not features
    exclude_non_feature = ["_filename", "file_name"]

    all_cols = list(features_df.columns)
    model_cols = [c for c in all_cols
                  if c not in explicit_leak and c not in exclude_non_feature]

    X = features_df[model_cols].copy()
    y = support_rate.copy()

    # Verify no leaked features remain
    leaked_check = [c for c in model_cols if
                    c.endswith('_rate') and not c.startswith('log_')]
    if leaked_check:
        print(f"WARNING: Still have leaked features: {leaked_check}")
    else:
        print("VERIFIED: No leaked features in model set")

    print(f"\nFinal features: {len(model_cols)}")
    print(f"Features: {model_cols}")

    # ============================================================
    # Step 7.5: Compute 16-dim item-level targets (awarded_i / claim_i)
    # 用于多任务学习 —— 每个赔偿分项的真实支持率
    # ============================================================
    item_targets = np.zeros((len(normal), len(CLAIM_ITEMS)), dtype=np.float32)
    for i, item in enumerate(CLAIM_ITEMS):
        c_col = f"{item}_原告主张"
        a_col = f"{item}_法院判决"
        cv = pd.to_numeric(normal[c_col], errors="coerce").fillna(0).values
        av = pd.to_numeric(normal[a_col], errors="coerce").fillna(0).values
        # 若诉请金额为0，则目标设为0（该项未被主张，掩码会将其排除）
        # 使用 np.divide 的 where 参数避免 cv==0 时的除零警告
        item_targets[:, i] = np.divide(av, cv, out=np.zeros_like(av, dtype=np.float64), where=(cv > 0))
        # Clip 到 [0, 1] 范围，防止极端异常值
        item_targets[:, i] = np.clip(item_targets[:, i], 0.0, 1.0)

    # item_masks: 16-dim 的 0/1 掩码，标记哪些分项被主张（即 item_has）
    item_masks = np.zeros((len(normal), len(CLAIM_ITEMS)), dtype=np.float32)
    for i, item in enumerate(CLAIM_ITEMS):
        sn = SHORT_NAMES[item]
        item_masks[:, i] = feat[f"{sn}_has"]

    # item_ratios: 16-dim 的诉请占比（claim_i / total_claim），已在 features_df 中
    item_ratios = np.zeros((len(normal), len(CLAIM_ITEMS)), dtype=np.float32)
    for i, item in enumerate(CLAIM_ITEMS):
        sn = SHORT_NAMES[item]
        item_ratios[:, i] = features_df[f"{sn}_ratio"].values

    print(f"Item targets shape: {item_targets.shape}")
    print(f"Item targets (non-zero) means: {np.mean(item_targets, axis=0)}")

    # ============================================================
    # Step 8: Stratified split 70/15/15
    # ============================================================
    try:
        strat_labels = pd.qcut(pd.Series(y), q=5, labels=False, duplicates='drop')
    except ValueError:
        strat_labels = pd.cut(pd.Series(y), bins=5, labels=False)

    # 将 item_targets, item_masks, item_ratios 纳入 split，保持索引一致
    it_df = pd.DataFrame(item_targets)
    im_df = pd.DataFrame(item_masks)
    ir_df = pd.DataFrame(item_ratios)

    X_train, X_temp, y_train, y_temp, s_train, s_temp, \
        it_train, it_temp, im_train, im_temp, ir_train, ir_temp = train_test_split(
        X, y, strat_labels,
        it_df, im_df, ir_df,
        test_size=0.30,
        random_state=RANDOM_STATE, stratify=strat_labels)

    X_val, X_test, y_val, y_test, s_val, s_test, \
        it_val, it_test, im_val, im_test, ir_val, ir_test = train_test_split(
        X_temp, y_temp, s_temp,
        it_temp, im_temp, ir_temp,
        test_size=0.50,
        random_state=RANDOM_STATE, stratify=s_temp)

    print(f"\nSplit: Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")

    # ============================================================
    # Step 9: Standardize
    # ============================================================
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=model_cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=model_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=model_cols)

    # ============================================================
    # Step 10: Save
    # ============================================================
    X_train_s.to_csv(PROCESSED_DIR / "X_train.csv", index=False, encoding="utf-8-sig")
    X_val_s.to_csv(PROCESSED_DIR / "X_val.csv", index=False, encoding="utf-8-sig")
    X_test_s.to_csv(PROCESSED_DIR / "X_test.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"support_rate": y_train}).to_csv(
        PROCESSED_DIR / "y_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"support_rate": y_val}).to_csv(
        PROCESSED_DIR / "y_val.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"support_rate": y_test}).to_csv(
        PROCESSED_DIR / "y_test.csv", index=False, encoding="utf-8-sig")

    # 保存 16 维分项目标、掩码、诉请占比（用于多任务学习）
    # 列顺序与 CLAIM_ITEMS / SHORT_NAMES 一致
    it_cols = [f"{SHORT_NAMES[item]}_target" for item in CLAIM_ITEMS]
    im_cols = [f"{SHORT_NAMES[item]}_mask" for item in CLAIM_ITEMS]
    ir_cols = [f"{SHORT_NAMES[item]}_ratio" for item in CLAIM_ITEMS]

    it_train.to_csv(PROCESSED_DIR / "item_targets_train.csv", index=False, encoding="utf-8-sig", header=it_cols)
    it_val.to_csv(PROCESSED_DIR / "item_targets_val.csv", index=False, encoding="utf-8-sig", header=it_cols)
    it_test.to_csv(PROCESSED_DIR / "item_targets_test.csv", index=False, encoding="utf-8-sig", header=it_cols)

    im_train.to_csv(PROCESSED_DIR / "item_masks_train.csv", index=False, encoding="utf-8-sig", header=im_cols)
    im_val.to_csv(PROCESSED_DIR / "item_masks_val.csv", index=False, encoding="utf-8-sig", header=im_cols)
    im_test.to_csv(PROCESSED_DIR / "item_masks_test.csv", index=False, encoding="utf-8-sig", header=im_cols)

    ir_train.to_csv(PROCESSED_DIR / "item_ratios_train.csv", index=False, encoding="utf-8-sig", header=ir_cols)
    ir_val.to_csv(PROCESSED_DIR / "item_ratios_val.csv", index=False, encoding="utf-8-sig", header=ir_cols)
    ir_test.to_csv(PROCESSED_DIR / "item_ratios_test.csv", index=False, encoding="utf-8-sig", header=ir_cols)

    # Save raw (unscaled) for reference
    X_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(PROCESSED_DIR / "X_test_raw.csv", index=False, encoding="utf-8-sig")

    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    with open(PROCESSED_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(model_cols, f, ensure_ascii=False)

    # Full modeling data
    modeling_df = features_df[model_cols].copy()
    modeling_df["support_rate"] = support_rate
    modeling_df.to_csv(PROCESSED_DIR / "modeling_data.csv", index=False, encoding="utf-8-sig")

    summary = {
        "total_raw": len(src1),
        "normal_samples": len(normal),
        "features": len(model_cols),
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test),
        "target_mean": float(support_rate.mean()),
        "target_std": float(support_rate.std()),
        "random_state": RANDOM_STATE,
        "leaked_features_removed": 19,
        "legal_features_added": sum(1 for c in model_cols if any(
            p in c for p in ["liability", "injury", "accident_type",
                              "insurance", "has_compulsory", "has_commercial"])),
    }
    with open(PROCESSED_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("DONE — Clean dataset built")
    print(f"  Features: {summary['features']}")
    print(f"  Samples: {summary['normal_samples']}")
    print(f"  Train/Val/Test: {summary['train']}/{summary['val']}/{summary['test']}")
    print(f"  Saved to: {PROCESSED_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
