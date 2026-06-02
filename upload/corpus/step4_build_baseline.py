"""
build_baseline.py
=================
从 02_length_identity_filtered_with_feature_flags.csv 中抽取无特征基线样本。
性别信息复用 add_gender_labels.py 中的 survey.csv 读取逻辑。

输出：
  - 09_baseline_sample.csv          主基线文件（6126 条）
  - 09_baseline_diagnostics.csv     抽样诊断报告（每个箱的候选量与需求量）
  - 09_baseline_quality_check.csv   基线与主样本的词数分布对比
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 0. 配置 —— 修改为你电脑上的实际路径
# ============================================================

# 项目根目录（CANDOR_extracted 和 outputs 所在的文件夹）
PROJECT_DIR = Path(r"C:\Users\ztthe\Desktop\CANDOR Project4")

# CANDOR 解压目录（survey.csv 文件在这里）
EXTRACT_DIR = PROJECT_DIR / "CANDOR_extracted"

# outputs 目录
OUTPUT_DIR = PROJECT_DIR / "outputs"

# 输入文件
FLAGGED_FILE = OUTPUT_DIR / "02_length_identity_filtered_with_feature_flags.csv"
MAIN_SAMPLE_FILE = OUTPUT_DIR / "08_gender_balanced_main_experiment_sample.csv"

# 输出文件
BASELINE_OUTPUT = OUTPUT_DIR / "09_baseline_sample.csv"
DIAGNOSTICS_OUTPUT = OUTPUT_DIR / "09_baseline_diagnostics.csv"
QUALITY_CHECK_OUTPUT = OUTPUT_DIR / "09_baseline_quality_check.csv"

RANDOM_SEED = 42

# 词数分箱边界
LENGTH_BINS = [0, 10, 14, 18, 23, 28, 99]
LENGTH_LABELS = ["08-10", "11-14", "15-18", "19-23", "24-28", "29-35"]


# ============================================================
# 1. 从 survey.csv 读取说话人性别
#    （逻辑来自 add_gender_labels.py，精简为所需部分）
# ============================================================

def normalize_gender_label(value):
    if pd.isna(value):
        return "missing"
    x = str(value).strip().lower()
    if x in ["", "nan", "none", "null", "na", "n/a", "missing"]:
        return "missing"
    if x in ["m", "male", "man", "boy", "1"]:
        return "male"
    if x in ["f", "female", "woman", "girl", "2"]:
        return "female"
    return "other"


def normalize_speaker_id(value):
    if pd.isna(value):
        return ""
    x = str(value).strip()
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_speaker_metadata(extract_dir):
    """从每个对话文件夹的 survey.csv 中读取说话人性别"""
    survey_files = list(extract_dir.glob("*/survey.csv"))
    if not survey_files:
        raise FileNotFoundError(f"没有在 {extract_dir} 中找到 survey.csv")

    print(f"[OK] 找到 {len(survey_files)} 个 survey.csv 文件")

    rows = []
    for survey_path in survey_files:
        conversation_id = survey_path.parent.name

        try:
            df = pd.read_csv(survey_path)
        except Exception as e:
            print(f"[WARN] 无法读取 {survey_path}: {e}")
            continue

        speaker_id_col = find_column(df, [
            "id", "speaker", "speaker_id", "user_id",
            "participant_id", "participant", "channel",
        ])
        gender_col = find_column(df, [
            "sex", "gender", "Gender", "Sex",
            "participant_gender", "speaker_gender",
        ])

        if speaker_id_col is None or gender_col is None:
            continue

        for row in df.itertuples(index=False):
            raw_speaker = getattr(row, speaker_id_col)
            raw_gender = getattr(row, gender_col)
            speaker_id = normalize_speaker_id(raw_speaker)
            gender_norm = normalize_gender_label(raw_gender)

            if not speaker_id:
                continue

            rows.append({
                "conversation_id": str(conversation_id),
                "speaker": speaker_id,
                "speaker_gender": gender_norm,
            })

    speaker_meta = pd.DataFrame(rows)
    speaker_meta = speaker_meta.drop_duplicates(
        subset=["conversation_id", "speaker"], keep="first"
    )

    print(f"[OK] speaker metadata: {len(speaker_meta)} 行")
    print(f"     性别分布: {speaker_meta['speaker_gender'].value_counts().to_dict()}")

    return speaker_meta


# ============================================================
# 2. 分层抽样函数
# ============================================================

def stratified_sample(main_subset, pool, seed):
    """
    按词数分箱 × 说话人性别，从 pool 中抽取与 main_subset
    各箱数量相同的样本。

    返回: (抽样结果 DataFrame, 诊断信息 list of dict)
    """
    main_subset = main_subset.copy()
    main_subset["length_bin"] = pd.cut(
        main_subset["n_words"], bins=LENGTH_BINS, labels=LENGTH_LABELS
    )

    pool = pool.copy()
    pool["length_bin"] = pd.cut(
        pool["n_words"], bins=LENGTH_BINS, labels=LENGTH_LABELS
    )

    bin_counts = main_subset.groupby(
        ["length_bin", "speaker_gender"], observed=True
    ).size()

    sampled_parts = []
    diagnostics = []
    rng = np.random.RandomState(seed)

    for (bin_label, gender), needed in bin_counts.items():
        pool_slice = pool[
            (pool["length_bin"] == bin_label) &
            (pool["speaker_gender"] == gender)
        ]
        available = len(pool_slice)
        used_replace = available < needed

        if available == 0:
            print(f"  ⚠ {bin_label}/{gender}: 需要 {needed} 条，候选为 0，跳过")
            diagnostics.append({
                "length_bin": bin_label,
                "speaker_gender": gender,
                "needed": needed,
                "available": available,
                "used_replace": True,
                "sampled": 0,
            })
            continue

        sampled = pool_slice.sample(
            n=needed,
            replace=used_replace,
            random_state=rng.randint(0, 99999),
        )

        if used_replace:
            n_duplicates = needed - sampled.index.nunique()
            print(
                f"  ⚠ {bin_label}/{gender}: 需要 {needed} 条，"
                f"仅有 {available} 条，有放回抽样（{n_duplicates} 条重复）"
            )

        sampled_parts.append(sampled)
        diagnostics.append({
            "length_bin": bin_label,
            "speaker_gender": gender,
            "needed": needed,
            "available": available,
            "used_replace": used_replace,
            "sampled": len(sampled),
        })

    if not sampled_parts:
        return pd.DataFrame(), diagnostics

    result = pd.concat(sampled_parts).reset_index(drop=True)
    return result, diagnostics


# ============================================================
# 3. 主函数
# ============================================================

def main():
    # ── 3.1 加载主样本（用于排除和匹配） ──────────────────────
    print("=" * 60)
    print("加载主样本")
    print("=" * 60)

    main = pd.read_csv(MAIN_SAMPLE_FILE)
    main_keys = set(zip(main["conversation_id"].astype(str), main["turn_id"]))
    print(f"主样本：{len(main)} 条")

    # ── 3.2 加载候选池 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("加载候选池")
    print("=" * 60)

    pool = pd.read_csv(FLAGGED_FILE)
    print(f"02_with_feature_flags 总行数：{len(pool)}")

    # 排除含有任何目标特征的句子
    pool = pool[pool["n_target_features"] == 0].copy()
    print(f"无特征句子：{len(pool)}")

    # 排除已在主样本中的句子
    pool["conversation_id"] = pool["conversation_id"].astype(str)
    pool["_in_main"] = pool.apply(
        lambda r: (r["conversation_id"], r["turn_id"]) in main_keys, axis=1
    )
    pool = pool[~pool["_in_main"]].drop(columns=["_in_main"]).copy()
    print(f"排除主样本后：{len(pool)}")

    # ── 3.3 合并性别信息 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("合并说话人性别")
    print("=" * 60)

    speaker_meta = load_speaker_metadata(EXTRACT_DIR)

    pool["speaker"] = pool["speaker"].apply(normalize_speaker_id)
    pool = pool.merge(
        speaker_meta[["conversation_id", "speaker", "speaker_gender"]],
        on=["conversation_id", "speaker"],
        how="left",
    )
    pool["speaker_gender"] = pool["speaker_gender"].fillna("missing")

    # 只保留 female / male
    pool = pool[pool["speaker_gender"].isin(["female", "male"])].copy()
    print(f"有性别信息的候选池：{len(pool)}")
    print(f"  性别分布: {pool['speaker_gender'].value_counts().to_dict()}")

    # ── 3.4 按特征类别分别抽样 ────────────────────────────────
    print("\n" + "=" * 60)
    print("分层抽样")
    print("=" * 60)

    feature_types = ["hedge", "politeness_softened_disagreement", "tag_question"]
    baseline_parts = []
    all_diagnostics = []

    for ft in feature_types:
        print(f"\n── {ft} ──")
        main_sub = main[main["feature_type"] == ft]
        print(f"  主样本数量: {len(main_sub)}")

        sampled, diag = stratified_sample(main_sub, pool, seed=RANDOM_SEED)

        if sampled.empty:
            print(f"  ⚠ {ft} 抽样失败")
            continue

        sampled["baseline_for_feature"] = ft
        sampled["baseline_id"] = [
            f"BL_{ft[:3].upper()}_{i:05d}" for i in range(len(sampled))
        ]

        baseline_parts.append(sampled)
        for d in diag:
            d["feature_type"] = ft
        all_diagnostics.extend(diag)

        print(f"  抽取完成: {len(sampled)} 条")
        print(f"  性别: {sampled['speaker_gender'].value_counts().to_dict()}")
        print(f"  词数均值: {sampled['n_words'].mean():.1f}（主样本: {main_sub['n_words'].mean():.1f}）")

    # ── 3.5 合并保存 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("保存输出")
    print("=" * 60)

    baseline = pd.concat(baseline_parts, ignore_index=True)
    baseline.to_csv(BASELINE_OUTPUT, index=False)
    print(f"[OK] 基线样本: {BASELINE_OUTPUT} ({len(baseline)} 条)")

    # 诊断报告
    diag_df = pd.DataFrame(all_diagnostics)
    diag_df.to_csv(DIAGNOSTICS_OUTPUT, index=False)
    print(f"[OK] 诊断报告: {DIAGNOSTICS_OUTPUT}")

    has_replace = diag_df["used_replace"].any()
    if has_replace:
        print("\n⚠ 部分箱使用了有放回抽样，详见诊断报告。")
    else:
        print("\n✓ 所有箱均为无放回抽样，无重复。")

    # ── 3.6 质量检验表 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("质量检验")
    print("=" * 60)

    quality_rows = []
    for ft in feature_types:
        m = main[main["feature_type"] == ft]["n_words"]
        b = baseline[baseline["baseline_for_feature"] == ft]["n_words"]

        from scipy import stats
        ks_stat, ks_p = stats.ks_2samp(m, b)

        quality_rows.append({
            "feature_type": ft,
            "main_mean": round(m.mean(), 2),
            "main_std": round(m.std(), 2),
            "baseline_mean": round(b.mean(), 2),
            "baseline_std": round(b.std(), 2),
            "main_n": len(m),
            "baseline_n": len(b),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(ks_p, 4),
        })

        print(f"\n{ft}:")
        print(f"  主样本: {m.mean():.1f} ± {m.std():.1f} (n={len(m)})")
        print(f"  基  线: {b.mean():.1f} ± {b.std():.1f} (n={len(b)})")
        print(f"  KS检验: D={ks_stat:.4f}, p={ks_p:.4f}")

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(QUALITY_CHECK_OUTPUT, index=False)
    print(f"\n[OK] 质量检验表: {QUALITY_CHECK_OUTPUT}")

    print("\n[DONE] 基线构造完成。")


if __name__ == "__main__":
    main()
