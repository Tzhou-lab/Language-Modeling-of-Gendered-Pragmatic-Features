# -*- coding: utf-8 -*-
"""
RQ3 分析：比较主样本（含目标语用特征）与基线样本（不含目标语用特征）的
prompt-ensemble 得分差异，检验语用特征的存在是否额外改变了模型的性别归因。

核心指标：
  Δ = main_mean − baseline_mean
  正 Δ → 语用特征使归因更偏女性（相对于同分布无特征基线）
  负 Δ → 语用特征使归因更偏男性

检验方法：
  独立样本 Welch t 检验（主样本与基线非配对，词数/性别分布匹配但非逐句对应）
  Cohen's d（两独立样本）

用法：
    python analyze_baseline_delta.py                  # 自动配对 results/ 下的 *_rq1.json 与 *_baseline.json
    python analyze_baseline_delta.py --exclude-model HuggingFaceH4/zephyr-7b-beta  # 排除特定模型

输出：
    - 控制台：每个模型 × 每类特征的 delta 表 + 基线均匀性检查
    - analysis/rq3_delta_summary.csv         （长表：model × feature 一行）
    - analysis/rq3_baseline_uniformity.csv   （基线三类特征得分的均匀性）
"""
import config

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 1. 数据加载
# ============================================================

def load_result(path):
    """读取单个结果 JSON，返回 (model_name, DataFrame)。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for r in d["results"]:
        rows.append({
            "sample_id":      r["sample_id"],
            "feature_type":   r["feature_type"],
            "speaker_gender": r["speaker_gender"],
            "ensemble_score": r["ensemble_score"],
        })
    return d["meta"]["model"], pd.DataFrame(rows)


def discover_pairs(results_dir):
    """
    在 results_dir 中自动配对 *_rq1.json 与 *_baseline.json。
    返回 [(model_name, rq1_path, baseline_path), ...]。
    """
    rq1_files = sorted(glob.glob(os.path.join(results_dir, "*_rq1.json")))
    pairs = []
    for rq1_path in rq1_files:
        # _rq1.json → _baseline.json
        base = rq1_path.rsplit("_rq1.json", 1)[0]
        bl_path = base + "_baseline.json"
        if os.path.exists(bl_path):
            model_name, _ = load_result(rq1_path)
            pairs.append((model_name, rq1_path, bl_path))
        else:
            fname = os.path.basename(rq1_path)
            print(f"[WARN] {fname} 没有对应的 baseline 文件，跳过。")
    return pairs


# ============================================================
# 2. Delta 统计
# ============================================================

def cohens_d_independent(x, y):
    """两独立样本 Cohen's d（pooled SD）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    pooled_sd = np.sqrt(pooled_var)
    return float((x.mean() - y.mean()) / pooled_sd) if pooled_sd > 0 else float("nan")


def test_delta(main_scores, baseline_scores):
    """对主样本与基线的得分差异做独立样本 Welch t 检验，含 95% CI。"""
    m = np.asarray(main_scores, float)
    b = np.asarray(baseline_scores, float)
    nm, nb = len(m), len(b)
    delta = float(m.mean() - b.mean())

    result = {
        "n_main":     nm,
        "n_baseline": nb,
        "main_mean":  float(m.mean()),
        "main_std":   float(m.std(ddof=1)),
        "bl_mean":    float(b.mean()),
        "bl_std":     float(b.std(ddof=1)),
        "delta":      delta,
    }

    # Welch t 检验（不假设等方差）
    t_stat, t_p = stats.ttest_ind(m, b, equal_var=False)
    result["t_stat"] = float(t_stat)
    result["t_p"]    = float(t_p)

    # 95% CI for delta（Welch-Satterthwaite 自由度）
    se_m = m.var(ddof=1) / nm
    se_b = b.var(ddof=1) / nb
    se_delta = np.sqrt(se_m + se_b)
    # Welch-Satterthwaite df
    df_ws = (se_m + se_b) ** 2 / (se_m ** 2 / (nm - 1) + se_b ** 2 / (nb - 1))
    t_crit = stats.t.ppf(0.975, df_ws)
    result["ci_lo"] = float(delta - t_crit * se_delta)
    result["ci_hi"] = float(delta + t_crit * se_delta)

    # 效应量
    result["cohens_d"] = cohens_d_independent(m, b)
    return result


# ============================================================
# 3. 分析流程
# ============================================================

def analyze_model(model_name, main_df, bl_df):
    """对一个模型做全部 delta 分析，返回行列表。"""
    feature_types = config.FEATURE_TYPES  # hedge, tag_question, politeness_softened_disagreement
    rows = []

    # 按特征分别分析
    for ft in feature_types:
        m_scores = main_df.loc[main_df.feature_type == ft, "ensemble_score"].values
        b_scores = bl_df.loc[bl_df.feature_type == ft, "ensemble_score"].values
        if len(m_scores) == 0 or len(b_scores) == 0:
            print(f"  [WARN] {model_name} / {ft}: 主样本或基线为空，跳过。")
            continue
        res = test_delta(m_scores, b_scores)
        res["model"] = model_name
        res["feature_type"] = ft
        rows.append(res)

    # 总体（三类合并）
    m_all = main_df["ensemble_score"].values
    b_all = bl_df["ensemble_score"].values
    res = test_delta(m_all, b_all)
    res["model"] = model_name
    res["feature_type"] = "__overall__"
    rows.append(res)

    return rows


def check_baseline_uniformity(model_name, bl_df):
    """
    检查基线样本在三类特征组内的得分是否趋于一致。
    若基线三类得分接近（range 很小），则主样本中的特征间差异
    更可归因于语用特征本身，而非句子长度/语域差异。
    """
    feature_types = config.FEATURE_TYPES
    means = {}
    for ft in feature_types:
        scores = bl_df.loc[bl_df.feature_type == ft, "ensemble_score"].values
        means[ft] = float(scores.mean()) if len(scores) > 0 else float("nan")

    vals = [v for v in means.values() if not np.isnan(v)]
    rng = max(vals) - min(vals) if len(vals) >= 2 else float("nan")

    return {
        "model": model_name,
        "bl_hedge":      means.get("hedge", float("nan")),
        "bl_tag":        means.get("tag_question", float("nan")),
        "bl_politeness": means.get("politeness_softened_disagreement", float("nan")),
        "bl_range":      rng,
    }


# ============================================================
# 4. 输出格式
# ============================================================

def sig_mark(p):
    if p != p:  # nan
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


FEATURE_LABELS = {
    "hedge":                              "hedge",
    "tag_question":                       "tag_question",
    "politeness_softened_disagreement":   "politeness/sd",
    "__overall__":                        "OVERALL",
}


def print_delta_table(model_name, rows):
    print(f"\n{'=' * 110}")
    print(f"模型: {model_name}")
    print(f"{'=' * 110}")
    hdr = (f"{'feature':<20}  {'main':>8}  {'baseline':>8}  "
           f"{'Δ':>8}  {'95% CI':>19}  {'d':>7}  {'p':>11}  sig")
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        ft = FEATURE_LABELS.get(r["feature_type"], r["feature_type"])
        mark = sig_mark(r["t_p"])
        direction = "→F" if r["delta"] > 0 else "→M" if r["delta"] < 0 else "="
        ci_str = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        print(f"{ft:<20}  {r['main_mean']:>+8.4f}  {r['bl_mean']:>+8.4f}  "
              f"{r['delta']:>+8.4f}  {ci_str:>19}  {r['cohens_d']:>+7.3f}  "
              f"{r['t_p']:>11.2e}  {mark} {direction}")


def print_uniformity_table(uniformity_rows):
    print(f"\n{'=' * 90}")
    print("基线均匀性检查（三类特征组的基线得分是否趋于一致）")
    print(f"{'=' * 90}")
    hdr = f"{'model':<50}  {'bl_hedge':>9}  {'bl_tag':>9}  {'bl_polit':>9}  {'range':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in uniformity_rows:
        print(f"{r['model']:<50}  {r['bl_hedge']:>+9.4f}  {r['bl_tag']:>+9.4f}  "
              f"{r['bl_politeness']:>+9.4f}  {r['bl_range']:>7.4f}")


# ============================================================
# 5. 主函数
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="RQ3: main vs baseline delta analysis")
    ap.add_argument("--exclude-model", nargs="*", default=[],
                    help="要排除的模型名称（如 HuggingFaceH4/zephyr-7b-beta）")
    ap.add_argument("--results-dir", default=config.RESULTS_DIR)
    ap.add_argument("--out-dir", default=config.ANALYSIS_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 发现所有 main + baseline 配对
    pairs = discover_pairs(args.results_dir)
    if not pairs:
        raise SystemExit("未找到任何 main + baseline 配对文件。")

    # 排除指定模型
    if args.exclude_model:
        excluded = set(args.exclude_model)
        pairs = [(m, r, b) for m, r, b in pairs if m not in excluded]
        print(f"[INFO] 排除模型: {args.exclude_model}")

    print(f"[INFO] 找到 {len(pairs)} 个模型配对。")

    all_delta_rows = []
    all_uniformity_rows = []

    for model_name, rq1_path, bl_path in pairs:
        _, main_df = load_result(rq1_path)
        _, bl_df   = load_result(bl_path)

        # delta 分析
        delta_rows = analyze_model(model_name, main_df, bl_df)
        print_delta_table(model_name, delta_rows)
        all_delta_rows.extend(delta_rows)

        # 基线均匀性
        uni = check_baseline_uniformity(model_name, bl_df)
        all_uniformity_rows.append(uni)

    # 打印基线均匀性汇总
    print_uniformity_table(all_uniformity_rows)

    # 保存 CSV
    delta_df = pd.DataFrame(all_delta_rows)
    cols = ["model", "feature_type", "n_main", "n_baseline",
            "main_mean", "main_std", "bl_mean", "bl_std",
            "delta", "ci_lo", "ci_hi", "cohens_d", "t_stat", "t_p"]
    delta_df = delta_df[[c for c in cols if c in delta_df.columns]]

    delta_csv = os.path.join(args.out_dir, "rq3_delta_summary.csv")
    delta_df.to_csv(delta_csv, index=False)
    print(f"\n[saved] {delta_csv}")

    uni_df = pd.DataFrame(all_uniformity_rows)
    uni_csv = os.path.join(args.out_dir, "rq3_baseline_uniformity.csv")
    uni_df.to_csv(uni_csv, index=False)
    print(f"[saved] {uni_csv}")

    # 打印核心发现摘要
    print(f"\n{'=' * 90}")
    print("核心发现摘要")
    print(f"{'=' * 90}")

    # 检查礼貌/缓和式不同意的 delta 是否在所有模型中一致为正
    polit_rows = [r for r in all_delta_rows
                  if r["feature_type"] == "politeness_softened_disagreement"]
    polit_positive = sum(1 for r in polit_rows if r["delta"] > 0)
    polit_sig      = sum(1 for r in polit_rows if r["t_p"] < 0.05 and r["delta"] > 0)
    print(f"  礼貌/缓和式不同意: Δ > 0 在 {polit_positive}/{len(polit_rows)} 个模型中成立"
          f"（其中 {polit_sig} 个显著）")

    hedge_rows = [r for r in all_delta_rows if r["feature_type"] == "hedge"]
    hedge_positive = sum(1 for r in hedge_rows if r["delta"] > 0)
    hedge_sig_pos  = sum(1 for r in hedge_rows if r["t_p"] < 0.05 and r["delta"] > 0)
    hedge_sig_neg  = sum(1 for r in hedge_rows if r["t_p"] < 0.05 and r["delta"] < 0)
    print(f"  模糊限制语:         Δ > 0 在 {hedge_positive}/{len(hedge_rows)} 个模型中成立"
          f"（显著正 {hedge_sig_pos} 个，显著负 {hedge_sig_neg} 个）")

    tag_rows = [r for r in all_delta_rows if r["feature_type"] == "tag_question"]
    tag_positive = sum(1 for r in tag_rows if r["delta"] > 0)
    tag_sig_pos  = sum(1 for r in tag_rows if r["t_p"] < 0.05 and r["delta"] > 0)
    tag_sig_neg  = sum(1 for r in tag_rows if r["t_p"] < 0.05 and r["delta"] < 0)
    print(f"  附加疑问句:         Δ > 0 在 {tag_positive}/{len(tag_rows)} 个模型中成立"
          f"（显著正 {tag_sig_pos} 个，显著负 {tag_sig_neg} 个）")

    # 基线均匀性摘要
    avg_range = np.mean([r["bl_range"] for r in all_uniformity_rows])
    print(f"\n  基线均匀性: 三类特征组基线得分的平均极差 = {avg_range:.4f}")
    print(f"  （越接近 0，说明基线越均匀，主样本的特征间差异越可归因于语用特征）")

    print("\n说明：")
    print("  Δ = main_mean − baseline_mean（正值 = 语用特征使归因更偏女性）")
    print("  检验为独立样本 Welch t 检验（主样本与基线非配对但分布匹配）")
    print("  * p<.05  ** p<.01  *** p<.001  ns = 不显著")


if __name__ == "__main__":
    main()
