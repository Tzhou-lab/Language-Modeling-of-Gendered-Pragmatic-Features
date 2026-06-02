# -*- coding: utf-8 -*-
"""
RQ1 分析：对每类语用特征下的 prompt-ensemble 得分分布做单样本 t 检验与
Wilcoxon 符号秩检验，检验均值/中位数是否显著偏离 0，以判断模型是否存在
系统性性别归因偏向（score = logit_female - logit_male，正=偏女性）。

用法：
    python analyze_rq1.py                       # 分析 results/ 下全部 *_rq1.json
    python analyze_rq1.py --inputs a.json b.json
    python analyze_rq1.py --inputs results/_smoke_*.json

输出：
    - 控制台打印每个模型 × 每类特征的统计表
    - analysis/rq1_summary.csv      （长表，便于跨模型比较）
"""
import config

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def load_result(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for r in d["results"]:
        rows.append({
            "sample_id": r["sample_id"],
            "feature_type": r["feature_type"],
            "speaker_gender": r["speaker_gender"],
            "ensemble_score": r["ensemble_score"],
            "ensemble_logprob_female": r.get("ensemble_logprob_female"),
            "ensemble_logprob_male": r.get("ensemble_logprob_male"),
        })
    df = pd.DataFrame(rows)
    return d["meta"]["model"], df


def cohens_d_onesample(x):
    """单样本 Cohen's d（相对 0）：mean / std。"""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def test_group(scores):
    """对一组得分做单样本 t 检验与 Wilcoxon 检验（均 vs 0）。"""
    x = np.asarray(scores, dtype=float)
    n = len(x)
    out = {
        "n": n,
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if n > 1 else float("nan"),
        "cohens_d": cohens_d_onesample(x) if n > 1 else float("nan"),
    }
    # t 检验
    if n > 1:
        t, p = stats.ttest_1samp(x, 0.0)
        out["t_stat"], out["t_p"] = float(t), float(p)
    else:
        out["t_stat"], out["t_p"] = float("nan"), float("nan")
    # Wilcoxon（需有非零差值）
    try:
        if np.any(x != 0):
            w, wp = stats.wilcoxon(x)
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(w), float(wp)
        else:
            out["wilcoxon_stat"], out["wilcoxon_p"] = float("nan"), float("nan")
    except Exception:
        out["wilcoxon_stat"], out["wilcoxon_p"] = float("nan"), float("nan")
    return out


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


def lean(mean):
    if mean > 0:
        return "→female"
    if mean < 0:
        return "→male"
    return "neutral"


def analyze(model, df):
    """返回该模型的长表行（按 feature_type，以及 overall）。"""
    rows = []
    groups = [(ft, df[df.feature_type == ft]) for ft in config.FEATURE_TYPES
              if (df.feature_type == ft).any()]
    groups.append(("__overall__", df))

    for ft, g in groups:
        res = test_group(g["ensemble_score"].values)
        res.update({"model": model, "feature_type": ft})
        rows.append(res)
    return rows


def print_model_table(model, rows):
    print(f"\n{'='*78}\n模型: {model}\n{'='*78}")
    hdr = f"{'feature_type':<34}{'n':>5}{'mean':>9}{'t_p':>10}{'wilcox_p':>10}  bias"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ft = r["feature_type"]
        mark = sig_mark(r["t_p"])
        bias = f"{lean(r['mean'])} {mark}" if mark != "ns" else "无显著偏向"
        print(f"{ft:<34}{r['n']:>5}{r['mean']:>9.4f}{r['t_p']:>10.2e}"
              f"{r['wilcoxon_p']:>10.2e}  {bias}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", help="指定 JSON 文件；默认 results/*_rq1.json")
    ap.add_argument("--out-csv", default=os.path.join(config.ANALYSIS_DIR, "rq1_summary.csv"))
    args = ap.parse_args()

    # 默认排除以 _ 开头的文件（如 _smoke_*）
    paths = args.inputs or sorted(glob.glob(os.path.join(config.RESULTS_DIR, "[!_]*_rq1.json")))
    if not paths:
        raise SystemExit(f"未找到结果文件，请先运行 rq1_inference.py。查找路径: {config.RESULTS_DIR}")

    all_rows = []
    for p in paths:
        model, df = load_result(p)
        rows = analyze(model, df)
        print_model_table(model, rows)
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    # 排序列
    cols = ["model", "feature_type", "n", "mean", "median", "std", "cohens_d",
            "t_stat", "t_p", "wilcoxon_stat", "wilcoxon_p"]
    out = out[[c for c in cols if c in out.columns]]
    os.makedirs(args.out_csv and os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[saved] {args.out_csv}")
    print("\n说明：score = logprob_female - logprob_male（整词条件 log-prob 之差，prompt-ensemble 均值）。")
    print("      mean>0 偏女性归因，<0 偏男性归因；*/**/*** 表示 t 检验 p<.05/.01/.001。")


if __name__ == "__main__":
    main()
