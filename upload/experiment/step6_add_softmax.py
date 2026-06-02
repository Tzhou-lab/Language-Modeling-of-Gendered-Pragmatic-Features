# -*- coding: utf-8 -*-
"""
后处理：直接在已有 results/*_rq1.json 上，对每个位置的 (logprob_female, logprob_male)
做 female/male 二分类 softmax，加入 prob_female / prob_male 字段，无需重跑模型。

  prob_female = exp(lp_f) / (exp(lp_f) + exp(lp_m)) = sigmoid(lp_f - lp_m)
  prob_male   = 1 - prob_female

同时为每条样本加 ensemble 概率（对 5 个模板的 prob 取均值）：
  ensemble_prob_female / ensemble_prob_male

用法：
    python add_softmax.py                  # 处理 results/ 下全部 *_rq1.json（原地加字段）
    python add_softmax.py --inputs a.json
"""
import os
import glob
import json
import math
import argparse

import config


def softmax2(lp_f, lp_m):
    """对两个 log-prob 做二分类 softmax，返回 (p_female, p_male)。数值稳定。"""
    m = max(lp_f, lp_m)
    ef = math.exp(lp_f - m)
    em = math.exp(lp_m - m)
    z = ef + em
    return ef / z, em / z


def process_file(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    for rec in d["results"]:
        pf_list, pm_list = [], []
        for tid, v in rec["per_template"].items():
            pf, pm = softmax2(v["logprob_female"], v["logprob_male"])
            v["prob_female"] = pf
            v["prob_male"] = pm
            pf_list.append(pf)
            pm_list.append(pm)
        if pf_list:
            rec["ensemble_prob_female"] = sum(pf_list) / len(pf_list)
            rec["ensemble_prob_male"] = sum(pm_list) / len(pm_list)

    note = ("prob_female/prob_male = 对 (logprob_female, logprob_male) 的二分类 softmax "
            "(= sigmoid(score))；ensemble_prob_* = 5 个模板 prob 的均值。")
    d["meta"]["softmax_note"] = note

    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return len(d["results"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+")
    args = ap.parse_args()
    paths = args.inputs or sorted(glob.glob(os.path.join(config.RESULTS_DIR, "[!_]*_rq1.json")))
    if not paths:
        raise SystemExit("未找到结果文件")
    for p in paths:
        n = process_file(p)
        print(f"[ok] {os.path.basename(p)}  +prob 字段 ({n} samples)")


if __name__ == "__main__":
    main()
