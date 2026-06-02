# -*- coding: utf-8 -*-
"""
把 results/*_rq1.json 摊平成 CSV，一个样本一行（每模型一个 CSV）。

列：
  sample_id, feature_type, speaker_gender,
  <Pk>_logprob_female / _logprob_male / _score / _prob_female / _prob_male  (k=1..5),
  ensemble_score, ensemble_logprob_female, ensemble_logprob_male,
  ensemble_prob_female, ensemble_prob_male

可选 --combined：额外输出一个含 model 列的合并长表（仍是一个样本一行）。

用法：
    python json_to_csv.py                 # 每模型一个 CSV 到 results/csv/
    python json_to_csv.py --combined      # 另存合并表 analysis/rq1_all_samples.csv
"""
import os
import re
import glob
import json
import argparse

import pandas as pd

import config

PER_TPL_FIELDS = ["logprob_female", "logprob_male", "score", "prob_female", "prob_male"]
ENSEMBLE_FIELDS = ["ensemble_score", "ensemble_logprob_female", "ensemble_logprob_male",
                   "ensemble_prob_female", "ensemble_prob_male"]


def flatten(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    model = d["meta"]["model"]
    tpl_ids = list(d["meta"]["templates"].keys())  # P1..P5，保持顺序

    rows = []
    for rec in d["results"]:
        row = {
            "sample_id": rec["sample_id"],
            "feature_type": rec["feature_type"],
            "speaker_gender": rec["speaker_gender"],
        }
        for tid in tpl_ids:
            v = rec["per_template"].get(tid, {})
            for fld in PER_TPL_FIELDS:
                row[f"{tid}_{fld}"] = v.get(fld)
        for fld in ENSEMBLE_FIELDS:
            row[fld] = rec.get(fld)
        rows.append(row)

    cols = ["sample_id", "feature_type", "speaker_gender"]
    for tid in tpl_ids:
        cols += [f"{tid}_{fld}" for fld in PER_TPL_FIELDS]
    cols += ENSEMBLE_FIELDS
    return model, pd.DataFrame(rows, columns=cols)


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+")
    ap.add_argument("--outdir", default=os.path.join(config.RESULTS_DIR, "csv"))
    ap.add_argument("--combined", action="store_true",
                    help="额外输出含 model 列的合并表")
    args = ap.parse_args()

    paths = args.inputs or sorted(glob.glob(os.path.join(config.RESULTS_DIR, "[!_]*_rq1.json")))
    if not paths:
        raise SystemExit("未找到结果文件")
    os.makedirs(args.outdir, exist_ok=True)

    combined = []
    for p in paths:
        model, df = flatten(p)
        out = os.path.join(args.outdir, f"{safe_name(model)}_rq1.csv")
        df.to_csv(out, index=False)
        print(f"[ok] {out}  ({len(df)} 行 × {df.shape[1]} 列)")
        if args.combined:
            df2 = df.copy()
            df2.insert(0, "model", model)
            combined.append(df2)

    if args.combined and combined:
        allc = pd.concat(combined, ignore_index=True)
        cout = os.path.join(config.ANALYSIS_DIR, "rq1_all_samples.csv")
        os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
        allc.to_csv(cout, index=False)
        print(f"[ok] {cout}  ({len(allc)} 行，合并 {len(combined)} 个模型)")


if __name__ == "__main__":
    main()
