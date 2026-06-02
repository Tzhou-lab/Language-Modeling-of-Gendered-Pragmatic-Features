# -*- coding: utf-8 -*-
"""
从 results/*_rq1.json 生成精简版 JSON：每个样本一条记录，只保留句子与各模板的
female/male 概率（softmax 后），结构形如：

  {
    "sample_id": "GB00001",
    "sentence": "...",
    "feature_type": "hedge",
    "speaker_gender": "male",
    "P1": {"female": 0.5156, "male": 0.4844},
    "P2": {"female": 0.6370, "male": 0.3630},
    ...
    "ensemble": {"female": 0.6141, "male": 0.3859}
  }

句子文本按 sample_id 从原始 CSV 取。输出到 results/concise/<model>.json。

用法：
    python make_concise_json.py            # 处理 results/ 下全部 *_rq1.json
    python make_concise_json.py --round 6  # 概率保留位数（默认 4）
"""
import os
import re
import glob
import json
import argparse

import pandas as pd

import config


def load_sentences():
    df = pd.read_csv(config.DATA_CSV, usecols=["sample_id", "utterance"])
    return dict(zip(df["sample_id"], df["utterance"].astype(str)))


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def concise(path, sentences, ndigits):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    model = d["meta"]["model"]
    tpl_ids = list(d["meta"]["templates"].keys())

    out = []
    for rec in d["results"]:
        sid = rec["sample_id"]
        item = {
            "sample_id": sid,
            "sentence": sentences.get(sid, ""),
            "feature_type": rec["feature_type"],
            "speaker_gender": rec["speaker_gender"],
        }
        for tid in tpl_ids:
            v = rec["per_template"].get(tid, {})
            item[tid] = {
                "female": round(v.get("prob_female", float("nan")), ndigits),
                "male": round(v.get("prob_male", float("nan")), ndigits),
            }
        item["ensemble"] = {
            "female": round(rec.get("ensemble_prob_female", float("nan")), ndigits),
            "male": round(rec.get("ensemble_prob_male", float("nan")), ndigits),
        }
        out.append(item)
    return model, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+")
    ap.add_argument("--outdir", default=os.path.join(config.RESULTS_DIR, "concise"))
    ap.add_argument("--round", type=int, default=4, dest="ndigits")
    args = ap.parse_args()

    paths = args.inputs or sorted(glob.glob(os.path.join(config.RESULTS_DIR, "[!_]*_rq1.json")))
    if not paths:
        raise SystemExit("未找到结果文件")
    os.makedirs(args.outdir, exist_ok=True)
    sentences = load_sentences()

    for p in paths:
        model, items = concise(p, sentences, args.ndigits)
        out = os.path.join(args.outdir, f"{safe_name(model)}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"[ok] {out}  ({len(items)} 条)")


if __name__ == "__main__":
    main()
