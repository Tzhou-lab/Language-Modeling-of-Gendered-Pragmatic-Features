# -*- coding: utf-8 -*-
"""
RQ1 推理：对每条 utterance、每个 prompt 模板，提取模型在最后一个 token 位置上
"female" 与 "male" 的 next-token logit，导出 JSON。

用法：
    # 单模型
    python rq1_inference.py --model Qwen/Qwen3-4B --gpu 0

    # 冒烟测试：每类特征只取 N 条
    python rq1_inference.py --model Qwen/Qwen3-4B --gpu 0 --limit-per-feature 20 \
        --out results/_smoke_Qwen3-4B_rq1.json

    # 跑配置里的全部模型（顺序）
    python rq1_inference.py --all --gpu 0

输出 JSON（每模型一个）结构见文件末尾 build_record / save。
"""
import config  # 必须先 import，以便在 import transformers 前设置 HF_ENDPOINT

import os
import re
import json
import math
import time
import argparse

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_samples(csv_path, limit_per_feature=None,
                 id_col="sample_id", utterance_col="utterance",
                 feature_col="feature_type", gender_col="speaker_gender"):
    """读取样本 CSV，按列映射规范化为 sample_id/utterance/feature_type/speaker_gender。

    不同 CSV 列名不同（如 baseline 文件用 baseline_id / baseline_for_feature），
    通过 *_col 参数映射到统一字段，推理逻辑保持完全一致。
    engine='python' 以稳健处理含换行/引号的 utterance。
    """
    df = pd.read_csv(csv_path, engine="python")
    rename = {id_col: "sample_id", utterance_col: "utterance",
              feature_col: "feature_type", gender_col: "speaker_gender"}
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV 缺少列 {missing}；现有列：{list(df.columns)}")
    df = df.rename(columns=rename)[["sample_id", "utterance", "feature_type", "speaker_gender"]].copy()
    df["utterance"] = df["utterance"].astype(str)

    if limit_per_feature is not None:
        # 每类特征内尽量保持男女平衡地各取一半
        parts = []
        for ft, g in df.groupby("feature_type"):
            half = max(1, limit_per_feature // 2)
            males = g[g.speaker_gender == "male"].head(half)
            females = g[g.speaker_gender == "female"].head(limit_per_feature - len(males))
            parts.append(pd.concat([males, females]))
        df = pd.concat(parts).reset_index(drop=True)

    return df.to_dict("records")


def build_prompts(samples, templates):
    """展开 (sample, template) -> 扁平 prompt 列表，便于批处理。"""
    flat = []
    for s in samples:
        for tid, tmpl in templates.items():
            flat.append({
                "sample_id": s["sample_id"],
                "feature_type": s["feature_type"],
                "speaker_gender": s["speaker_gender"],
                "template_id": tid,
                "prompt": tmpl.format(u=s["utterance"]),
            })
    return flat


# ---------------------------------------------------------------------------
# 模型 / tokenizer
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(model_name, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # 左侧 padding：所有 prompt 右对齐，便于在末尾拼接候选词 token 做 teacher-forcing
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE_MAP[dtype],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tok, model


def candidate_token_ids(tok, word):
    """返回 word（含前导空格）的完整 subtoken id 列表。"""
    ids = tok.encode(word, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"无法对候选词 {word!r} 进行编码")
    return ids


# ---------------------------------------------------------------------------
# 推理核心
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model_name, samples, templates, gender_words,
                  device, dtype, batch_size, max_length):
    tok, model = load_model(model_name, device, dtype)

    # 候选词 -> 完整 subtoken id 列表
    word_tok = {label: candidate_token_ids(tok, word)
                for label, word in gender_words.items()}
    word_ntok = {label: len(ids) for label, ids in word_tok.items()}
    if any(n > 1 for n in word_ntok.values()):
        print(f"[info] 候选词分词：{word_tok}（n_subtokens={word_ntok}）；"
              f"采用整词条件 log-prob，多 subtoken / 共享空格前缀均可正确处理。")

    flat = build_prompts(samples, templates)

    # sample_id -> 聚合记录
    by_sample = {}
    for s in samples:
        by_sample[s["sample_id"]] = {
            "sample_id": s["sample_id"],
            "feature_type": s["feature_type"],
            "speaker_gender": s["speaker_gender"],
            "per_template": {},
        }

    t0 = time.time()
    n = len(flat)
    for start in range(0, n, batch_size):
        batch = flat[start:start + batch_size]
        prompts = [b["prompt"] for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        B, Tp = ids.shape

        # 对每个候选词，teacher-forcing 计算其条件 log-prob 之和
        lp = {}
        for label, wids in word_tok.items():
            w = torch.tensor(wids, device=device)
            L = w.numel()
            block = w.unsqueeze(0).expand(B, L)
            full_ids = torch.cat([ids, block], dim=1)
            full_attn = torch.cat([attn, torch.ones(B, L, dtype=attn.dtype, device=device)], dim=1)
            out = model(input_ids=full_ids, attention_mask=full_attn)
            # 仅取预测候选词所需的 L 个位置后再做 softmax，避免对整段 [B,T,V] 求 softmax 爆显存。
            # 左 padding 下 prompt 末位在 Tp-1；预测第 j 个候选 subtoken 的位置是 Tp-1+j。
            sl = out.logits[:, Tp - 1: Tp - 1 + L, :].float()  # [B, L, V]
            logp = torch.log_softmax(sl, dim=-1)               # [B, L, V]
            rows = torch.arange(B, device=device)
            total = torch.zeros(B, device=device)
            for j in range(L):
                total += logp[rows, j, w[j]]
            lp[label] = total  # [B] 整词 log-prob

        lpf, lpm = lp["female"], lp["male"]
        for i, b in enumerate(batch):
            rec = by_sample[b["sample_id"]]
            rec["per_template"][b["template_id"]] = {
                "logprob_female": float(lpf[i]),
                "logprob_male": float(lpm[i]),
                # 主指标：整词条件 log-prob 之差，正=偏女性归因
                "score": float(lpf[i] - lpm[i]),
            }

        done = min(start + batch_size, n)
        if (start // batch_size) % 20 == 0 or done == n:
            rate = done / max(1e-9, time.time() - t0)
            print(f"  [{model_name}] {done}/{n} prompts ({rate:.1f}/s)", flush=True)

    # 计算 prompt-ensemble（对模板取均值）
    records = []
    for s in samples:
        rec = by_sample[s["sample_id"]]
        pts = rec["per_template"]
        if not pts:
            continue
        scores = [v["score"] for v in pts.values()]
        lfs = [v["logprob_female"] for v in pts.values()]
        lms = [v["logprob_male"] for v in pts.values()]
        rec["ensemble_score"] = sum(scores) / len(scores)
        rec["ensemble_logprob_female"] = sum(lfs) / len(lfs)
        rec["ensemble_logprob_male"] = sum(lms) / len(lms)
        records.append(rec)

    meta = {
        "model": model_name,
        "n_samples": len(records),
        "n_templates": len(templates),
        "templates": templates,
        "gender_words": gender_words,
        "gender_subtoken_ids": word_tok,
        "gender_n_subtokens": word_ntok,
        "dtype": dtype,
        "score_definition": "score = logprob_female - logprob_male，"
                            "其中 logprob 为整词在 prompt 后的条件 log-prob 之和"
                            "(teacher-forcing，对多 subtoken / 共享空格前缀均稳健)；"
                            "ensemble_score = 5 个模板 score 的均值。正值=偏女性归因。",
    }

    # 释放显存
    del model
    torch.cuda.empty_cache()
    return meta, records


def save(out_path, meta, records):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": records}, f, ensure_ascii=False)
    print(f"[saved] {out_path}  ({len(records)} samples)")


def default_out_path(model_name, tag="rq1"):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return os.path.join(config.RESULTS_DIR, f"{safe}_{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="单个 HF repo id")
    ap.add_argument("--all", action="store_true", help="跑 config.MODELS 全部模型")
    ap.add_argument("--models", nargs="+", help="覆盖模型列表（空格分隔）")
    ap.add_argument("--gpu", type=int, default=0, help="使用的 GPU 序号（设置 CUDA_VISIBLE_DEVICES）")
    ap.add_argument("--limit-per-feature", type=int, default=None,
                    help="每类特征仅取 N 条（冒烟测试用）")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    ap.add_argument("--dtype", default=config.DTYPE, choices=list(DTYPE_MAP))
    ap.add_argument("--out", help="输出 JSON 路径（仅单模型时有意义）")
    # 输入 CSV 与列映射（默认主样本；baseline 用 --csv ... 并指定列名）
    ap.add_argument("--csv", default=config.DATA_CSV, help="输入样本 CSV 路径")
    ap.add_argument("--id-col", default="sample_id")
    ap.add_argument("--feature-col", default="feature_type")
    ap.add_argument("--gender-col", default="speaker_gender")
    ap.add_argument("--utterance-col", default="utterance")
    ap.add_argument("--tag", default="rq1",
                    help="输出文件名后缀，区分不同样本集（如 baseline）；默认 rq1")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请检查环境。")
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES 已限制，逻辑上恒为 0

    if args.models:
        model_list = args.models
    elif args.all:
        model_list = config.MODELS
    elif args.model:
        model_list = [args.model]
    else:
        raise SystemExit("请指定 --model / --models / --all 之一")

    samples = load_samples(args.csv, args.limit_per_feature,
                           id_col=args.id_col, utterance_col=args.utterance_col,
                           feature_col=args.feature_col, gender_col=args.gender_col)
    print(f"加载样本 {len(samples)} 条（{os.path.basename(args.csv)}，tag={args.tag}）；"
          f"模板 {len(config.PROMPT_TEMPLATES)} 个；待测模型 {len(model_list)} 个。")

    for mname in model_list:
        print(f"\n==== {mname} ====")
        try:
            meta, records = run_inference(
                mname, samples, config.PROMPT_TEMPLATES, config.GENDER_WORDS,
                device, args.dtype, args.batch_size, args.max_length)
        except Exception as e:
            print(f"[error] {mname} 失败：{type(e).__name__}: {e}")
            continue
        meta["sample_set"] = args.tag
        meta["source_csv"] = os.path.basename(args.csv)
        out = args.out if (args.out and len(model_list) == 1) else default_out_path(mname, args.tag)
        save(out, meta, records)


if __name__ == "__main__":
    main()
