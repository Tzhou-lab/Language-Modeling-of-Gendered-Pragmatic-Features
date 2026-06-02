#!/usr/bin/env bash
# RQ1 实验运行脚本。默认使用 tabebm25 conda 环境（torch 2.9 + transformers 4.56）。
set -euo pipefail
cd "$(dirname "$0")"

PY=/data1/ysf/miniconda3/envs/tabebm25/bin/python
GPU="${GPU:-0}"                       # 用环境变量覆盖：GPU=1 ./run.sh smoke
# 默认直连 huggingface.co（本机经代理可用）。如需镜像： HF_ENDPOINT=https://hf-mirror.com ./run.sh ...
[ -n "${HF_ENDPOINT:-}" ] && export HF_ENDPOINT

case "${1:-smoke}" in
  smoke)
    # 冒烟测试：Qwen3-4B，每类特征 20 条
    "$PY" rq1_inference.py --model Qwen/Qwen3-4B --gpu "$GPU" \
        --limit-per-feature 20 --batch-size 16 \
        --out results/_smoke_Qwen3-4B_rq1.json
    "$PY" analyze_rq1.py --inputs results/_smoke_Qwen3-4B_rq1.json \
        --out-csv analysis/_smoke_summary.csv
    ;;
  one)
    # 单模型全量：./run.sh one Qwen/Qwen3-8B
    "$PY" rq1_inference.py --model "${2:?需要模型名}" --gpu "$GPU"
    ;;
  all)
    # 全部 6 个模型全量
    "$PY" rq1_inference.py --all --gpu "$GPU"
    "$PY" analyze_rq1.py
    ;;
  analyze)
    "$PY" analyze_rq1.py
    ;;
  *)
    echo "用法: ./run.sh [smoke|one <model>|all|analyze]"; exit 1;;
esac
