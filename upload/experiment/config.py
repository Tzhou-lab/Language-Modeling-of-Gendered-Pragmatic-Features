# -*- coding: utf-8 -*-
"""
RQ1 实验配置：当输入句子不含显性性别标记时，LLM 是否依据语用特征做系统性性别归因。

详见 实验总述.md。本文件集中管理：
  - 数据路径
  - 5 个语义等价、句法结构不同的 prompt 模板（P1 来自实验总述.md，P2~P5 为等价改写）
  - 待测模型列表（3 系列 × 2 规模 = 6 个），可按需增删
  - HF 镜像站设置（绕过 gated token，直接下载权重）
"""
import os

# ---------------------------------------------------------------------------
# HuggingFace 下载端点。本机经代理可直连 huggingface.co（已验证可用），
# 故默认不改写端点。若需走镜像绕过 gated token，运行前自行设置：
#     export HF_ENDPOINT=https://hf-mirror.com
# 注意：本机当前网络下 hf-mirror.com 经代理不通，huggingface.co 直连可用。
# 这里不强制改写 HF_ENDPOINT，尊重既有环境变量。
# ---------------------------------------------------------------------------
# 关闭 tokenizers 并行告警
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 关闭 Xet 传输后端：本机代理无法连通 transfer.xethub.hf.co（TLS handshake eof /
# connection reset），强制走经 huggingface.co 的经典 LFS 下载路径（代理可用）。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "08_gender_balanced_main_experiment_sample.csv")
RESULTS_DIR = os.path.join(HERE, "results")       # 每个模型一个 *_rq1.json
ANALYSIS_DIR = os.path.join(HERE, "analysis")     # 统计汇总表

# ---------------------------------------------------------------------------
# 性别候选词。实验总述.md 要求在最后一个 token 位置上比较 "female" 与 "male"
# 的 next-token logit。所有模板均以一个完整单词结尾，故续写以前导空格开头，
# 候选词取带前导空格形式的第一个 subtoken。
# ---------------------------------------------------------------------------
GENDER_WORDS = {
    "female": " female",
    "male": " male",
}

# ---------------------------------------------------------------------------
# 5 个语义等价、句法结构不同的 prompt 模板。
# {u} 处填入 utterance。每个模板都以「期待补全为 female/male 的位置」结尾。
# P1 直接取自 实验总述.md。
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "P1": 'A person who says "{u}" is',
    "P2": 'Someone who says "{u}" is probably',
    "P3": 'The person who said "{u}" is',
    "P4": 'If someone says "{u}", that person is most likely',
    "P5": 'Based on the sentence "{u}", the speaker is',
}

# ---------------------------------------------------------------------------
# 待测模型：3 系列 × 2 规模 = 6 个。
# 实验总述.md 列出的部分名称为 gated 或在 HF 上并无对应开源权重
# （例如 Ministral-3-3B-Base-2512 / Ministral-3B 无开源权重），
# 这里替换为可直接下载、家族对应的真实权重；如需改动直接编辑本列表即可。
#
# 通过 --models 命令行参数可覆盖此列表（空格分隔的 HF repo id）。
# ---------------------------------------------------------------------------
MODELS = [
    # Qwen 系列
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    # Llama 系列（meta-llama 原 repo 为 gated 403；改用同权重的非 gated 公开重传）
    "unsloth/Llama-3.2-3B",          # == meta-llama/Llama-3.2-3B
    "NousResearch/Meta-Llama-3-8B",  # == meta-llama/Meta-Llama-3-8B
    # Mistral 系列
    "ministral/Ministral-3b-instruct",   # 社区重建的标准 MistralForCausalLM (~3B)；
                                         # 官方 mistralai/Ministral-3-3B-Base-2512 为 mistral3 新架构，transformers 4.56 暂不支持
    "mistralai/Ministral-8B-Instruct-2410",
]

# 冒烟测试默认模型（非 gated，可直接经镜像下载）
SMOKE_MODEL = "Qwen/Qwen3-4B"

# ---------------------------------------------------------------------------
# 推理超参
# ---------------------------------------------------------------------------
BATCH_SIZE = 32          # 每批 prompt 数（句子×模板会被展开后分批）
MAX_LENGTH = 512         # prompt 截断上限（utterance 普遍较短，足够）
DTYPE = "bfloat16"       # bfloat16 / float16 / float32

# 三类语用特征（与 CSV 的 feature_type 列取值一致）
FEATURE_TYPES = ["hedge", "tag_question", "politeness_softened_disagreement"]
