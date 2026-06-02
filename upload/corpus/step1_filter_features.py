import zipfile
import re
from pathlib import Path
import pandas as pd

from config import (
    PROJECT_DIR,
    ZIP_PATH,
    EXTRACT_DIR,
    OUTPUT_DIR,
    MIN_WORDS,
    MAX_WORDS,
    TRANSCRIPT_FILENAME,
    RANDOM_SEED,
    MAX_PER_CONV_PER_FEATURE,
    MIN_CONTENT_WORDS,
    PREFIX_DEDUP_WORDS,
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. 自动解压 CANDOR.zip
# ============================================================

def unzip_candor():
    if EXTRACT_DIR.exists():
        print(f"[OK] 已发现解压文件夹：{EXTRACT_DIR}")
        return

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {ZIP_PATH}。请确认 CANDOR.zip 在 CANDOR Project 文件夹中。"
        )

    print("[INFO] 正在解压 CANDOR.zip，这一步可能需要一些时间...")
    EXTRACT_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print(f"[OK] 解压完成：{EXTRACT_DIR}")


# ============================================================
# 4. 文本清理与基础函数
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text):
    return len(re.findall(r"\b[A-Za-z]+(?:[''][A-Za-z]+)?\b", text))


def contains_question_mark(text):
    return "?" in text


def contains_first_person(text):
    return bool(re.search(r"\b(i|me|my|mine|we|us|our|ours)\b", text, re.IGNORECASE))


def contains_negation(text):
    return bool(re.search(
        r"\b(not|no|never|n't|n't|don['']?t|doesn['']?t|didn['']?t|isn['']?t"
        r"|wasn['']?t|weren['']?t|wouldn['']?t|couldn['']?t|shouldn['']?t)\b",
        text, re.IGNORECASE,
    ))


# ============================================================
# 4.1 低信息量 / 转录噪音过滤
# ============================================================

BACKCHANNEL_REGEX = re.compile(
    r"\b(yeah|yep|okay|ok|right|mhm|mm-hm|uh-huh|mm|hmm)\b",
    flags=re.IGNORECASE,
)

FILLER_WORDS = {
    "yeah", "yep", "okay", "ok", "right",
    "mhm", "mm", "hmm", "uh", "um", "oh", "ah"
}


def _count_backchannel_fragments(text):
    """
    按句子边界切分文本，统计"仅由 backchannel/filler 词构成"的片段数量。
    数量 >= 3 说明这段文本是多话轮拼接，而非单一 utterance。

    例：
      "Mhm. I get it. Yeah. Yeah. I watched them too."
      → 片段: ["Mhm"✓, "I get it"✗, "Yeah"✓, "Yeah"✓, "I watched them too"✗]
      → 纯 backchannel 片段数 = 3 → 过滤
    """
    fragments = re.split(r'[.!?]+', text)
    count = 0
    for frag in fragments:
        words = re.findall(r'\b[A-Za-z]+\b', frag.strip().lower())
        if words and all(w in FILLER_WORDS for w in words):
            count += 1
    return count


def is_low_content_noise(text):
    if pd.isna(text):
        return True
    text_str = str(text).strip().lower()
    tokens = re.findall(r"[A-Za-z]+(?:[''][A-Za-z]+)?", text_str)
    if not tokens:
        return True

    bc_count   = len(BACKCHANNEL_REGEX.findall(text_str))
    fill_count = sum(t in FILLER_WORDS for t in tokens)

    # Rule 1：短句里 backchannel 密集
    if bc_count >= 2 and len(tokens) <= 10:
        return True

    # Rule 2：大多数词都是 filler
    if len(tokens) >= 6 and fill_count / len(tokens) >= 0.7:
        return True

    # Rule 3：3 个以上纯 backchannel 句子片段 → 多话轮拼接段
    # 解决"Thank you. Mhm. Okay. Yeah. Yeah. Mhm. ..."这类长拼接转录
    if _count_backchannel_fragments(str(text)) >= 3:
        return True

    return False


# ============================================================
# 4.2 内容词过滤（新增）
# ============================================================
# 目标：确保 feature 候选句在去除 feature 表达式之后，
# 仍有足够的实质内容词，避免消融后近乎为空的句子进入实验。
#
# 实现逻辑（两步）：
#   Step 1  用与检测完全相同的 FEATURE_PATTERNS 正则，
#           把 feature 表达式从句子里扣除。
#           → 不另立词表，与检测规则严格对应。
#   Step 2  对扣除后的剩余文本，去掉 BASIC_FUNCTION_WORDS
#           （语言学公认的功能词），再计算剩余内容词数。
#
# BASIC_FUNCTION_WORDS 只收录无争议的功能词类别：
#   冠词、介词、连词、代词、助动词、口语感叹词。
#   不含任何 feature 相关词——feature 由 Step 1 的正则负责。
# ─────────────────────────────────────────────────────────────

BASIC_FUNCTION_WORDS = {
    # 冠词 / 介词 / 连词
    "a", "an", "the",
    "and", "or", "but", "if", "so", "nor",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "as", "than", "though", "although",
    "while", "since", "because", "after", "before",
    # 代词
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose",
    "myself", "yourself", "himself", "herself", "itself",
    # 助动词 / be 动词
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "shall", "can",
    # 无实义副词 / 限定词
    "not", "no", "just", "also", "too", "very", "quite",
    "more", "most", "then", "now", "here", "there",
    "both", "all", "any", "some", "each", "every",
    "other", "same", "own", "only",
    # 口语感叹词（不含语义内容）
    "oh", "ah", "uh", "um", "yeah", "yes", "yep",
    "okay", "ok", "mhm", "nah", "wow",
}


def count_residual_content_words(text, matched_features):
    """
    Step 1：用 FEATURE_PATTERNS 里对应的正则，把 feature 表达式从文本中扣除。
    Step 2：对剩余文本，去除 BASIC_FUNCTION_WORDS 后计算内容词数。

    注意：跳过含 .{0, 的长跨度模式（如 r"\bi think .{0,40}\bmight\b"）。
    这类模式用于检测时是正确的，但用于内容扣除时会把中间的实质内容词一并删掉。
    例如 "I think this approach might work"：
      - 若使用长跨度模式，会删除 "I think this approach might"，只剩 "work"（1词）
      - 跳过长跨度后，只删 "I think"，"might" 已在 BASIC_FUNCTION_WORDS 里处理
        剩余内容词：approach, work → 2词 → 正确通过

    参数
    ----
    text             : 原始 utterance
    matched_features : detect_features() 返回的 list，如 ["hedge"]
    """
    masked = text
    for ft in matched_features:
        if ft in FEATURE_PATTERNS:
            for pattern in FEATURE_PATTERNS[ft]:
                if ".{0," in pattern:   # 跳过长跨度模式，只用于检测
                    continue
                masked = re.sub(pattern, " ", masked, flags=re.IGNORECASE)

    words = re.findall(r"\b[A-Za-z]+\b", masked.lower())
    return sum(1 for w in words if w not in BASIC_FUNCTION_WORDS)


# ============================================================
# 4.3 前缀近重复去重（新增）
# ============================================================
# 目标：在同一 feature_type 组内，对前 PREFIX_DEDUP_WORDS 个词
# 完全相同的句子做去重，只保留内容词最多的那条。
#
# 解决的问题：
#   "Oh thank you. Thank you. You also you to you know..."
#   "Oh thank you. Thank you. You too as well."
#   两句前 6 词完全一致 → 保留内容词较多的一条，删除另一条。
# ─────────────────────────────────────────────────────────────

def _normalize_prefix(text, n_words):
    """取前 n 个词的小写形式作为指纹（去标点）"""
    words = re.findall(r"\b[A-Za-z]+\b", text.lower())
    return " ".join(words[:n_words])


def deduplicate_by_prefix(df, prefix_words=PREFIX_DEDUP_WORDS):
    """
    在 df 内部，按 feature_type + 前缀指纹去重，
    保留内容词最多的一条（内容最丰富的句子优先）。
    """
    if df.empty or "feature_type" not in df.columns:
        return df

    df = df.copy()
    df["_prefix"]        = df["utterance"].apply(lambda t: _normalize_prefix(t, prefix_words))
    # 去重时用原始文本的内容词数排序（不需要 feature 感知，直接用 BASIC_FUNCTION_WORDS）
    df["_content_words"] = df["utterance"].apply(
        lambda t: sum(1 for w in re.findall(r"\b[A-Za-z]+\b", t.lower())
                      if w not in BASIC_FUNCTION_WORDS)
    )

    before = len(df)

    # 同一 feature_type + prefix 组内按内容词降序，保留第一条
    df = (
        df.sort_values("_content_words", ascending=False)
          .drop_duplicates(subset=["feature_type", "_prefix"], keep="first")
          .drop(columns=["_prefix", "_content_words"])
    )

    removed = before - len(df)
    print(f"[INFO] 前缀近重复去重：移除 {removed} 条，剩余 {len(df)} 条")

    return df


# ============================================================
# 5. 显性身份线索过滤
# ============================================================

IDENTITY_PATTERNS = [
    r"\bhe\b", r"\bshe\b", r"\bhim\b", r"\bher\b",
    r"\bhis\b", r"\bhers\b", r"\bhimself\b", r"\bherself\b",
    r"\bman\b", r"\bwoman\b", r"\bmen\b", r"\bwomen\b",
    r"\bmale\b", r"\bfemale\b",
    r"\bboy\b", r"\bgirl\b", r"\bboys\b", r"\bgirls\b",
    r"\bboyfriend\b", r"\bgirlfriend\b",
    r"\bhusband\b", r"\bwife\b",
    r"\bfiance\b", r"\bfiancée\b",
    r"\bfather\b", r"\bmother\b", r"\bdad\b", r"\bmom\b", r"\bmum\b",
    r"\bson\b", r"\bdaughter\b",
    r"\bbrother\b", r"\bsister\b",
    r"\bgrandfather\b", r"\bgrandmother\b",
    r"\bgrandpa\b", r"\bgrandma\b",
    r"\buncle\b", r"\baunt\b",
]

IDENTITY_REGEX = re.compile("|".join(IDENTITY_PATTERNS), flags=re.IGNORECASE)


def has_identity_cue(text):
    return bool(IDENTITY_REGEX.search(text))


# ============================================================
# 6. 三类 feature 的检索规则
# ============================================================

FEATURE_PATTERNS = {
    "hedge": [
        r"\bi think\b", r"\bi guess\b", r"\bi suppose\b", r"\bi feel like\b",
        r"\bmaybe\b", r"\bprobably\b", r"\bperhaps\b",
        r"\bsort of\b", r"\bkind of\b",
        r"\bit seems\b", r"\bit seems like\b", r"\bpossibly\b",
        r"\bi think .{0,40}\bmight\b", r"\bi guess .{0,40}\bmight\b",
        r"\bi suppose .{0,40}\bmight\b",
        r"\bit might be\b", r"\bthat might be\b", r"\bthis might be\b",
        r"\bthere might be\b", r"\bmight have\b",
        r"\bi think .{0,40}\bcould\b", r"\bi guess .{0,40}\bcould\b",
        r"\bi suppose .{0,40}\bcould\b",
        r"\bit could be\b", r"\bthat could be\b", r"\bthis could be\b",
        r"\bthere could be\b", r"\bcould have\b",
    ],
    "tag_question": [
        r",?\s*right\?\s*$",
        r",?\s*isn['']?t it\?\s*$",
        r",?\s*aren['']?t you\?\s*$",
        r",?\s*don['']?t you\?\s*$",
        r",?\s*doesn['']?t it\?\s*$",
        r",?\s*didn['']?t you\?\s*$",
        r",?\s*wasn['']?t it\?\s*$",
        r",?\s*weren['']?t you\?\s*$",
        r",?\s*you know\?\s*$",
        r",?\s*is that fair\?\s*$",
    ],
    "politeness_softened_disagreement": [
        r"\bplease\b", r"\bthank you\b", r"\bthanks\b",
        r"\bexcuse me\b", r"\bi appreciate\b", r"\bif you don['']?t mind\b",
        r"\bsorry,?\s*(i|that|about|for)\b",
        r"\bi['']?m sorry\b", r"\bi am sorry\b",
        r"\bi see your point,?\s*but\b", r"\bi understand,?\s*but\b",
        r"\bi get that,?\s*but\b",
        r"\bthat['']?s true,?\s*but\b", r"\bthat['']?s true,?\s*although\b",
        r"\bi['']?m not sure that['']?s right\b",
        r"\bi am not sure that['']?s right\b",
        r"\bi wouldn['']?t say that\b",
        r"\bi don['']?t know if that works\b",
    ],
}

COMPILED_FEATURES = {
    feature: re.compile("|".join(patterns), flags=re.IGNORECASE)
    for feature, patterns in FEATURE_PATTERNS.items()
}


def detect_features(text):
    found = []
    for feature, regex in COMPILED_FEATURES.items():
        if regex.search(text):
            found.append(feature)
    return found


# ============================================================
# 7. 查找 transcript_cliffhanger.csv
# ============================================================

def find_transcript_files():
    files = list(EXTRACT_DIR.rglob(TRANSCRIPT_FILENAME))
    if not files:
        raise FileNotFoundError(
            f"没有在 {EXTRACT_DIR} 中找到任何 {TRANSCRIPT_FILENAME}。"
        )
    print(f"[OK] 找到 {len(files)} 个 {TRANSCRIPT_FILENAME} 文件")
    return files


def infer_conversation_id(transcript_path):
    if transcript_path.parent.name.lower() == "transcription":
        return transcript_path.parent.parent.name
    return transcript_path.parent.name


# ============================================================
# 8. 自动识别文本列和 speaker 列
# ============================================================

def find_text_column(df):
    candidates = ["utterance", "text", "transcript", "sentence", "content", "value"]
    for col in candidates:
        if col in df.columns:
            return col
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not object_cols:
        return None
    avg_lengths = {col: df[col].astype(str).str.len().mean() for col in object_cols}
    return max(avg_lengths, key=avg_lengths.get)


def find_speaker_column(df):
    candidates = ["speaker", "speaker_id", "participant", "channel", "user_id", "id"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ============================================================
# 9. 读取所有 utterances
# ============================================================

def load_all_utterances(transcript_files):
    rows = []
    for path in transcript_files:
        conversation_id = infer_conversation_id(path)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] 无法读取 {path}: {e}")
            continue
        text_col    = find_text_column(df)
        speaker_col = find_speaker_column(df)
        if text_col is None:
            print(f"[WARN] 找不到文本列，跳过：{path}")
            continue
        for turn_id, row in df.iterrows():
            utterance = normalize_text(row[text_col])
            if not utterance:
                continue
            rows.append({
                "conversation_id": conversation_id,
                "turn_id":         turn_id,
                "speaker":         row[speaker_col] if speaker_col else None,
                "utterance":       utterance,
                "source_file":     str(path),
            })
    all_utts = pd.DataFrame(rows)
    if all_utts.empty:
        raise ValueError("没有读取到任何 utterance。请检查 transcript 文件结构。")
    print(f"[OK] 共读取 utterances: {len(all_utts)}")
    return all_utts


# ============================================================
# 10. 筛选 features
# ============================================================

def filter_features(all_utts):
    df = all_utts.copy()

    df["n_words"]              = df["utterance"].apply(count_words)
    df["has_identity_cue"]     = df["utterance"].apply(has_identity_cue)
    df["is_low_content_noise"] = df["utterance"].apply(is_low_content_noise)
    df["question_mark"]        = df["utterance"].apply(contains_question_mark)
    df["first_person"]         = df["utterance"].apply(contains_first_person)
    df["negation"]             = df["utterance"].apply(contains_negation)

    # 长度 + 身份线索 + 低信息量噪音过滤
    filtered = df[
        (df["n_words"] >= MIN_WORDS)
        & (df["n_words"] <= MAX_WORDS)
        & (~df["has_identity_cue"])
        & (~df["is_low_content_noise"])
    ].copy()

    filtered["features"]   = filtered["utterance"].apply(detect_features)
    filtered["n_features"] = filtered["features"].apply(len)

    # 所有命中特征的候选句
    feature_candidates = filtered[filtered["n_features"] > 0].copy()

    # ── 新增：内容词过滤 ───────────────────────────────────────
    # 用 FEATURE_PATTERNS 正则扣除 feature 表达式后，
    # 剩余内容词不足 MIN_CONTENT_WORDS 的句子过滤掉。
    feature_candidates["content_word_count"] = feature_candidates.apply(
        lambda row: count_residual_content_words(row["utterance"], row["features"]),
        axis=1,
    )
    n_before_content_filter = len(feature_candidates)
    feature_candidates = feature_candidates[
        feature_candidates["content_word_count"] >= MIN_CONTENT_WORDS
    ].copy()
    print(
        f"[INFO] 内容词过滤（>= {MIN_CONTENT_WORDS} 词）："
        f"过滤掉 {n_before_content_filter - len(feature_candidates)} 条，"
        f"剩余 {len(feature_candidates)} 条"
    )
    # ─────────────────────────────────────────────────────────

    # single-feature：主实验只用这一类
    single_feature = feature_candidates[feature_candidates["n_features"] == 1].copy()
    if not single_feature.empty:
        single_feature["feature_type"] = single_feature["features"].apply(lambda x: x[0])
    else:
        single_feature["feature_type"] = pd.Series(dtype="object")

    # mixed-feature：保存，但不进入主实验
    mixed_feature = feature_candidates[feature_candidates["n_features"] > 1].copy()
    if not mixed_feature.empty:
        mixed_feature["feature_combo"] = mixed_feature["features"].apply(lambda x: "+".join(x))
    else:
        mixed_feature["feature_combo"] = pd.Series(dtype="object")

    print(f"[INFO] 长度 + 身份线索 + 低信息量过滤后 utterances: {len(filtered)}")
    print(f"[INFO] 所有 feature candidates（含内容词过滤）: {len(feature_candidates)}")
    print(f"[INFO] single-feature candidates: {len(single_feature)}")
    print(f"[INFO] mixed-feature candidates: {len(mixed_feature)}")

    if not single_feature.empty:
        print("\n[INFO] Single-feature 候选数量，cap 前：")
        print(single_feature.groupby("feature_type").size().sort_values())

    return filtered, feature_candidates, single_feature, mixed_feature


# ============================================================
# 11. 每个 conversation 每类 feature 设置上限
# ============================================================

def cap_per_conversation(single_feature):
    if single_feature.empty:
        return single_feature.copy()
    if "feature_type" not in single_feature.columns:
        raise KeyError("single_feature 中没有 feature_type 列。")

    capped_parts = []
    for (feature_type, conversation_id), group in single_feature.groupby(
        ["feature_type", "conversation_id"]
    ):
        sampled = group.sample(
            n=min(len(group), MAX_PER_CONV_PER_FEATURE),
            random_state=RANDOM_SEED,
        ).copy()
        sampled["feature_type"]    = feature_type
        sampled["conversation_id"] = conversation_id
        capped_parts.append(sampled)

    capped = pd.concat(capped_parts, ignore_index=True)
    print(f"\n[INFO] Per-conversation cap 后 single-feature 候选数量：{len(capped)}")
    return capped


# ============================================================
# 11.5 前缀近重复去重（新增，在 cap 之后调用）
# ============================================================
# 见上方 deduplicate_by_prefix() 函数定义（4.3 节）


# ============================================================
# 12. 平衡样本
# ============================================================

def create_balanced_sample(single_feature_pool):
    if single_feature_pool.empty:
        print("[WARN] 输入为空，无法创建平衡样本。")
        return pd.DataFrame()

    if "feature_type" not in single_feature_pool.columns:
        raise KeyError("缺少 feature_type 列。")

    counts = (
        single_feature_pool
        .groupby("feature_type")
        .size()
        .sort_values(ascending=True)
    )
    print("\n[INFO] 平衡前各类候选数量：")
    print(counts)

    expected_features = [
        "hedge",
        "tag_question",
        "politeness_softened_disagreement",
    ]
    missing = [f for f in expected_features if f not in counts.index]
    if missing:
        print(f"\n[WARN] 以下 feature 没有候选样本：{missing}")
        return pd.DataFrame()

    min_n = counts.min()
    print(f"\n[INFO] 平衡样本每类数量：{min_n}")

    balanced = pd.concat([
        single_feature_pool[single_feature_pool["feature_type"] == f].sample(
            n=min_n, random_state=RANDOM_SEED
        )
        for f in expected_features
    ], ignore_index=True)

    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    balanced.insert(0, "sample_id", [f"S{i+1:05d}" for i in range(len(balanced))])

    keep_cols = [
        "sample_id", "conversation_id", "turn_id", "speaker", "utterance",
        "feature_type", "n_words", "content_word_count",
        "has_identity_cue", "is_low_content_noise",
        "question_mark", "first_person", "negation", "features", "source_file",
    ]
    keep_cols = [c for c in keep_cols if c in balanced.columns]
    return balanced[keep_cols]


# ============================================================
# 13. 输出文件
# ============================================================

def save_outputs(
    all_utts, filtered, feature_candidates,
    single_feature, mixed_feature,
    capped_single_feature, deduped_single_feature,
    balanced,
):
    all_utts.to_csv(OUTPUT_DIR / "01_all_utterances.csv", index=False)
    filtered.to_csv(OUTPUT_DIR / "02_length_identity_filtered.csv", index=False)
    feature_candidates.to_csv(OUTPUT_DIR / "03_all_feature_candidates.csv", index=False)
    single_feature.to_csv(OUTPUT_DIR / "04_single_feature_candidates.csv", index=False)
    mixed_feature.to_csv(OUTPUT_DIR / "05_mixed_feature_candidates_excluded.csv", index=False)
    capped_single_feature.to_csv(
        OUTPUT_DIR / "07_capped_single_feature_candidates.csv", index=False
    )
    deduped_single_feature.to_csv(
        OUTPUT_DIR / "07b_deduped_single_feature_candidates.csv", index=False
    )
    if not balanced.empty:
        balanced.to_csv(OUTPUT_DIR / "06_balanced_main_experiment_sample.csv", index=False)

    summary_rows = [
        {"stage": "all_utterances",                                      "count": len(all_utts)},
        {"stage": f"length_{MIN_WORDS}_{MAX_WORDS}_identity_filtered",   "count": len(filtered)},
        {"stage": "all_feature_candidates",                              "count": len(feature_candidates)},
        {"stage": "single_feature_before_cap",                           "count": len(single_feature)},
        {"stage": f"single_feature_after_cap_max_{MAX_PER_CONV_PER_FEATURE}", "count": len(capped_single_feature)},
        {"stage": f"single_feature_after_prefix_dedup_{PREFIX_DEDUP_WORDS}w", "count": len(deduped_single_feature)},
        {"stage": "mixed_feature_excluded",                              "count": len(mixed_feature)},
    ]
    if not balanced.empty:
        summary_rows.append({"stage": "balanced_main_experiment_sample", "count": len(balanced)})

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "00_summary_counts.csv", index=False)
    print(f"\n[OK] 输出文件已保存到：{OUTPUT_DIR}")


# ============================================================
# 14. 抽样查看
# ============================================================

def print_samples(single_feature, mixed_feature, deduped_single_feature):
    for label, df in [("cap 前", single_feature), ("去重后", deduped_single_feature)]:
        print(f"\n{'='*30}\nSingle-feature 样本预览，{label}\n{'='*30}")
        if df.empty:
            print("[WARN] 无样本。")
            continue
        for feature in sorted(df["feature_type"].unique()):
            print(f"\n--- {feature} ---")
            subset = df[df["feature_type"] == feature]
            for utt in subset.sample(n=min(8, len(subset)), random_state=RANDOM_SEED)["utterance"]:
                print(f"- {utt}")

    if not mixed_feature.empty:
        print(f"\n{'='*30}\nMixed-feature 样本预览\n{'='*30}")
        for _, row in mixed_feature.sample(n=min(8, len(mixed_feature)), random_state=RANDOM_SEED).iterrows():
            print(f"- [{row['feature_combo']}] {row['utterance']}")


# ============================================================
# 15. 主函数
# ============================================================

def main():
    unzip_candor()

    transcript_files = find_transcript_files()
    all_utts = load_all_utterances(transcript_files)

    filtered, feature_candidates, single_feature, mixed_feature = filter_features(all_utts)

    capped_single_feature  = cap_per_conversation(single_feature)

    # 新增步骤：前缀近重复去重
    deduped_single_feature = deduplicate_by_prefix(capped_single_feature)

    balanced = create_balanced_sample(deduped_single_feature)

    save_outputs(
        all_utts=all_utts,
        filtered=filtered,
        feature_candidates=feature_candidates,
        single_feature=single_feature,
        mixed_feature=mixed_feature,
        capped_single_feature=capped_single_feature,
        deduped_single_feature=deduped_single_feature,
        balanced=balanced,
    )

    print_samples(single_feature, mixed_feature, deduped_single_feature)
    print("\n[DONE] CANDOR feature 筛选完成。")


if __name__ == "__main__":
    main()
