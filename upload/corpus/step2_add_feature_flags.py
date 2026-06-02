import pandas as pd

from config import OUTPUT_DIR, TARGET_FEATURES

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- 1. Find input files ----------

if (OUTPUT_DIR / "02_length_identity_filtered.csv").exists():
    BASE_FILE = OUTPUT_DIR / "02_length_identity_filtered.csv"
elif (OUTPUT_DIR.parent / "02_length_identity_filtered.csv").exists():
    BASE_FILE = OUTPUT_DIR.parent / "02_length_identity_filtered.csv"
else:
    raise FileNotFoundError(
        "Cannot find 02_length_identity_filtered.csv in either root folder or outputs folder."
    )

CANDIDATE_FILE = OUTPUT_DIR / "03_all_feature_candidates.csv"

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        "Cannot find outputs/03_all_feature_candidates.csv. "
        "Please make sure you already ran filter_candor_features.py."
    )

OUTPUT_FILE = OUTPUT_DIR / "02_length_identity_filtered_with_feature_flags.csv"


# ---------- 2. Helper functions ----------

def check_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )


def choose_key_columns(base, cand):
    """
    Prefer stable utterance-level keys.
    """
    possible_keys = [
        ["conversation_id", "turn_id", "speaker", "utterance"],
        ["conversation_id", "turn_id", "utterance"],
        ["conversation_id", "speaker", "utterance"],
        ["source_file", "turn_id", "speaker", "utterance"],
        ["utterance"],
    ]

    for keys in possible_keys:
        if all(k in base.columns for k in keys) and all(k in cand.columns for k in keys):
            return keys

    raise ValueError(
        "Cannot find shared key columns between base file and candidate file.\n"
        f"Base columns:\n{list(base.columns)}\n\n"
        f"Candidate columns:\n{list(cand.columns)}"
    )


def clean_feature_name(x):
    if pd.isna(x):
        return None

    x = str(x).strip().lower()

    # 修复：删除 "hedge": "hedge" 等自映射条目。
    # aliases.get(x, x) 已经能处理"不在字典里就原样返回"的情况，
    # 显式写出自映射既冗余也会误导读者以为存在特殊转换逻辑。
    aliases = {
        "hedges": "hedge",

        "tag_questions": "tag_question",
        "tag question":  "tag_question",

        "politeness":                         "politeness_softened_disagreement",
        "softened_disagreement":              "politeness_softened_disagreement",
        "politeness_softened":                "politeness_softened_disagreement",
        "politeness / softened disagreement": "politeness_softened_disagreement",
    }

    return aliases.get(x, x)


# ---------- 3. Load files ----------

base = pd.read_csv(BASE_FILE)
cand = pd.read_csv(CANDIDATE_FILE)

print("Loaded base file:", BASE_FILE)
print("Base shape:", base.shape)

print("\nLoaded candidate file:", CANDIDATE_FILE)
print("Candidate shape:", cand.shape)

check_columns(base, ["utterance"], "base file")
check_columns(cand, ["utterance"], "candidate file")

key_cols = choose_key_columns(base, cand)

print("\nUsing key columns:")
print(key_cols)


# ---------- 4. Initialize feature flags in base ----------

for feature in TARGET_FEATURES:
    base[feature] = False

# ---------- 4.5 Normalize candidate file: parse 'features' list column ----------

import ast

if "feature_type" not in cand.columns and "features" in cand.columns:
    def parse_feature_list(x):
        if pd.isna(x) or str(x).strip() in ("[]", ""):
            return []
        try:
            val = ast.literal_eval(str(x))
            return val if isinstance(val, list) else []
        except Exception:
            return []

    cand = cand.copy()
    cand["feature_type"] = cand["features"].apply(parse_feature_list)
    cand = cand.explode("feature_type")
    cand = cand[cand["feature_type"].notna() & (cand["feature_type"] != "")].reset_index(drop=True)

    print("\nParsed feature_type from 'features' column.")
    print(cand["feature_type"].value_counts(dropna=False))

# ---------- 5. Build feature flags from candidate file ----------

if "feature_type" in cand.columns:
    cand = cand.copy()
    cand["feature_type"] = cand["feature_type"].apply(clean_feature_name)

    cand = cand[cand["feature_type"].isin(TARGET_FEATURES)].copy()

    print("\nCandidate feature_type counts:")
    print(cand["feature_type"].value_counts(dropna=False))

    # 修复：用 pd.crosstab 替代 pivot_table。
    # 原来的做法需要先 drop_duplicates + assign(value=True) 再透视，
    # 对大规模数据（百万级行）会引入不必要的中间副本和内存开销。
    # crosstab 直接聚合计数，再 > 0 转为布尔值，语义完全等价且更简洁。
    cand_flags = pd.crosstab(
        index=[cand[k] for k in key_cols],
        columns=cand["feature_type"],
    ).reset_index()

    # 清除 columns 轴的 "feature_type" 名称标签，并确保所有列名为纯字符串
    cand_flags.columns.name = None
    cand_flags.columns = [str(c) for c in cand_flags.columns]

    # 将计数 (0/1/2/…) 转换为布尔标记；不在候选中出现的 feature 补 False
    for feature in TARGET_FEATURES:
        if feature in cand_flags.columns:
            cand_flags[feature] = cand_flags[feature] > 0
        else:
            cand_flags[feature] = False

    cand_flags = cand_flags[key_cols + TARGET_FEATURES]

else:
    # If 03_all_feature_candidates.csv already has feature flag columns
    existing_flags = [f for f in TARGET_FEATURES if f in cand.columns]

    if not existing_flags:
        raise ValueError(
            "Candidate file has neither feature_type column nor feature flag columns.\n"
            f"Available columns:\n{list(cand.columns)}"
        )

    cand_flags = cand[key_cols + existing_flags].copy()

    for feature in TARGET_FEATURES:
        if feature not in cand_flags.columns:
            cand_flags[feature] = False

    cand_flags = (
        cand_flags
        .groupby(key_cols, as_index=False)[TARGET_FEATURES]
        .max()
    )


print("\nCandidate flags shape:", cand_flags.shape)
print("Candidate flag counts:")
print(cand_flags[TARGET_FEATURES].sum())


# ---------- 6. Merge flags back into 02 base file ----------

base_no_flags = base.drop(columns=TARGET_FEATURES, errors="ignore")

merged = base_no_flags.merge(
    cand_flags,
    on=key_cols,
    how="left",
)

for feature in TARGET_FEATURES:
    merged[feature] = merged[feature].fillna(False).astype(bool)


# ---------- 7. Add optional summary columns ----------

merged["n_target_features"] = merged[TARGET_FEATURES].sum(axis=1)

def make_feature_list(row):
    feats = [f for f in TARGET_FEATURES if row[f]]
    return ";".join(feats)

merged["target_features"] = merged.apply(make_feature_list, axis=1)


# ---------- 8. Save ----------

merged.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("Output shape:", merged.shape)

print("\nFinal feature flag counts:")
print(merged[TARGET_FEATURES].sum())

print("\nNumber of utterances with any target feature:")
print((merged["n_target_features"] > 0).sum())

print("\nNumber of utterances with no target feature:")
print((merged["n_target_features"] == 0).sum())
