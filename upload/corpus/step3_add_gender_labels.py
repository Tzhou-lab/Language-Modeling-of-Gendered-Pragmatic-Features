from pathlib import Path
import pandas as pd

from config import (
    EXTRACT_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
)

# ============================================================
# 1. 脚本级配置（非共享，保留在此处）
# ============================================================

# 是否尝试创建 feature_type × speaker_gender 平衡样本
CREATE_GENDER_BALANCED_SAMPLE = True

# 如果 True，只使用 male / female；other / missing 不参与平衡抽样，但仍会被报告
USE_BINARY_ONLY_FOR_BALANCE = True


# ============================================================
# 2. 基础检查
# ============================================================

def check_paths():
    if not EXTRACT_DIR.exists():
        raise FileNotFoundError(
            f"没有找到 {EXTRACT_DIR}。请先运行 filter_candor_features.py 解压 CANDOR.zip。"
        )

    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"没有找到 {OUTPUT_DIR}。请先运行 filter_candor_features.py 生成 outputs。"
        )


# ============================================================
# 3. 推断 conversation_id
# ============================================================

def infer_conversation_id_from_survey(survey_path):
    """
    survey.csv 通常位于：
    conversation_folder/survey.csv

    所以 conversation_id 通常是 survey.csv 的父文件夹名。
    """
    return survey_path.parent.name


# ============================================================
# 4. 自动识别 survey.csv 中的 speaker id 列和 sex/gender 列
# ============================================================

def find_speaker_id_column(df):
    """
    CANDOR survey.csv 中可能出现的 speaker id 列名。
    你之前检查过，常见是 id。
    """
    candidates = [
        "id",
        "speaker",
        "speaker_id",
        "user_id",
        "participant_id",
        "participant",
        "channel",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def find_gender_column(df):
    """
    自动寻找 sex/gender 标签列。
    CANDOR 中你之前看到的是 sex。
    """
    candidates = [
        "sex",
        "gender",
        "Gender",
        "Sex",
        "participant_gender",
        "speaker_gender",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


# ============================================================
# 5. gender/sex 标签标准化
# ============================================================

def normalize_gender_label(value):
    """
    把 survey.csv 中的 sex/gender 标签统一成：
    male / female / other / missing

    注意：
    这里不假设 CANDOR 的 sex 一定是二元完整标签。
    """
    if pd.isna(value):
        return "missing"

    x = str(value).strip().lower()

    if x in ["", "nan", "none", "null", "na", "n/a", "missing"]:
        return "missing"

    # 常见 male 标签
    if x in ["m", "male", "man", "boy"]:
        return "male"

    # 常见 female 标签
    if x in ["f", "female", "woman", "girl"]:
        return "female"

    # 可能的其他标签
    if x in [
        "other",
        "non-binary",
        "nonbinary",
        "nb",
        "genderqueer",
        "prefer not to say",
        "prefer not to answer",
        "unknown",
    ]:
        return "other"

    # 如果是数字编码，先做保守处理
    # 具体编码如果你之后确认了，可以在这里改
    if x in ["1"]:
        return "male"
    if x in ["2"]:
        return "female"

    return "other"


def normalize_speaker_id(value):
    """
    统一 speaker id 格式，避免 1 和 1.0、字符串和数字对不上。

    修复：原来用 re.fullmatch(r"\\d+\\.0", x) 配合切片，只能处理恰好一位小数的情况，
    遇到 1.00、1.000 或其他精度时会失效。
    改用数值转换路径：任意浮点字符串 -> float -> int -> str，可安全归一化所有情况。
    """
    if pd.isna(value):
        return ""

    x = str(value).strip()

    try:
        # "1.0"、"1.00"、"1" 都会安全收敛为 "1"
        return str(int(float(x)))
    except ValueError:
        # 纯字母字符串（如 "user_A"）直接原样返回
        return x


# ============================================================
# 6. 读取所有 survey.csv，构建 speaker metadata 表
# ============================================================

def load_speaker_metadata():
    # 修复：用 glob("*/survey.csv") 替代 rglob("survey.csv")。
    # CANDOR 的 survey 文件固定位于 conversation_folder/survey.csv（一级目录），
    # 递归的 rglob 在大型语料库中会遍历所有子目录，I/O 开销远大于仅扫描一级。
    survey_files = list(EXTRACT_DIR.glob("*/survey.csv"))

    if not survey_files:
        raise FileNotFoundError(
            f"没有在 {EXTRACT_DIR} 中找到 survey.csv。请检查 CANDOR 解压结构。"
        )

    print(f"[OK] 找到 {len(survey_files)} 个 survey.csv 文件")

    rows = []

    skipped_no_id = 0
    skipped_no_gender = 0

    for survey_path in survey_files:
        conversation_id = infer_conversation_id_from_survey(survey_path)

        try:
            df = pd.read_csv(survey_path)
        except Exception as e:
            print(f"[WARN] 无法读取 {survey_path}: {e}")
            continue

        speaker_id_col = find_speaker_id_column(df)
        gender_col = find_gender_column(df)

        if speaker_id_col is None:
            skipped_no_id += 1
            print(f"[WARN] 找不到 speaker id 列，跳过：{survey_path}")
            print(f"       columns = {df.columns.tolist()}")
            continue

        if gender_col is None:
            skipped_no_gender += 1
            print(f"[WARN] 找不到 sex/gender 列，跳过：{survey_path}")
            print(f"       columns = {df.columns.tolist()}")
            continue

        # 修复：用 itertuples(index=False) 替代 iterrows()。
        # iterrows() 会将每行转为 Series 并可能隐式改变数据类型（如 int -> float），
        # itertuples 保留原始类型，且速度通常快几十倍。
        # 用 getattr(row, col) 安全访问列值（列名均为合法 Python 标识符）。
        for row in df.itertuples(index=False):
            raw_speaker = getattr(row, speaker_id_col)
            raw_gender  = getattr(row, gender_col)

            speaker_id  = normalize_speaker_id(raw_speaker)
            gender_norm = normalize_gender_label(raw_gender)

            if not speaker_id:
                continue

            rows.append({
                "conversation_id":     str(conversation_id),
                "speaker":             speaker_id,
                "speaker_gender_raw":  raw_gender,
                "speaker_gender":      gender_norm,
                "survey_speaker_id_col": speaker_id_col,
                "survey_gender_col":   gender_col,
                "survey_file":         str(survey_path),
            })

    speaker_meta = pd.DataFrame(rows)

    if speaker_meta.empty:
        raise ValueError("没有成功读取任何 speaker gender metadata。")

    # 去重，避免同一个 speaker 重复
    speaker_meta = speaker_meta.drop_duplicates(
        subset=["conversation_id", "speaker"],
        keep="first"
    )

    print(f"[OK] 成功读取 speaker metadata: {len(speaker_meta)} 行")
    print(f"[INFO] skipped_no_id: {skipped_no_id}")
    print(f"[INFO] skipped_no_gender: {skipped_no_gender}")

    print("\n[INFO] speaker gender 分布，survey metadata 总体：")
    print(speaker_meta["speaker_gender"].value_counts(dropna=False))

    return speaker_meta


# ============================================================
# 7. 合并 gender label 到现有 outputs
# ============================================================

def merge_gender_into_file(input_filename, output_filename, speaker_meta):
    input_path = OUTPUT_DIR / input_filename

    if not input_path.exists():
        print(f"[WARN] 找不到 {input_path}，跳过。")
        return None

    df = pd.read_csv(input_path)

    if "conversation_id" not in df.columns or "speaker" not in df.columns:
        print(f"[WARN] {input_filename} 缺少 conversation_id 或 speaker 列，无法合并 gender。")
        print(f"       columns = {df.columns.tolist()}")
        return None

    df = df.copy()
    df["conversation_id"] = df["conversation_id"].astype(str)
    df["speaker"] = df["speaker"].apply(normalize_speaker_id)

    merged = df.merge(
        speaker_meta[
            [
                "conversation_id",
                "speaker",
                "speaker_gender",
                "speaker_gender_raw",
                "survey_speaker_id_col",
                "survey_gender_col",
            ]
        ],
        on=["conversation_id", "speaker"],
        how="left",
    )

    merged["speaker_gender"]     = merged["speaker_gender"].fillna("missing")
    merged["speaker_gender_raw"] = merged["speaker_gender_raw"].fillna("missing")

    output_path = OUTPUT_DIR / output_filename
    merged.to_csv(output_path, index=False)

    matched = (merged["speaker_gender"] != "missing").sum()
    total   = len(merged)

    print(f"\n[OK] 已输出：{output_filename}")
    print(f"     rows: {total}")
    print(f"     matched non-missing gender: {matched}")
    print(f"     missing gender: {total - matched}")

    if "feature_type" in merged.columns:
        print("\n[INFO] feature_type × speaker_gender 分布：")
        tab = pd.crosstab(
            merged["feature_type"],
            merged["speaker_gender"],
            margins=True
        )
        print(tab)

    return merged


# ============================================================
# 8. 输出 gender distribution tables
# ============================================================

def save_gender_distribution(df, name_prefix):
    if df is None or df.empty:
        return

    if "feature_type" not in df.columns or "speaker_gender" not in df.columns:
        return

    # 计数表
    count_table = (
        df.groupby(["feature_type", "speaker_gender"])
        .size()
        .reset_index(name="count")
        .sort_values(["feature_type", "speaker_gender"])
    )

    count_table.to_csv(
        OUTPUT_DIR / f"{name_prefix}_gender_distribution_counts.csv",
        index=False
    )

    # 宽表
    wide_table = pd.crosstab(
        df["feature_type"],
        df["speaker_gender"],
        margins=True
    )

    wide_table.to_csv(
        OUTPUT_DIR / f"{name_prefix}_gender_distribution_crosstab.csv"
    )

    print(f"[OK] 已输出 gender distribution: {name_prefix}")


# ============================================================
# 9. 尝试创建 feature_type × speaker_gender 平衡样本
# ============================================================

def create_feature_gender_balanced_sample(df):
    """
    在已经合并 speaker_gender 的样本中，尝试按 feature_type × speaker_gender 平衡。

    如果 USE_BINARY_ONLY_FOR_BALANCE = True:
    只使用 male/female。
    """
    if df is None or df.empty:
        print("[WARN] 输入为空，无法创建 gender-balanced sample。")
        return pd.DataFrame()

    required_cols = ["feature_type", "speaker_gender"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARN] 缺少 {col}，无法创建 gender-balanced sample。")
            return pd.DataFrame()

    work = df.copy()

    if USE_BINARY_ONLY_FOR_BALANCE:
        work = work[work["speaker_gender"].isin(["male", "female"])].copy()

    if work.empty:
        print("[WARN] 没有可用于 gender-balanced sampling 的 male/female 样本。")
        return pd.DataFrame()

    counts = (
        work.groupby(["feature_type", "speaker_gender"])
        .size()
        .reset_index(name="count")
        .sort_values(["feature_type", "speaker_gender"])
    )

    print("\n[INFO] feature_type × speaker_gender 候选数量：")
    print(counts)

    expected_features = [
        "hedge",
        "tag_question",
        "politeness_softened_disagreement",
    ]

    expected_genders = ["female", "male"] if USE_BINARY_ONLY_FOR_BALANCE else sorted(
        work["speaker_gender"].unique()
    )

    # 检查每个 cell 是否都有样本
    missing_cells = []

    for feature in expected_features:
        for gender in expected_genders:
            cell_n = len(
                work[
                    (work["feature_type"] == feature)
                    & (work["speaker_gender"] == gender)
                ]
            )
            if cell_n == 0:
                missing_cells.append((feature, gender))

    if missing_cells:
        print("\n[WARN] 以下 feature × gender cell 没有样本，无法严格平衡：")
        for cell in missing_cells:
            print("      ", cell)
        return pd.DataFrame()

    min_n = (
        work.groupby(["feature_type", "speaker_gender"])
        .size()
        .min()
    )

    print(f"\n[INFO] feature_type × speaker_gender 平衡后每个 cell 数量：{min_n}")

    balanced_parts = []

    for feature in expected_features:
        for gender in expected_genders:
            part = work[
                (work["feature_type"] == feature)
                & (work["speaker_gender"] == gender)
            ].sample(
                n=min_n,
                random_state=RANDOM_SEED,
            )

            balanced_parts.append(part)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # 重新生成 sample_id
    if "sample_id" in balanced.columns:
        balanced = balanced.drop(columns=["sample_id"])

    balanced.insert(0, "sample_id", [f"GB{i+1:05d}" for i in range(len(balanced))])

    output_path = OUTPUT_DIR / "08_gender_balanced_main_experiment_sample.csv"
    balanced.to_csv(output_path, index=False)

    print(f"\n[OK] 已输出 gender-balanced sample：{output_path}")
    print(f"     total rows: {len(balanced)}")

    print("\n[INFO] gender-balanced sample 分布：")
    print(pd.crosstab(balanced["feature_type"], balanced["speaker_gender"], margins=True))

    return balanced


# ============================================================
# 10. 主函数
# ============================================================

def main():
    check_paths()

    speaker_meta = load_speaker_metadata()

    # 把 gender label 合并到现有几个关键文件
    merged_single = merge_gender_into_file(
        input_filename="04_single_feature_candidates.csv",
        output_filename="04_single_feature_candidates_with_gender.csv",
        speaker_meta=speaker_meta,
    )

    merged_capped = merge_gender_into_file(
        input_filename="07_capped_single_feature_candidates.csv",
        output_filename="07_capped_single_feature_candidates_with_gender.csv",
        speaker_meta=speaker_meta,
    )

    merged_main = merge_gender_into_file(
        input_filename="06_balanced_main_experiment_sample.csv",
        output_filename="06_balanced_main_experiment_sample_with_gender.csv",
        speaker_meta=speaker_meta,
    )

    # 保存 gender distribution
    save_gender_distribution(
        merged_single,
        "04_single_feature_candidates"
    )

    save_gender_distribution(
        merged_capped,
        "07_capped_single_feature_candidates"
    )

    save_gender_distribution(
        merged_main,
        "06_balanced_main_experiment_sample"
    )

    # 尝试创建 feature × gender 都平衡的主实验样本
    if CREATE_GENDER_BALANCED_SAMPLE:
        # 优先从 capped single-feature candidates 中创建，
        # 因为它还没有被 feature-only balance 限制，更适合重新按 gender 平衡。
        gender_balanced = create_feature_gender_balanced_sample(merged_capped)

        if gender_balanced.empty:
            print(
                "\n[INFO] 无法创建严格 gender-balanced sample。"
                "可以使用 06_balanced_main_experiment_sample_with_gender.csv，"
                "并在论文中报告 gender distribution。"
            )

    print("\n[DONE] gender label 合并完成。")


if __name__ == "__main__":
    main()
