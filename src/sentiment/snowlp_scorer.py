"""
情感打分模块（支路四）
====================
使用 SnowNLP 对股吧评论做中文情感分析，
输出日级别情感特征，与技术指标对齐后作为时序输入的一部分。

输出特征（7维，每交易日一行）：
  sentiment_mean     日均情感得分 [0,1]（>0.5 偏正向）
  sentiment_std      日内情感标准差（反映分歧程度）
  positive_ratio     当日正向评论占比（得分 > 0.6）
  negative_ratio     当日负向评论占比（得分 < 0.4）
  comment_count      当日评论数量（取 log1p 缩放）
  sentiment_momentum 与前一交易日均值之差（情感变化速度）
  sentiment_ma5      5日情感均值（平滑短期噪声）

用法：
    python -m src.sentiment.snowlp_scorer
"""

from __future__ import annotations

import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 文本清洗
# ─────────────────────────────────────────────

# 股评里常见的无意义噪声
_NOISE_PATTERN = re.compile(
    r"(https?://\S+)"           # URL
    r"|(\[.*?\])"               # 表情符 [哭泣]
    r"|([^\u4e00-\u9fa5\w，。！？、；：""''（）【】])"  # 非中文非字母标点
    , re.UNICODE
)

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = _NOISE_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# 单条打分
# ─────────────────────────────────────────────

def score_text(text: str) -> float:
    """
    返回 [0, 1] 情感分，> 0.5 偏正向。
    文本过短或解析失败时返回 0.5（中性）。
    """
    text = clean_text(text)
    if len(text) < 2:
        return 0.5
    try:
        from snownlp import SnowNLP
        return float(SnowNLP(text).sentiments)
    except Exception:
        return 0.5


# ─────────────────────────────────────────────
# 批量打分（带进度条）
# ─────────────────────────────────────────────

def batch_score(texts: list[str], batch_size: int = 200) -> list[float]:
    """
    批量对文本列表打分，每 batch_size 条打印一次进度。
    """
    scores = []
    total = len(texts)
    for i, text in enumerate(texts):
        scores.append(score_text(text))
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            print(f"\r  打分进度: {i+1}/{total}", end="", flush=True)
    print()
    return scores


# ─────────────────────────────────────────────
# 日级别特征聚合
# ─────────────────────────────────────────────

def aggregate_daily_sentiment(
    comments_csv: str,
    text_col: str = "pl",
    date_col: str = "publish_time",
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    从股吧评论 CSV 计算日级别 7 维情感特征。

    Args:
        comments_csv: 股吧评论文件路径
        text_col:     评论文本列名（默认 "pl"）
        date_col:     发布时间列名（默认 "publish_time"）
        output_csv:   若指定则保存到文件

    Returns:
        DataFrame，每行是一个交易日，列为 7 维情感特征 + date
    """
    print(f"[1/3] 读取评论数据: {comments_csv}")
    df = pd.read_csv(comments_csv)
    print(f"      共 {len(df)} 条评论")

    df["date"] = pd.to_datetime(df[date_col]).dt.normalize()

    print("[2/3] 情感打分中（首次运行较慢，约 1~3 分钟）...")
    df["sentiment"] = batch_score(df[text_col].tolist())

    print("[3/3] 聚合为日级别特征...")
    def _agg(g: pd.Series) -> pd.Series:
        return pd.Series({
            "sentiment_mean":    g.mean(),
            "sentiment_std":     g.std(ddof=0),
            "positive_ratio":    (g > 0.6).mean(),
            "negative_ratio":    (g < 0.4).mean(),
            "comment_count":     np.log1p(len(g)),   # log 缩放，避免量纲差异
        })

    daily = df.groupby("date")["sentiment"].apply(_agg).reset_index()

    # 情感动量（当日均值 - 前一日均值）
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["sentiment_momentum"] = daily["sentiment_mean"].diff().fillna(0.0)

    # 5日情感均值（平滑短期噪声）
    daily["sentiment_ma5"] = (
        daily["sentiment_mean"].rolling(5, min_periods=1).mean()
    )

    # 填充 NaN
    daily = daily.fillna(0.0)

    if output_csv:
        daily.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] 日级别情感特征已保存 -> {output_csv}  ({len(daily)} 天, 7维)")

    return daily


# ─────────────────────────────────────────────
# 与价格/技术指标数据对齐
# ─────────────────────────────────────────────

SENTIMENT_COLS = [
    "sentiment_mean",
    "sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "comment_count",
    "sentiment_momentum",
    "sentiment_ma5",
]

def merge_with_indicators(
    indicators_df: pd.DataFrame,
    sentiment_csv: str,
) -> pd.DataFrame:
    """
    将情感特征按交易日对齐并合并到技术指标 DataFrame 中。

    非交易日（如节假日）的情感数据用前向填充（forward fill），
    再用 0.5 填充开头的缺失值（冷启动期取中性）。

    Args:
        indicators_df: 技术指标 DataFrame，必须有 "date" 列
        sentiment_csv: aggregate_daily_sentiment 输出的文件路径

    Returns:
        合并后的 DataFrame（行数与 indicators_df 相同）
    """
    sent = pd.read_csv(sentiment_csv, parse_dates=["date"])

    merged = indicators_df.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.merge(sent[["date"] + SENTIMENT_COLS], on="date", how="left")

    # 交易日可能没有对应情感数据（如节假日刚过完）→ 前向填充
    merged[SENTIMENT_COLS] = merged[SENTIMENT_COLS].ffill()

    # 冷启动期（最早几天没有历史评论）→ 填中性值
    neutral_defaults = {
        "sentiment_mean":     0.5,
        "sentiment_std":      0.0,
        "positive_ratio":     0.5,
        "negative_ratio":     0.5,
        "comment_count":      0.0,
        "sentiment_momentum": 0.0,
        "sentiment_ma5":      0.5,
    }
    for col, val in neutral_defaults.items():
        merged[col] = merged[col].fillna(val)

    return merged


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    comments_csv  = "data/raw/贵州茅台600519_股吧评论.csv"
    sentiment_csv = "data/processed/moutai_daily_sentiment.csv"
    indicator_csv = "data/processed/moutai_technical_indicators.csv"

    # 1. 计算日级别情感特征
    daily = aggregate_daily_sentiment(
        comments_csv=comments_csv,
        output_csv=sentiment_csv,
    )

    print("\n--- 情感特征预览（前5行）---")
    print(daily.head().to_string(index=False))
    print(f"\n情感均值范围: [{daily['sentiment_mean'].min():.3f}, "
          f"{daily['sentiment_mean'].max():.3f}]")
    print(f"正向评论最高占比: {daily['positive_ratio'].max():.1%}")
    print(f"负向评论最高占比: {daily['negative_ratio'].max():.1%}")

    # 2. 若技术指标文件存在，测试对齐合并
    if os.path.exists(indicator_csv):
        ind = pd.read_csv(indicator_csv, parse_dates=["date"])
        merged = merge_with_indicators(ind, sentiment_csv)
        print(f"\n合并后维度: {merged.shape}  "
              f"（技术指标 {len(ind.columns)} 列 + 情感 {len(SENTIMENT_COLS)} 列）")
        print("缺失值检查:", merged[SENTIMENT_COLS].isnull().sum().sum(), "个")
