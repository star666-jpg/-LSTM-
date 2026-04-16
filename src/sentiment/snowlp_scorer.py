"""
情感打分模块（支路四）
使用 SnowNLP 对股吧评论/新闻标题进行中文情感评分，
输出 [0, 1] 浮点数（>0.5 为正向情感），
再聚合为日级别情感得分作为时序特征。
"""

from __future__ import annotations

import pandas as pd
from snownlp import SnowNLP


def score_text(text: str) -> float:
    try:
        return SnowNLP(str(text)).sentiments
    except Exception:
        return 0.5  # 无法解析时取中性值


def aggregate_daily_sentiment(
    comments_csv: str,
    text_col: str = "pl",
    date_col: str = "publish_time",
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    从评论 CSV 计算日级别情感均值。

    Args:
        comments_csv: 股吧评论文件路径
        text_col:     帖子正文列名
        date_col:     发布时间列名
        output_csv:   若指定则同时保存到文件

    Returns:
        DataFrame with columns: date, sentiment_mean, sentiment_std, comment_count
    """
    df = pd.read_csv(comments_csv)
    df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
    df["sentiment"] = df[text_col].astype(str).apply(score_text)

    daily = (
        df.groupby("date")["sentiment"]
        .agg(sentiment_mean="mean", sentiment_std="std", comment_count="count")
        .reset_index()
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)

    if output_csv:
        daily.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] 日级别情感得分已保存 -> {output_csv}  ({len(daily)} 天)")

    return daily


if __name__ == "__main__":
    aggregate_daily_sentiment(
        comments_csv="data/raw/贵州茅台600519_股吧评论.csv",
        output_csv="data/processed/moutai_daily_sentiment.csv",
    )
