"""
新闻/股吧文本数据处理模块
负责：加载股吧评论 CSV → 每日聚合标题文本 → 与交易日序列对齐
输出供 precompute_bert.py 使用，也可直接获取每日文本列表。

股吧 CSV 字段（guba_crawler.py 输出）：
    reply_num, read_num, pl(标题), time1, publish_time, author
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


# publish_time 可能有多种格式，依次尝试
_DATE_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"]


def _parse_dates(series: pd.Series) -> pd.Series:
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


def load_news(
    csv_path: str,
    text_col: str = "pl",
    date_col: str = "publish_time",
) -> pd.DataFrame:
    """
    加载股吧评论 CSV，返回含 date(date) + text(str) 两列的 DataFrame。
    会过滤掉文本为空的行。
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if text_col not in df.columns:
        candidates = [c for c in df.columns if "title" in c.lower() or "标题" in c or "pl" in c]
        if not candidates:
            raise ValueError(f"找不到文本列，当前列名：{df.columns.tolist()}")
        text_col = candidates[0]

    df["text"] = df[text_col].str.strip()
    df = df[df["text"] != ""]

    df["date"] = _parse_dates(df[date_col]).dt.date
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "text", "reply_num", "read_num"]].copy()


def aggregate_daily_text(
    news_df: pd.DataFrame,
    max_titles: int = 5,
    sep: str = "。",
) -> pd.DataFrame:
    """
    每个交易日取热度最高（reply_num 降序）的 max_titles 条标题，
    用 sep 拼接成单条文本。
    返回 DataFrame：date | daily_text
    """
    news_df = news_df.copy()
    news_df["reply_num"] = pd.to_numeric(news_df["reply_num"], errors="coerce").fillna(0)

    def _agg(g: pd.DataFrame) -> str:
        top = g.nlargest(max_titles, "reply_num")["text"].tolist()
        return sep.join(top)

    daily = news_df.groupby("date").apply(_agg).reset_index()
    daily.columns = ["date", "daily_text"]
    return daily


def align_to_trading_days(
    daily_text: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    empty_text: str = "",
) -> pd.Series:
    """
    将每日文本对齐到 trade_dates（左侧填充：找最近的过去一条，无则用 empty_text）。
    返回 Series，index = trade_dates，值 = 文本字符串。
    """
    text_map = dict(zip(daily_text["date"], daily_text["daily_text"]))
    result = []
    for d in trade_dates:
        # 向前找最近有文本的日期（不超过7天）
        for offset in range(8):
            candidate = d - pd.Timedelta(days=offset)
            if candidate in text_map:
                result.append(text_map[candidate])
                break
        else:
            result.append(empty_text)
    return pd.Series(result, index=trade_dates)


def build_daily_text_series(
    comments_csv: str,
    trade_dates: pd.DatetimeIndex,
    max_titles: int = 5,
) -> pd.Series:
    """
    一站式接口：CSV → 每日聚合文本 → 对齐交易日。
    返回 Series[str]，index = trade_dates。
    """
    news_df   = load_news(comments_csv)
    daily     = aggregate_daily_text(news_df, max_titles=max_titles)
    aligned   = align_to_trading_days(daily, trade_dates)
    return aligned
