"""
情感打分模块单元测试
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import tempfile

from src.sentiment.snowlp_scorer import (
    clean_text,
    score_text,
    aggregate_daily_sentiment,
    merge_with_indicators,
    SENTIMENT_COLS,
)


# ─────────────────────────────────────────────
# 文本清洗
# ─────────────────────────────────────────────

def test_clean_removes_url():
    assert "http" not in clean_text("看看这个 https://xxx.com 分析")

def test_clean_removes_emoji():
    assert "[" not in clean_text("茅台大涨[鼓掌][鼓掌]")

def test_clean_short_text():
    assert clean_text("  ") == ""

def test_clean_normal_text():
    result = clean_text("茅台今天涨停，明天继续看好！")
    assert len(result) > 0


# ─────────────────────────────────────────────
# 单条打分
# ─────────────────────────────────────────────

def test_score_positive():
    score = score_text("茅台真的太厉害了，大涨！强烈看好！")
    assert 0.0 <= score <= 1.0

def test_score_negative():
    score = score_text("亏死了，跌停，割肉跑路！")
    assert 0.0 <= score <= 1.0

def test_score_empty_returns_neutral():
    assert score_text("") == 0.5
    assert score_text("  ") == 0.5

def test_score_range():
    texts = ["好", "坏", "涨", "跌", "不知道", ""]
    for t in texts:
        s = score_text(t)
        assert 0.0 <= s <= 1.0, f"score_text('{t}') = {s} 超出范围"


# ─────────────────────────────────────────────
# 日级别聚合（用构造数据，不依赖真实文件）
# ─────────────────────────────────────────────

def _make_comments_csv(tmp_dir: str) -> str:
    """构造一个最小可用的评论 CSV"""
    data = {
        "pl": [
            "茅台大涨，买买买！",
            "今天跌了好多，心痛",
            "继续持有，长期看好",
            "割肉了，亏了好多",
            "明天应该会涨",
        ],
        "publish_time": [
            "2026-01-05 09:00:00",
            "2026-01-05 10:00:00",
            "2026-01-06 09:30:00",
            "2026-01-06 14:00:00",
            "2026-01-07 11:00:00",
        ],
    }
    path = os.path.join(tmp_dir, "test_comments.csv")
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_aggregate_returns_dataframe():
    with tempfile.TemporaryDirectory() as tmp:
        csv = _make_comments_csv(tmp)
        result = aggregate_daily_sentiment(csv)
        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns


def test_aggregate_output_columns():
    with tempfile.TemporaryDirectory() as tmp:
        csv = _make_comments_csv(tmp)
        result = aggregate_daily_sentiment(csv)
        for col in SENTIMENT_COLS:
            assert col in result.columns, f"缺少列: {col}"


def test_aggregate_three_days():
    with tempfile.TemporaryDirectory() as tmp:
        csv = _make_comments_csv(tmp)
        result = aggregate_daily_sentiment(csv)
        assert len(result) == 3   # 2026-01-05, 01-06, 01-07


def test_aggregate_sentiment_range():
    with tempfile.TemporaryDirectory() as tmp:
        csv = _make_comments_csv(tmp)
        result = aggregate_daily_sentiment(csv)
        assert result["sentiment_mean"].between(0, 1).all()
        assert result["positive_ratio"].between(0, 1).all()
        assert result["negative_ratio"].between(0, 1).all()


def test_aggregate_no_nan():
    with tempfile.TemporaryDirectory() as tmp:
        csv = _make_comments_csv(tmp)
        result = aggregate_daily_sentiment(csv)
        assert not result[SENTIMENT_COLS].isnull().any().any()


def test_aggregate_saves_csv():
    with tempfile.TemporaryDirectory() as tmp:
        csv = _make_comments_csv(tmp)
        out = os.path.join(tmp, "out.csv")
        aggregate_daily_sentiment(csv, output_csv=out)
        assert os.path.exists(out)


# ─────────────────────────────────────────────
# 与技术指标对齐
# ─────────────────────────────────────────────

def _make_sentiment_csv(tmp_dir: str) -> str:
    data = {
        "date": ["2026-01-05", "2026-01-06", "2026-01-07"],
        "sentiment_mean":     [0.7, 0.3, 0.6],
        "sentiment_std":      [0.1, 0.2, 0.1],
        "positive_ratio":     [0.8, 0.2, 0.7],
        "negative_ratio":     [0.1, 0.7, 0.2],
        "comment_count":      [1.1, 0.7, 0.9],
        "sentiment_momentum": [0.0, -0.4, 0.3],
        "sentiment_ma5":      [0.7, 0.5, 0.53],
    }
    path = os.path.join(tmp_dir, "sentiment.csv")
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_merge_adds_sentiment_cols():
    with tempfile.TemporaryDirectory() as tmp:
        sent_csv = _make_sentiment_csv(tmp)
        indicators = pd.DataFrame({
            "date":  ["2026-01-05", "2026-01-06", "2026-01-07"],
            "close": [1700.0, 1680.0, 1720.0],
        })
        merged = merge_with_indicators(indicators, sent_csv)
        for col in SENTIMENT_COLS:
            assert col in merged.columns


def test_merge_no_nan_after_fill():
    with tempfile.TemporaryDirectory() as tmp:
        sent_csv = _make_sentiment_csv(tmp)
        # 技术指标多一天（1月8日，无对应情感数据 → 前向填充）
        indicators = pd.DataFrame({
            "date":  ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
            "close": [1700.0, 1680.0, 1720.0, 1710.0],
        })
        merged = merge_with_indicators(indicators, sent_csv)
        assert not merged[SENTIMENT_COLS].isnull().any().any()


def test_merge_row_count_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        sent_csv = _make_sentiment_csv(tmp)
        indicators = pd.DataFrame({
            "date":  ["2026-01-05", "2026-01-06"],
            "close": [1700.0, 1680.0],
        })
        merged = merge_with_indicators(indicators, sent_csv)
        assert len(merged) == len(indicators)


if __name__ == "__main__":
    tests = [
        test_clean_removes_url,
        test_clean_removes_emoji,
        test_clean_short_text,
        test_clean_normal_text,
        test_score_positive,
        test_score_negative,
        test_score_empty_returns_neutral,
        test_score_range,
        test_aggregate_returns_dataframe,
        test_aggregate_output_columns,
        test_aggregate_three_days,
        test_aggregate_sentiment_range,
        test_aggregate_no_nan,
        test_aggregate_saves_csv,
        test_merge_adds_sentiment_cols,
        test_merge_no_nan_after_fill,
        test_merge_row_count_preserved,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 通过")
