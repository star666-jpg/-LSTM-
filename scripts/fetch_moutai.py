"""
茅台日线数据爬取脚本
====================
用 akshare 调用东方财富接口拉取贵州茅台（600519）日线 OHLCV，
保存为 data/raw/moutai_price.csv，供后续技术指标 / 训练流水线使用。

输出列名（中文，与 src/data/indicators.py:COL_RENAME 对齐）：
    日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额

用法：
    python scripts/fetch_moutai.py
    python scripts/fetch_moutai.py --start 20180101 --end 20260501
"""

import argparse
import os
import sys

import akshare as ak
import pandas as pd


KEEP_COLS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="600519")
    p.add_argument("--start",  default="20180101")
    p.add_argument("--end",    default="20260501")
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""])
    p.add_argument("--out",    default="data/raw/moutai_price.csv")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"拉取 {args.symbol}  {args.start} → {args.end}  adjust={args.adjust}")

    df = ak.stock_zh_a_hist(
        symbol=args.symbol,
        period="daily",
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
    )

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"[ERR] 接口返回缺列：{missing}\n实际列：{df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    df = df[KEEP_COLS].copy()
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("日期").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存 {len(df)} 条 → {args.out}")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
