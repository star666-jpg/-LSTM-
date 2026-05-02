"""
东方财富新闻数据填充脚本（替代失效的股吧爬虫）
================================================
说明：guba_crawler.py 依赖的旧 JSON 接口（GetList.aspx?type=AShareGuba）
已被东方财富下线/反爬，2026 年起返回 HTML 页面壳，无法抓取帖子流。
临时方案：用 akshare.stock_news_em 拉取新闻列表（~10 条最新），
转换成与 guba_crawler.py 输出兼容的 CSV 格式（pl=标题, publish_time=时间）。

输出：data/raw/贵州茅台600519_股吧评论.csv

用法：
    python scripts/fetch_news_em.py
"""

import argparse
import os
import sys

import akshare as ak
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="600519")
    p.add_argument("--out",
                   default="data/raw/贵州茅台600519_股吧评论.csv")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"拉取新闻：{args.symbol}")
    df = ak.stock_news_em(symbol=args.symbol)

    if df is None or df.empty:
        print("[ERR] 接口返回空", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame({
        "reply_num":    [0] * len(df),
        "read_num":     [0] * len(df),
        "pl":           df["新闻标题"].astype(str).str.strip(),
        "time1":        df["发布时间"],
        "publish_time": df["发布时间"],
        "author":       df["文章来源"].astype(str),
    })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存 {len(out)} 条 → {args.out}")
    print(out[["pl", "publish_time"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
