"""
东方财富股吧评论爬虫
====================
目标：爬取贵州茅台（600519）股吧帖子，保存到 CSV。

接口：东方财富 JSON API（比 HTML 解析更稳定，返回结构化数据）
输出字段：reply_num, read_num, pl, time1, publish_time, author

特性：
  - 支持日期范围过滤（只保留指定时间段内的帖子）
  - 断点续爬（追加模式，不覆盖已有数据）
  - 自动重试（失败最多重试3次）
  - 礼貌爬取（随机间隔 1~2.5 秒）

用法：
    # 爬取最近50页
    python -m src.data.crawler.guba_crawler

    # 指定页数和日期范围
    python -m src.data.crawler.guba_crawler --pages 100 --start 2019-01-01 --end 2026-04-17

    # 追加模式（不覆盖已有数据，继续往后爬）
    python -m src.data.crawler.guba_crawler --pages 200 --append
"""

import json
import time
import random
import csv
import argparse
import os
from datetime import datetime

import requests

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

STOCK_CODE = "600519"
STOCK_NAME = "贵州茅台"

# 东方财富股吧 JSON 接口
API_URL = (
    "https://guba.eastmoney.com/interface/GetList.aspx"
    "?cb=jQuery&type=AShareGuba&code={code}&page={page}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": f"https://guba.eastmoney.com/list,{STOCK_CODE}.html",
}

FIELDS = ["reply_num", "read_num", "pl", "time1", "publish_time", "author"]
DEFAULT_OUTPUT = f"data/raw/{STOCK_NAME}{STOCK_CODE}_股吧评论.csv"


# ─────────────────────────────────────────────
# 单页爬取（含重试）
# ─────────────────────────────────────────────

def fetch_page(
    page: int,
    session: requests.Session,
    max_retries: int = 3,
) -> list[dict]:
    """
    爬取第 page 页的帖子列表，失败自动重试。
    返回帖子字典列表，失败时返回空列表。
    """
    url = API_URL.format(code=STOCK_CODE, page=page)

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()

            # 响应格式：jQuery({...}) → 去掉包裹取出 JSON
            raw = resp.text.strip()
            if raw.startswith("jQuery("):
                raw = raw[7:]
            if raw.endswith(")"):
                raw = raw[:-1]

            data = json.loads(raw)
            return data.get("re", [])

        except Exception as e:
            if attempt < max_retries:
                wait = attempt * 2
                print(f"  [重试 {attempt}/{max_retries}] 第{page}页失败: {e}，{wait}秒后重试")
                time.sleep(wait)
            else:
                print(f"  [跳过] 第{page}页连续失败: {e}")
                return []


# ─────────────────────────────────────────────
# 主爬取函数
# ─────────────────────────────────────────────

def crawl(
    total_pages: int = 50,
    output: str = DEFAULT_OUTPUT,
    start_date: str | None = None,
    end_date: str | None = None,
    append: bool = False,
) -> int:
    """
    爬取股吧帖子并保存到 CSV。

    Args:
        total_pages: 最多爬取页数（每页约20条）
        output:      输出文件路径
        start_date:  只保留此日期之后的帖子，格式 "YYYY-MM-DD"
        end_date:    只保留此日期之前的帖子，格式 "YYYY-MM-DD"
        append:      True=追加到已有文件，False=覆盖

    Returns:
        实际保存的帖子数量
    """
    # 解析日期过滤范围
    dt_start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    dt_end   = datetime.strptime(end_date,   "%Y-%m-%d") if end_date   else None

    os.makedirs(os.path.dirname(output), exist_ok=True)

    # 追加模式：不写表头；覆盖模式：写表头
    mode = "a" if (append and os.path.exists(output)) else "w"
    write_header = (mode == "w")

    session   = requests.Session()
    saved     = 0
    stopped   = False   # 遇到超出日期范围时提前停止

    with open(output, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for page in range(1, total_pages + 1):
            if stopped:
                break

            posts = fetch_page(page, session)
            if not posts:
                print(f"  第{page}页无数据，停止爬取")
                break

            page_saved = 0
            for post in posts:
                # 解析发布时间
                pub_str = post.get("publish_time", "")
                try:
                    pub_dt = datetime.strptime(pub_str[:19], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    pub_dt = None

                # 日期过滤
                if pub_dt:
                    if dt_end   and pub_dt > dt_end:
                        continue           # 太新，跳过
                    if dt_start and pub_dt < dt_start:
                        stopped = True     # 已经早于起始日期，后续页也不用爬了
                        break

                row = {k: post.get(k, "") for k in FIELDS}
                writer.writerow(row)
                page_saved += 1
                saved += 1

            print(f"  第 {page:3d}/{total_pages} 页  本页保存 {page_saved} 条  累计 {saved} 条")

            # 礼貌爬取：随机等待
            time.sleep(random.uniform(1.0, 2.5))

    print(f"\n[完成] 共保存 {saved} 条帖子 → {output}")
    return saved


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="东方财富茅台股吧评论爬虫")
    p.add_argument("--pages",  type=int, default=50,           help="最多爬取页数（默认50）")
    p.add_argument("--output", default=DEFAULT_OUTPUT,          help="输出文件路径")
    p.add_argument("--start",  default=None,                    help="起始日期 YYYY-MM-DD")
    p.add_argument("--end",    default=None,                    help="截止日期 YYYY-MM-DD")
    p.add_argument("--append", action="store_true",             help="追加模式，不覆盖已有数据")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"目标：{STOCK_NAME}（{STOCK_CODE}）股吧")
    print(f"页数：{args.pages}  日期：{args.start or '不限'} ~ {args.end or '不限'}")
    print(f"模式：{'追加' if args.append else '覆盖'}  输出：{args.output}\n")
    crawl(
        total_pages=args.pages,
        output=args.output,
        start_date=args.start,
        end_date=args.end,
        append=args.append,
    )
