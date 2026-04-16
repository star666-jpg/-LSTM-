"""
东方财富股吧评论爬虫
目标：贵州茅台 600519 股吧帖子标题 + 发帖时间
输出：data/raw/贵州茅台600519_股吧评论.csv
"""

import time
import random
import csv
import requests
from datetime import datetime

BASE_URL = (
    "https://guba.eastmoney.com/interface/GetList.aspx"
    "?cb=jQuery&type=AShareGuba&code=600519&page={page}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://guba.eastmoney.com/",
}

FIELDS = ["reply_num", "read_num", "pl", "time1", "publish_time", "author"]


def fetch_page(page: int, session: requests.Session) -> list[dict]:
    url = BASE_URL.format(page=page)
    resp = session.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    # 响应格式: jQuery({...})
    raw = resp.text.strip()
    if raw.startswith("jQuery("):
        raw = raw[7:-1]
    import json
    data = json.loads(raw)
    return data.get("re", [])


def crawl(total_pages: int = 50, output: str = "data/raw/贵州茅台600519_股吧评论.csv"):
    session = requests.Session()
    rows = []
    for page in range(1, total_pages + 1):
        try:
            posts = fetch_page(page, session)
            for p in posts:
                rows.append({k: p.get(k, "") for k in FIELDS})
            print(f"  第 {page}/{total_pages} 页  累计 {len(rows)} 条")
        except Exception as e:
            print(f"  [WARN] 第 {page} 页失败: {e}")
        time.sleep(random.uniform(1.0, 2.5))  # 礼貌爬取

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] 共 {len(rows)} 条评论已保存至 {output}")


if __name__ == "__main__":
    crawl(total_pages=50)
