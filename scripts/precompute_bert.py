"""
预计算 BERT [CLS] 向量并缓存到磁盘（只需运行一次）

流程：
    股吧评论 CSV
        → 每日聚合标题文本
        → BERT-Base-Chinese tokenize + forward
        → 取 [CLS] 向量 (768维)
        → 保存 data/processed/bert_cls_daily.npy  (shape: [n_trade_days, 768])
               data/processed/bert_dates.csv       (对应日期)

用法：
    python scripts/precompute_bert.py
    python scripts/precompute_bert.py --bert bert-base-chinese --batch 32 --device cpu
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset

from src.data.indicators import load_price, compute_indicators
from src.data.news_dataset import build_daily_text_series


# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--price_csv",    default="data/raw/moutai_price.csv")
    p.add_argument("--comments_csv", default="data/raw/贵州茅台600519_股吧评论.csv")
    p.add_argument("--out_dir",      default="data/processed")
    p.add_argument("--bert",         default="bert-base-chinese")
    p.add_argument("--max_len",      type=int, default=128)
    p.add_argument("--batch",        type=int, default=16)
    p.add_argument("--max_titles",   type=int, default=5)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class TextListDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
        }


@torch.no_grad()
def compute_cls_vectors(
    texts: list[str],
    bert_name: str,
    max_len: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """返回 (N, 768) 的 [CLS] 向量数组。"""
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    model     = BertModel.from_pretrained(bert_name).to(device).eval()

    dataset = TextListDataset(texts, tokenizer, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    cls_list = []
    for i, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out  = model(input_ids, attention_mask=attention_mask)
        cls  = out.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)
        cls_list.append(cls)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(loader)}] batches done")

    return np.concatenate(cls_list, axis=0)   # (N, 768)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 取交易日序列（从行情数据）
    print("加载行情数据...")
    df = compute_indicators(load_price(args.price_csv))
    trade_dates = pd.to_datetime(df["date"].values)

    # 2. 每日聚合文本并对齐
    print("聚合股吧标题...")
    daily_texts = build_daily_text_series(
        args.comments_csv, trade_dates, max_titles=args.max_titles
    )
    texts = daily_texts.tolist()
    # 无文本的日期用占位符（BERT 会输出接近零向量的 CLS）
    texts = [t if t.strip() else "无相关新闻" for t in texts]

    # 3. BERT 前向 → [CLS]
    print(f"运行 BERT ({args.bert}) on {args.device}，共 {len(texts)} 天...")
    cls_vectors = compute_cls_vectors(texts, args.bert, args.max_len, args.batch, args.device)

    # 4. 保存
    out_cls   = os.path.join(args.out_dir, "bert_cls_daily.npy")
    out_dates = os.path.join(args.out_dir, "bert_dates.csv")
    np.save(out_cls, cls_vectors)
    pd.DataFrame({"date": trade_dates.strftime("%Y-%m-%d")}).to_csv(out_dates, index=False)
    print(f"已保存：{out_cls}  shape={cls_vectors.shape}")
    print(f"已保存：{out_dates}")


if __name__ == "__main__":
    main()
