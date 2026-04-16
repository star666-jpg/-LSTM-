"""
预测结果可视化
==============
读取 phase1_predictions.csv，画出真实价格 vs 预测价格对比图。

用法：
    python scripts/plot_predictions.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Windows 中文字体
matplotlib.rcParams["font.family"]      = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_phase1(
    pred_csv: str = "data/processed/phase1_predictions.csv",
    price_csv: str = "data/raw/moutai_price.csv",
    output:    str = "data/processed/phase1_result.png",
):
    # ── 读取预测结果
    df = pd.read_csv(pred_csv)
    true_vals = df["真实收盘价"].values
    pred_vals = df["预测收盘价"].values
    n = len(true_vals)

    # ── 尝试读取真实日期（用于 x 轴）
    dates = None
    try:
        price = pd.read_csv(price_csv)
        price = price.rename(columns={"日期": "date"})
        price["date"] = pd.to_datetime(price["date"])
        price = price.sort_values("date").reset_index(drop=True)
        # 测试集是后30%，从滑窗后开始
        window = 30
        test_start = int(len(price) * 0.7) + window
        dates = price["date"].values[test_start : test_start + n]
        if len(dates) != n:
            dates = None
    except Exception:
        dates = None

    x = dates if dates is not None else np.arange(n)

    # ── 计算误差
    mae = np.mean(np.abs(true_vals - pred_vals))
    r2  = 1 - np.sum((true_vals - pred_vals)**2) / np.sum((true_vals - true_vals.mean())**2)

    # ── 画图
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("贵州茅台（600519）收盘价预测 — 第一阶段\n"
                 f"特征：开盘/收盘/最高/最低/成交量（5维）  |  测试集 30%  |  R²={r2:.4f}  MAE={mae:.1f}元",
                 fontsize=13, y=0.98)

    # 上图：真实 vs 预测
    ax1 = axes[0]
    ax1.plot(x, true_vals, color="#2c7bb6", linewidth=1.5, label="真实收盘价", zorder=3)
    ax1.plot(x, pred_vals, color="#d7191c", linewidth=1.2,
             linestyle="--", label="预测收盘价", alpha=0.85, zorder=2)
    ax1.fill_between(x, true_vals, pred_vals,
                     alpha=0.12, color="#fdae61", label="误差区域")
    ax1.set_ylabel("收盘价（元）", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_xlim(x[0], x[-1])
    if dates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # 下图：误差（预测 - 真实）
    ax2 = axes[1]
    error = pred_vals - true_vals
    colors = ["#d7191c" if e > 0 else "#2c7bb6" for e in error]
    ax2.bar(x, error, color=colors, width=1.0 if dates is None else 0.8, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline( mae, color="#d7191c", linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.axhline(-mae, color="#2c7bb6", linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.set_ylabel("预测误差（元）", fontsize=11)
    ax2.set_xlabel("交易日" if dates is None else "日期", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_xlim(x[0], x[-1])
    if dates is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[OK] 图表已保存 → {output}")
    plt.show()


if __name__ == "__main__":
    plot_phase1()
