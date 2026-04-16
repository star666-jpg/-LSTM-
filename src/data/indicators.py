"""
技术指标计算模块
基于 ta 库从茅台 OHLCV 数据计算多维度技术指标，
供后续 TIFS 图像生成和时序建模使用。
"""

import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna

# 与论文一致的 15 维核心特征列表（趋势 / 动量 / 波动 / 成交量 四维度）
CORE_FEATURES = [
    "date", "open", "high", "low", "close", "volume",
    # 趋势
    "trend_sma_fast",
    "trend_ema_fast",
    "trend_macd",
    "trend_macd_signal",
    "trend_cci",
    "trend_adx",
    # 动量
    "momentum_rsi",
    "momentum_stoch",
    "momentum_roc",
    # 波动率
    "volatility_bbh",
    "volatility_bbl",
    "volatility_atr",
    # 成交量
    "volume_mfi",
    "volume_obv",
]

COL_RENAME = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def load_price(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COL_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return dropna(df)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df_ta = add_all_ta_features(
        df,
        open="open", high="high", low="low",
        close="close", volume="volume",
        fillna=True,
    )
    available = [c for c in CORE_FEATURES if c in df_ta.columns]
    return df_ta[available].copy()


def build_indicator_dataset(
    price_csv: str,
    output_csv: str,
    start_date: str | None = None,
) -> pd.DataFrame:
    df = load_price(price_csv)
    df_out = compute_indicators(df)
    if start_date:
        df_out = df_out[df_out["date"] >= pd.to_datetime(start_date)]
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 技术指标已保存 -> {output_csv}  ({len(df_out)} 条, {len(df_out.columns)} 维)")
    return df_out


if __name__ == "__main__":
    build_indicator_dataset(
        price_csv="data/raw/moutai_price.csv",
        output_csv="data/processed/moutai_technical_indicators.csv",
    )
