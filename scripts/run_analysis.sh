#!/bin/bash
# =============================================================================
# 脚本目标：安装依赖并计算茅台技术指标
# 适应环境：WSL2 (Ubuntu) / Linux 原生环境
# =============================================================================
set -euo pipefail

echo ">>> [1/3] 检查并激活 Python 虚拟环境..."
if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
else
    echo "    未检测到 .venv 目录，正在创建..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

echo ">>> [2/3] 安装依赖 (使用清华源加速)..."
python -m pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ">>> [3/3] 计算技术指标..."
python -c "from src.data.indicators import build_indicator_dataset; \
build_indicator_dataset('data/raw/moutai_price.csv', 'data/processed/moutai_technical_indicators.csv')"

echo ">>> [Done]"
echo "可使用以下命令查看结果："
echo "  head -n 5 data/processed/moutai_technical_indicators.csv"
echo "  wc -l data/processed/moutai_technical_indicators.csv"
