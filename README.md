# 基于多模态数据融合的股票趋势预测模型

融合孙惠莹(2025)与昝泓含(2024)两篇论文的方法，
以贵州茅台（600519）为研究对象，构建四模态协同预测系统。

## 系统架构

```
输入四路模态
  ├── 支路1：技术指标时序 (OHLCV + 15维指标)
  │         → TCN-LSTM-Transformer  → feat_dim
  ├── 支路2：TIFS 蜡烛图 (3×H×W)
  │         → CS-ACNN (3层+通道/空间注意力)
  │         → INSA (图文早期融合)   → feat_dim
  ├── 支路3：中文新闻标题
  │         → BERT-Base-Chinese + GFM → feat_dim
  └── 支路4：股吧情感得分 (SnowNLP 日聚合)
            → MLP                    → feat_dim

融合层: H-MoE (4专家，动态门控路由)
输出:
  ├── 回归头  → 次日收盘价
  └── 分类头  → 涨跌方向 (0/1)
```

## 目录结构

```
├── data/
│   ├── raw/          原始行情 + 股吧评论
│   └── processed/    技术指标 + 情感得分
├── src/
│   ├── data/         数据处理 (指标计算 / 预处理 / 爬虫)
│   ├── models/       四条支路模型定义
│   ├── sentiment/    SnowNLP 情感打分
│   └── fusion/       四模态融合总模型
├── configs/          超参数配置
├── scripts/          训练入口 + 数据处理脚本
└── tests/            单元测试
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 计算技术指标
bash scripts/run_analysis.sh

# 3. 运行情感打分
python -m src.sentiment.snowlp_scorer

# 4. 阶段一训练（时序支路基线）
python scripts/train.py
```

## 参考论文

- 孙惠莹. 融合情感分析的改进LSTM模型股票预测[D]. 大连理工大学, 2025.
- 昝泓含. 基于多模态数据融合的股票趋势预测模型[D]. 燕山大学, 2024.
