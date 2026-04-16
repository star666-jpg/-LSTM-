# 基于多模态数据融合的股票趋势预测模型

融合孙惠莹(2025)与昝泓含(2024)两篇论文的方法，以贵州茅台（600519）为研究对象，构建四模态协同预测系统，同时输出次日收盘价（回归）与涨跌方向（分类）。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      输入四路模态                         │
│                                                          │
│  支路1：技术指标时序                支路4：股吧情感得分    │
│  OHLCV + 15维技术指标              SnowNLP 7维日级特征    │
│        ↓                                  ↓              │
│  TCN-LSTM-Transformer              情感特征拼入支路1      │
│        ↓                                                  │
│  支路2：TIFS 蜡烛图      支路3：中文新闻标题               │
│  CS-ACNN（注意力CNN）    BERT-Base-Chinese + GFM          │
│        ↓                        ↓                        │
│        └──── INSA 图文早期融合 ──┘                        │
│                     ↓                                    │
│         ┌───────────────────────┐                        │
│         │   H-MoE 层次化融合    │                        │
│         │  第一级：多头 / 空头   │                        │
│         │  第二级：4个叶专家    │                        │
│         └───────────────────────┘                        │
│               ↓           ↓                              │
│           回归头         分类头                           │
│         次日收盘价      涨跌方向(0/1)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
-LSTM-/
├── data/
│   ├── raw/                        原始数据（不上传 GitHub）
│   │   ├── moutai_price.csv        茅台历史行情（2019至今，~1700条）
│   │   └── 贵州茅台600519_股吧评论.csv  东方财富股吧评论（~1600条）
│   └── processed/                  处理后数据（不上传 GitHub）
│       ├── moutai_technical_indicators.csv  15维技术指标
│       └── moutai_daily_sentiment.csv       7维日级情感特征
│
├── src/
│   ├── data/
│   │   ├── indicators.py           技术指标计算（ta库，15维特征）
│   │   ├── preprocess.py           滑窗切分 + MinMax标准化 + 数据集划分
│   │   └── crawler/
│   │       └── guba_crawler.py     东方财富股吧评论爬虫
│   ├── models/
│   │   ├── tcn_lstm_transformer.py 支路1：时序建模（TCN+LSTM+Transformer）
│   │   ├── cs_acnn.py              支路2：图像特征（CS-ACNN）
│   │   ├── bert_gfm.py             支路3：文本特征（BERT+GFM）
│   │   └── hmoe.py                 融合层：H-MoE（层次化混合专家）
│   ├── sentiment/
│   │   └── snowlp_scorer.py        支路4：SnowNLP情感打分 + 日级聚合
│   └── fusion/
│       └── multimodal_model.py     四模态总模型
│
├── configs/
│   └── config.yaml                 超参数配置（窗口/学习率/批大小等）
├── scripts/
│   ├── train_stage1.py             阶段一训练：支路1+H-MoE时序基线
│   └── run_analysis.sh             一键安装依赖并计算技术指标
├── tests/
│   ├── test_tcn.py                 TCN-LSTM-Transformer 测试（7个用例）
│   ├── test_hmoe.py                H-MoE 测试（10个用例）
│   ├── test_sentiment.py           情感模块测试（17个用例）
│   └── test_indicators.py          技术指标测试（3个用例）
├── docs/
│   ├── 代码说明.md                  各模块详细文档
│   └── git使用说明.md               Git 操作指南
└── requirements.txt
```

---

## 快速开始

### 环境安装

```bash
# Windows（VS Code 终端）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 数据准备

```bash
# 1. 计算技术指标（从 moutai_price.csv 生成）
python -m src.data.indicators

# 2. 情感打分（从股吧评论生成，首次约 1~3 分钟）
python -m src.sentiment.snowlp_scorer
```

### 训练

```bash
# 阶段一：时序支路基线（TCN-LSTM-Transformer + H-MoE）
python scripts/train_stage1.py

# 自定义参数
python scripts/train_stage1.py --epochs 100 --window 30 --lr 0.0005
```

### 测试

```bash
python tests/test_tcn.py
python tests/test_hmoe.py
python tests/test_sentiment.py
```

---

## 训练阶段规划

| 阶段 | 内容 | 输入特征数 | 目标 R² |
|------|------|-----------|--------|
| 阶段一 | 支路1（技术指标）+ H-MoE | ~20维 | ≥ 0.42（LSTM基线） |
| 阶段二 | 阶段一 + 支路4（情感） | ~27维 | ≥ 0.68（论文TCN基线） |
| 阶段三 | 加入支路3（BERT文本） | 多模态 | ≥ 0.71（论文完整模型） |
| 阶段四 | 加入支路2（TIFS图像） | 四模态 | 超越论文基准 |

---

## 参考论文

- 孙惠莹. 融合情感分析的改进LSTM模型股票预测[D]. 大连理工大学, 2025.
- 昝泓含. 基于多模态数据融合的股票趋势预测模型[D]. 燕山大学, 2024.
- Bai S, et al. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling[J]. arXiv, 2018.
- Vaswani A, et al. Attention Is All You Need[C]. NeurIPS, 2017.
- Devlin J, et al. BERT: Pre-training of Deep Bidirectional Transformers[C]. NAACL, 2019.
- Shazeer N, et al. Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer[C]. ICLR, 2017.
