# Transformer Implementation

## 概述

这是 Transformer 模型的 PyTorch 实现，基于论文 "Attention Is All You Need" (Vaswani et al., 2017)。

## 来源仓库

本实现主要参考以下仓库：

### 1. Harvard NLP - The Annotated Transformer （主要参考）
- **仓库地址**: https://github.com/harvardnlp/annotated-transformer
- **论文解读**: https://nlp.seas.harvard.edu/annotated-transformer/
- **描述**: 这是业界公认最清晰、最易理解的 Transformer PyTorch 实现，由 Harvard NLP 团队编写，代码逐行对应论文内容。

### 2. TensorFlow Tensor2Tensor （官方原始实现）
- **仓库地址**: https://github.com/tensorflow/tensor2tensor
- **描述**: 这是 Google 官方原始的 TensorFlow 实现，由论文作者团队开发。该库已停止维护，后续项目为 Trax。

### 3. 原始论文
- **论文标题**: Attention Is All You Need
- **论文链接**: https://arxiv.org/abs/1706.03762
- **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- **发表**: NeurIPS 2017

## 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Model                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌───────────────┐              ┌───────────────┐          │
│   │   Encoder     │              │   Decoder     │          │
│   │   (N=6)       │              │   (N=6)       │          │
│   │               │              │               │          │
│   │ ┌───────────┐ │              │ ┌───────────┐ │          │
│   │ │ Self-Attn │ │              │ │ Masked    │ │          │
│   │ └───────────┘ │              │ │ Self-Attn │ │          │
│   │      ↓        │              │ └───────────┘ │          │
│   │ ┌───────────┐ │              │      ↓        │          │
│   │ │ FFN       │ │──Memory───→  │ ┌───────────┐ │          │
│   │ └───────────┘ │              │ │ Cross-Attn│ │          │
│   │               │              │ └───────────┘ │          │
│   │               │              │      ↓        │          │
│   │               │              │ ┌───────────┐ │          │
│   │               │              │ │ FFN       │ │          │
│   │               │              │ └───────────┘ │          │
│   └───────────────┘              └───────────────┘          │
│          ↑                              ↑                    │
│   ┌───────────────┐              ┌───────────────┐          │
│   │ + Positional  │              │ + Positional  │          │
│   │   Encoding    │              │   Encoding    │          │
│   └───────────────┘              └───────────────┘          │
│          ↑                              ↑                    │
│   ┌───────────────┐              ┌───────────────┐          │
│   │  Embedding    │              │  Embedding    │          │
│   └───────────────┘              └───────────────┘          │
│          ↑                              ↑                    │
│      Input Tokens               Output Tokens (shifted)      │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

| 组件 | 类名 | 描述 |
|------|------|------|
| 多头注意力 | `MultiHeadedAttention` | 并行计算多个注意力头 |
| 缩放点积注意力 | `attention()` | Scaled Dot-Product Attention |
| 前馈网络 | `PositionwiseFeedForward` | 两层线性变换 + ReLU |
| 位置编码 | `PositionalEncoding` | 正弦/余弦位置编码 |
| 层归一化 | `LayerNorm` | Layer Normalization |
| Embedding | `Embeddings` | 词嵌入层 |
| 编码器层 | `EncoderLayer` | Self-Attention + FFN |
| 解码器层 | `DecoderLayer` | Masked Self-Attention + Cross-Attention + FFN |
| 完整模型 | `EncoderDecoder` | 完整的 Encoder-Decoder 架构 |

## 默认超参数（Base Model）

| 参数 | 值 | 描述 |
|------|------|------|
| N | 6 | 编码器/解码器层数 |
| d_model | 512 | 模型维度 |
| d_ff | 2048 | 前馈网络隐藏层维度 |
| h | 8 | 注意力头数 |
| d_k | 64 | 每个头的维度 (d_model / h) |
| dropout | 0.1 | Dropout 概率 |

## 使用示例

```python
import torch
from transformer import make_model, subsequent_mask

# 创建模型
src_vocab = 10000  # 源语言词汇表大小
tgt_vocab = 10000  # 目标语言词汇表大小
model = make_model(src_vocab, tgt_vocab)

# 准备输入
batch_size = 32
src_len = 100
tgt_len = 100

src = torch.randint(0, src_vocab, (batch_size, src_len))
tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

# 创建掩码
src_mask = torch.ones(batch_size, 1, src_len)
tgt_mask = subsequent_mask(tgt_len)

# 前向传播
output = model(src, tgt, src_mask, tgt_mask)
logits = model.generator(output)

print(f"Output shape: {logits.shape}")  # (batch_size, tgt_len, tgt_vocab)
```

## 运行测试

```bash
python transformer.py
```

## 训练模型

### 1. 安装依赖

```bash
pip install torch spacy tqdm
```

下载 spaCy 语言模型（德英翻译）：

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### 2. 下载数据集

#### Multi30k（推荐入门）
- **仓库**: https://github.com/multi30k/dataset
- **描述**: 约 3 万对德英句子，适合快速测试

```bash
# 下载后将文件放入 data/ 目录，格式如下：
# data/train.de, data/train.en
# data/val.de, data/val.en
```

#### 其他数据集
| 数据集 | 链接 | 描述 |
|--------|------|------|
| WMT14 | https://www.statmt.org/wmt14/translation-task.html | 原论文使用的大规模数据集 |
| IWSLT | https://wit3.fbk.eu/ | TED 演讲翻译 |
| OpenSubtitles | https://opus.nlpl.eu/OpenSubtitles.php | 多语言字幕 |

### 3. 训练命令

#### 使用合成数据测试（无需下载）

```bash
python train.py --test
```

#### 使用真实数据训练

```bash
python train.py \
    --train_src data/train.de \
    --train_tgt data/train.en \
    --val_src data/val.de \
    --val_tgt data/val.en \
    --epochs 10 \
    --batch_size 32 \
    --d_model 512 \
    --h 8 \
    --N 6
```

### 4. 训练参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--d_model` | 512 | 模型维度 |
| `--d_ff` | 2048 | 前馈层维度 |
| `--h` | 8 | 注意力头数 |
| `--N` | 6 | 编码器/解码器层数 |
| `--warmup` | 4000 | 学习率预热步数 |
| `--checkpoint_dir` | checkpoints | 模型保存目录 |

### 5. 文件结构

```
transformer/
├── transformer.py   # 模型定义
├── config.py        # 配置管理
├── data.py          # 数据处理
├── train.py         # 训练脚本
├── README.md
├── data/            # 数据目录（需手动创建）
├── checkpoints/     # 模型保存目录（自动创建）
└── vocab/           # 词汇表目录（自动创建）
```


## 参考文献

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## 许可证

MIT License
