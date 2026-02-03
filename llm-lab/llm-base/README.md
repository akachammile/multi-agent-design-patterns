# llm-base

该目录包含大语言模型（LLM）的基础模型实现，主要包括经典的 Transformer 架构和 BERT 模型。
transformer: 用于复习 NLP 模型的基础原理。后面要做到手撕MHA以及其数学原理。
bert: 文本分类，主要用于提供RAG时可能的一种方案


## 目录结构

### 1. Transformer
- **路径**: `transformer/`
- **描述**: 基于论文 "Attention Is All You Need" (Vaswani et al., 2017) 的 PyTorch 实现。
- **参考**: 主要参考了 Harvard NLP 的 "The Annotated Transformer"。
- **主要文件**:
  - `transformer.py`: Transformer 模型的完整架构定义（Encoder, Decoder, Attention, etc.）。
  - `train.py`: 模型的训练脚本。
  - `config.py`: 配置管理。
  - `data.py`: 数据处理工具。

### 2. BERT-pytorch
- **路径**: `BERT-pytorch/`
- **描述**: Google AI 2018 BERT 模型的 PyTorch 实现。
- **来源**: [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
- **主要功能**:
  - 提供了 BERT 模型的预训练（Masked Language Model 和 Next Sentence Prediction）和微调功能。
  - 代码结构简洁，易于理解。