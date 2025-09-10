# ZipVoice Voice Conversion 使用指南

本文档详细说明如何使用修改后的ZipVoice进行语音转换(Voice Conversion)训练和推理。

## 项目概述

本项目将ZipVoice架构适配为语音转换模型：

1. **内容编码器**: 使用预训练的Zipformer-CR-CTC模型替换原始文本编码器
2. **训练策略**: 采用自重建(Self-reconstruction)训练方式
3. **数据集**: 使用LibriTTS-100h进行训练
4. **生成器**: 保持原始ZipVoice的Flow Matching解码器

## 关键修改说明

### 1. 模型架构修改 (`zipvoice/models/zipvoice_vc.py`)

**修改内容**:
- 用预训练Zipformer替换文本编码器作为内容编码器
- 添加CTC线性层提取语音内容特征
- 保持Flow Matching解码器不变

**作用**:
- 利用ASR预训练的语音理解能力提取内容信息
- CTC后验概率作为内容表示，包含丰富的语音内容信息
- 避免了文本-语音对齐的复杂性

**关键函数**:
- `forward_content_condition()`: 从源语音提取内容特征
- `expand_content_condition()`: 将内容特征扩展到目标长度
- `load_pretrained_content_encoder()`: 加载预训练ASR模型

### 2. 数据集适配 (`zipvoice/dataset/dataset_vc.py`)

**修改内容**:
- 实现自重建训练数据加载
- 添加语音条件掩码机制
- 支持动态特征提取

**作用**:
- 源特征 = 原始语音（内容编码）
- 目标特征 = 原始语音（重建目标）
- 语音条件 = 掩码语音（说话人条件）

**关键参数**:
- `mask_prob`: 掩码概率 (默认0.8)
- `mask_length`: 连续掩码长度 (默认10帧)

### 3. 训练脚本 (`zipvoice/bin/train_zipvoice_vc.py`)

**修改内容**:
- 适配VC数据流
- 支持预训练模型加载
- 添加内容编码器冻结选项

**作用**:
- 自动加载预训练Zipformer权重
- 支持分布式训练
- 混合精度训练加速

## 使用步骤

### 步骤1: 准备环境

确保已安装所有依赖：
```bash
cd /dkucc/home/tp286/ZipVoice
pip install -r requirements.txt
```

### 步骤2: 下载预训练模型

从icefall下载Zipformer-CR-CTC模型：
```bash
mkdir -p download/zipformer_cr_ctc_small
cd download/zipformer_cr_ctc_small

# 下载small版本 (推荐开始使用)
wget https://huggingface.co/k2-fsa/icefall-asr-librispeech-zipformer-cr-ctc-2023-04-17/resolve/main/exp/pretrained.pt

cd ../..
```

### 步骤3: 运行完整训练流程

```bash
cd egs/zipvoice

# 运行完整流程 (数据准备 + 训练 + 推理)
bash run_zipvoice_vc.sh

# 或者分步骤运行:
# Stage 1: 数据准备
bash run_zipvoice_vc.sh --stage 1 --stop-stage 1

# Stage 2: 检查预训练模型
bash run_zipvoice_vc.sh --stage 2 --stop-stage 2

# Stage 3: 训练
bash run_zipvoice_vc.sh --stage 3 --stop-stage 3
```

### 步骤4: 模型推理

#### 单个文件转换:
```bash
python3 -m zipvoice.bin.infer_zipvoice_vc \
    --model-name zipvoice_vc \
    --model-dir exp/zipvoice_vc \
    --checkpoint-name epoch-30-avg-5.pt \
    --source-audio /path/to/source.wav \
    --target-audio /path/to/target.wav \
    --res-dir results/single_test
```

#### 批量转换:
```bash
# 编辑 test_vc.tsv 添加测试样本
python3 -m zipvoice.bin.infer_zipvoice_vc \
    --model-name zipvoice_vc \
    --model-dir exp/zipvoice_vc \
    --checkpoint-name epoch-30-avg-5.pt \
    --test-list test_vc.tsv \
    --res-dir results/batch_test
```

## 测试和验证方法

### 1. 快速验证模型加载

```bash
cd zipvoice/models
python3 zipvoice_vc.py
```
应该输出: "ZipVoiceVC test passed!"

### 2. 验证数据集加载

```bash
cd zipvoice/dataset
python3 dataset_vc.py
```
应该输出: "VoiceConversionDataset test passed!"

### 3. 检查训练日志

训练过程中监控以下指标：
```bash
# 查看训练日志
tail -f exp/zipvoice_vc/train.log

# 使用tensorboard查看训练曲线
tensorboard --logdir exp/zipvoice_vc/tensorboard
```

关键指标：
- `train/loss`: 训练损失（应该逐步下降）
- `val/loss`: 验证损失（应该下降且不过拟合）

### 4. 音频质量评估

#### 主观评估:
1. 听觉质量: 转换后音频是否清晰自然
2. 说话人相似度: 是否保持目标说话人特征
3. 内容保真度: 是否保持源语音的语言内容

#### 客观评估 (需要额外实现):
```bash
# 说话人相似度 (需要说话人识别模型)
python3 -m zipvoice.eval.speaker_similarity \
    --converted-dir results/batch_test \
    --target-dir /path/to/target/audios

# 内容保真度 (需要ASR模型)
python3 -m zipvoice.eval.content_preservation \
    --source-dir /path/to/source/audios \
    --converted-dir results/batch_test
```

## 故障排除

### 1. 内存不足错误
```bash
# 减少batch size
# 在 run_zipvoice_vc.sh 中修改:
max_duration=200  # 从400减少到200
```

### 2. 预训练模型加载失败
```bash
# 检查模型文件
ls -la download/zipformer_cr_ctc_small/pretrained.pt

# 检查模型内容
python3 -c "
import torch
ckpt = torch.load('download/zipformer_cr_ctc_small/pretrained.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
if 'model' in ckpt:
    print('Model keys:', [k for k in ckpt['model'].keys() if 'encoder' in k][:5])
"
```

### 3. CUDA内存错误
```bash
# 使用CPU训练 (调试用)
# 在训练脚本中设置:
world_size=1
use_fp16=0
max_duration=100
```

### 4. 数据加载错误
```bash
# 检查数据文件
python3 -c "
from lhotse import load_manifest_lazy
cuts = load_manifest_lazy('data/fbank/libritts_cuts_vc_train.jsonl.gz')
print('Number of cuts:', len(list(cuts)))
"
```

## 实验建议

### 1. 超参数调优
- `mask_prob`: 0.6-0.9 (掩码概率)
- `base_lr`: 0.0005-0.002 (学习率)
- `freeze_content_encoder`: 0/1 (是否冻结内容编码器)

### 2. 训练策略
- 先用冻结内容编码器训练10-15个epoch
- 然后解冻进行端到端微调5-10个epoch

### 3. 模型大小选择
- 开发阶段: 使用small模型快速验证
- 最终版本: 可尝试medium/large模型获得更好性能

## 预期结果

成功训练后，您应该能够：
1. 将任意说话人A的语音内容转换为说话人B的声音
2. 保持原始语音的语言内容和韵律
3. 获得与目标说话人相似的音色特征

训练时间估计：
- 小规模测试: 2-4小时 (4 GPU)
- 完整训练: 8-12小时 (4 GPU)

## 下一步扩展

1. **多语言支持**: 使用多语言ASR预训练模型
2. **零样本转换**: 支持未见过的目标说话人
3. **实时转换**: 优化模型结构支持流式处理
4. **情感转换**: 结合情感识别进行情感语音转换
