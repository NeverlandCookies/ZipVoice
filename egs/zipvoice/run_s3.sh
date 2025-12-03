#!/bin/bash

# ZipVoice-S3 完整训练脚本
# 基于LibriTTS数据集，使用S3Tokenizer训练ZipVoice模型

# 添加项目根目录到PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Bash调试模式
set -e
set -u
set -o pipefail

# ============== 支持命令行参数 ==============
# 默认值
stage=1
stop_stage=3

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            stage="$2"
            shift 2
            ;;
        --stop-stage)
            stop_stage="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--stage N] [--stop-stage N]"
            echo "  --stage N        : 开始阶段 (默认: 1)"
            echo "  --stop-stage N   : 结束阶段 (默认: 3)"
            echo "  -h, --help       : 显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 验证参数
if [ ${stage} -lt 1 ] || [ ${stage} -gt 3 ]; then
    echo "错误: stage 必须在 1-3 之间"
    exit 1
fi

if [ ${stop_stage} -lt 1 ] || [ ${stop_stage} -gt 3 ]; then
    echo "错误: stop_stage 必须在 1-3 之间"
    exit 1
fi

if [ ${stage} -gt ${stop_stage} ]; then
    echo "错误: stage (${stage}) 不能大于 stop_stage (${stop_stage})"
    exit 1
fi

# ====================================================================
# 🔥 关键超参数配置区域 - 请根据需要调整
# ====================================================================

# 🎯 Batch Size控制 (这是最重要的参数!)
MAX_DURATION=100        # 单位: 秒 
                        # 💡 这个参数直接控制batch size大小
                        # 📊 测试建议: 100-200 (避免OOM)
                        # 🚀 正式训练: 250-500 (更大batch更稳定)

# 🔄 训练轮数控制  
NUM_EPOCHS=60          # 💡 测试建议: 5-10轮快速验证
                       # 🚀 正式训练: 60-100轮

# 🖥️ 硬件配置
WORLD_SIZE=4           # GPU数量

# 📚 学习率相关
BASE_LR=0.02           # 基础学习率 (推荐保持0.02)
LR_EPOCHS=10           # 学习率衰减周期

# 🎵 音频长度限制
MAX_LEN=20             # 最大音频长度(秒) - 测试用20，正式用20-30
MIN_LEN=1.0            # 最小音频长度(秒)

# 📚 数据集配置
DATASET="libritts-clean100"     # 数据集名称

# 🔧 实验配置
EXPERIMENT_PREFIX="zipvoice_s3"
INCLUDE_TIMESTAMP=false

# ====================================================================
# 🏷️ 动态生成实验目录名
# ====================================================================
if [ "${INCLUDE_TIMESTAMP}" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXP_DIR="exp/${EXPERIMENT_PREFIX}_${DATASET}_${WORLD_SIZE}gpu_epoch${NUM_EPOCHS}_bs${MAX_DURATION}s_lr${BASE_LR}_${TIMESTAMP}"
else
    EXP_DIR="exp/${EXPERIMENT_PREFIX}_${DATASET}_${WORLD_SIZE}gpu_epoch${NUM_EPOCHS}_bs${MAX_DURATION}s_lr${BASE_LR}"
fi

# ====================================================================
# 配置信息显示
# ====================================================================
echo "========================================================"
echo "🚀 ZipVoice-S3 训练配置"
echo "========================================================"
echo "🖥️  GPU数量: ${WORLD_SIZE}"
echo "📦 Batch Size控制: ${MAX_DURATION} 秒"
echo "🔄 训练轮数: ${NUM_EPOCHS}"
echo "📚 学习率: ${BASE_LR}"
echo "📚 学习率衰减周期: ${LR_EPOCHS}"
echo "🎵 音频长度: ${MIN_LEN}-${MAX_LEN} 秒"
echo "📁 实验目录: ${EXP_DIR}"
echo "📊 数据集: ${DATASET}"
echo "🔤 Tokenizer: S3"
echo "⏰ 开始时间: $(date)"
echo "========================================================"
echo ""

# ====================================================================
# 📝 创建实验信息文件
# ====================================================================
mkdir -p ${EXP_DIR}
cat > ${EXP_DIR}/experiment_info.txt << EOF
ZipVoice-S3 实验信息
====================
开始时间: $(date)
数据集: ${DATASET}
GPU数量: ${WORLD_SIZE}
训练轮数: ${NUM_EPOCHS}
Batch控制: ${MAX_DURATION} 秒
学习率: ${BASE_LR}
学习率衰减周期: ${LR_EPOCHS}
音频长度范围: ${MIN_LEN}-${MAX_LEN} 秒
实验目录: ${EXP_DIR}
Tokenizer: S3
EOF

# ====================================================================
# Stage 1: 数据准备
# ====================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "🔧 Stage 1: 数据准备 - ${DATASET} (使用S3 Tokenizer)"
    echo ""
    
    # 检查数据准备脚本是否存在
    if [ ! -f "local/prepare_libritts.sh" ]; then
        echo "❌ 错误: local/prepare_libritts.sh 不存在!"
        echo "请确保你在 egs/zipvoice/ 目录下运行此脚本"
        exit 1
    fi
    
    # 运行数据准备
    bash local/prepare_libritts.sh
    
    # 验证数据是否准备完成
    if [ ! -f "data/s3_fbank/libritts_cuts_train-clean-100.jsonl.gz" ]; then
        echo "❌ 数据准备失败，请检查prepare脚本输出"
        exit 1
    fi
    
    echo "✅ 数据准备完成!"
    echo ""
fi

# ====================================================================  
# Stage 2: ZipVoice-S3模型训练
# ====================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "🚀 Stage 2: ZipVoice-S3 模型训练"
    echo "配置:"
    echo "  - GPU数量: ${WORLD_SIZE}"
    echo "  - Batch控制: ${MAX_DURATION}秒"  
    echo "  - 训练轮数: ${NUM_EPOCHS}"
    echo "  - 数据集: ${DATASET}"
    echo "  - Tokenizer: S3"
    echo ""
    
    # 创建实验目录
    mkdir -p ${EXP_DIR}

    # ===== 添加：设置 NCCL 超时和内存优化 =====
    export NCCL_TIMEOUT=1800  # 30 分钟超时（验证阶段可能需要较长时间）
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # 开始训练（使用 train_zipvoice.py）
    python3 -m zipvoice.bin.train_zipvoice \
        --world-size ${WORLD_SIZE} \
        --use-fp16 1 \
        --num-epochs ${NUM_EPOCHS} \
        --max-duration ${MAX_DURATION} \
        --base-lr ${BASE_LR} \
        --lr-epochs ${LR_EPOCHS} \
        --max-len ${MAX_LEN} \
        --min-len ${MIN_LEN} \
        --valid-by-epoch 1 \
        --model-config conf/zipvoice_base.json \
        --tokenizer s3 \
        --token-file data/tokens_s3.txt \
        --dataset ${DATASET} \
        --manifest-dir data/s3_fbank \
        --exp-dir ${EXP_DIR}
    
    if [ $? -eq 0 ]; then
        echo "✅ 训练完成! 模型保存在: ${EXP_DIR}/"
    else
        echo "❌ 训练失败，请检查错误信息"
        exit 1
    fi
    echo ""
fi

# ====================================================================
# Stage 3: 模型平均化
# ====================================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "📊 Stage 3: 模型平均化"
    
    # 计算平均的epoch数量 (取最后几个epoch，但不超过总epoch数)
    AVG_NUM=$((NUM_EPOCHS > 10 ? 10 : NUM_EPOCHS))
    
    echo "配置:"
    echo "  - 目标epoch: ${NUM_EPOCHS}"
    echo "  - 平均epoch数: ${AVG_NUM}"
    echo ""
    
    # 检查是否有足够的checkpoint
    CHECKPOINT_COUNT=$(find ${EXP_DIR} -name "epoch-*.pt" | wc -l)
    if [ ${CHECKPOINT_COUNT} -lt ${AVG_NUM} ]; then
        echo "⚠️  警告: 只找到 ${CHECKPOINT_COUNT} 个checkpoint，少于所需的 ${AVG_NUM} 个"
        AVG_NUM=${CHECKPOINT_COUNT}
        echo "调整平均数量为: ${AVG_NUM}"
    fi
    
    # 使用 s3_generate_averaged_model.py 进行模型平均化
    python3 -m zipvoice.bin.generate_averaged_model \
        --epoch ${NUM_EPOCHS} \
        --avg ${AVG_NUM} \
        --model-name zipvoice_s3 \
        --exp-dir ${EXP_DIR}
    
    AVERAGED_MODEL="${EXP_DIR}/epoch-${NUM_EPOCHS}-avg-${AVG_NUM}.pt"
    if [ -f "${AVERAGED_MODEL}" ]; then
        echo "✅ 模型平均化完成!"
        echo "📁 平均化模型: ${AVERAGED_MODEL}"
    else
        echo "❌ 模型平均化失败"
        exit 1
    fi
    echo ""
fi

# ====================================================================
# 训练完成总结
# ====================================================================
echo "🎉 ZipVoice-S3 训练流程完成!"
echo "========================================================"
echo "📊 训练总结:"
echo "  🖥️  GPU数量: ${WORLD_SIZE}"
echo "  📦 Batch控制: ${MAX_DURATION} 秒"
echo "  🔄 训练轮数: ${NUM_EPOCHS}"
echo "  📁 实验目录: ${EXP_DIR}/"
echo "  🎯 数据集: ${DATASET}"
echo "  🔤 Tokenizer: S3"
echo ""
echo "📁 生成的文件:"
echo "  🤖 训练模型: ${EXP_DIR}/epoch-${NUM_EPOCHS}.pt"
AVG_NUM=$((NUM_EPOCHS > 10 ? 10 : NUM_EPOCHS))
echo "  📊 平均模型: ${EXP_DIR}/epoch-${NUM_EPOCHS}-avg-${AVG_NUM}.pt"
echo ""
echo "🔧 超参数调整指南:"
echo "  📦 调整batch size: 修改 MAX_DURATION"
echo "     - 减小 = 更小batch (避免OOM)"
echo "     - 增大 = 更大batch (更稳定训练)"
echo "  🔄 调整训练轮数: 修改 NUM_EPOCHS"
echo "  🚀 正式训练推荐: MAX_DURATION=250-500, NUM_EPOCHS=60-100"
echo ""
echo "🚀 下一步操作:"
echo "  1️⃣  检查训练日志: ${EXP_DIR}/log/log-train"
echo "  2️⃣  如果效果好，增加超参数进行正式训练"
echo "  3️⃣  准备推理脚本（待实现）"
echo "========================================================"