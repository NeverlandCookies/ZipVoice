#!/bin/bash
# ZipVoice-VC 单句语音转换推理脚本
# 使用说明: 设置源语音和目标语音路径，然后运行

# 🎵 音频文件路径 (请替换为实际文件路径)
SOURCE_WAV="egs/zipvoice/download/LibriTTS/test-clean/672/122797/672_122797_000000_000000.wav"      # 源语音文件 (要转换的内容)
TARGET_WAV="egs/zipvoice/download/LibriTTS/test-clean/1284/1180/1284_1180_000001_000000.wav"      # 目标语音文件 (要模仿的音色)

# 📁 输出路径
OUTPUT_WAV="infer_try/output_converted30.wav"

EXP_DIR="egs/zipvoice/exp/zipvoice_s3_libritts-clean100_4gpu_epoch60_bs100s_lr0.02"
MODEL_NAME="epoch-60-avg-10.pt"

# 🔍 检查输入文件
if [ ! -f "${SOURCE_WAV}" ]; then
    echo "❌ 源语音文件不存在: ${SOURCE_WAV}"
    echo "请设置正确的 SOURCE_WAV 路径"
    exit 1
fi

if [ ! -f "${TARGET_WAV}" ]; then
    echo "❌ 目标语音文件不存在: ${TARGET_WAV}"
    echo "请设置正确的 TARGET_WAV 路径"  
    exit 1
fi

echo "🚀 开始语音转换..."
echo "📥 源语音: ${SOURCE_WAV}"
echo "🎯 目标音色: ${TARGET_WAV}"
echo "📤 输出文件: ${OUTPUT_WAV}"

# 🔄 执行推理
python3 -m zipvoice.bin.s3_infer_zipvoice \
    --model-dir ${EXP_DIR} \
    --checkpoint-name ${MODEL_NAME} \
    --vocoder-path "/dkucc/home/tp286/ZipVC/models/vocos-mel-24khz"\
    --tokenizer s3 \
    --source-wav "${SOURCE_WAV}" \
    --target-wav "${TARGET_WAV}" \
    --res-wav-path "${OUTPUT_WAV}" \
    --num-step 16 \
    --guidance-scale 1.0 \  #  up WER down >1
    --t-shift 0.5 \
    --target-rms 0.1

if [ $? -eq 0 ]; then
    echo "✅ 语音转换完成!"
    echo "📁 输出文件: ${OUTPUT_WAV}"
else
    echo "❌ 语音转换失败"
fi