#!/bin/bash

# ZipVoice Voice Conversion Training Script
# This script trains a voice conversion model using ZipVoice architecture
# with pre-trained Zipformer as content encoder

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=5

# Pre-trained Zipformer model path (you need to download this)
pretrained_zipformer="download/zipformer_cr_ctc_small/pretrained.pt"

# Training parameters
world_size=4
use_fp16=1
num_epochs=30
max_duration=400
max_len=15.0
base_lr=0.001
mask_prob=0.8
mask_length=10
freeze_content_encoder=1

# Experiment directory
exp_dir="exp/zipvoice_vc"

echo "ZipVoice Voice Conversion Training Pipeline"
echo "==========================================="

#### Data Preparation (Stage 1)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Prepare LibriTTS-100h data for Voice Conversion"
    bash local/prepare_libritts_vc.sh
fi

#### Download Pre-trained Model (Stage 2)

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Download pre-trained Zipformer-CR-CTC model"
    
    mkdir -p download/zipformer_cr_ctc_small
    
    # NOTE: You need to manually download the pre-trained model
    # from https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#zipformer-zipformer--cr-ctc
    
    if [ ! -f "$pretrained_zipformer" ]; then
        echo "ERROR: Pre-trained Zipformer model not found at: $pretrained_zipformer"
        echo "Please download the model from:"
        echo "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#zipformer-zipformer--cr-ctc"
        echo ""
        echo "Suggested download commands:"
        echo "cd download/zipformer_cr_ctc_small"
        echo "wget https://huggingface.co/k2-fsa/icefall-asr-librispeech-zipformer-cr-ctc-2023-04-17/resolve/main/exp/pretrained.pt"
        echo "cd ../.."
        exit 1
    fi
    
    echo "Pre-trained model found: $pretrained_zipformer"
fi

#### Training (Stage 3)

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Train ZipVoice Voice Conversion model"
    
    python3 -m zipvoice.bin.train_zipvoice_vc \
        --world-size ${world_size} \
        --use-fp16 ${use_fp16} \
        --model-config conf/zipvoice_vc_base.json \
        --pretrained-content-encoder ${pretrained_zipformer} \
        --freeze-content-encoder ${freeze_content_encoder} \
        --train-manifest data/fbank/libritts_cuts_vc_train.jsonl.gz \
        --dev-manifest data/fbank/libritts_cuts_vc_dev.jsonl.gz \
        --num-epochs ${num_epochs} \
        --base-lr ${base_lr} \
        --max-duration ${max_duration} \
        --max-len ${max_len} \
        --mask-prob ${mask_prob} \
        --mask-length ${mask_length} \
        --exp-dir ${exp_dir} \
        --save-every-n 5 \
        --valid-every-n 2
fi

#### Model Averaging (Stage 4)

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Average the best checkpoints"
    
    python3 -m zipvoice.bin.generate_averaged_model \
        --epoch ${num_epochs} \
        --avg 5 \
        --model-name zipvoice_vc \
        --exp-dir ${exp_dir}
    
    # The generated model will be: ${exp_dir}/epoch-${num_epochs}-avg-5.pt
    echo "Averaged model saved: ${exp_dir}/epoch-${num_epochs}-avg-5.pt"
fi

#### Inference Test (Stage 5)

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Test Voice Conversion inference"
    
    # Create a simple test script
    python3 -m zipvoice.bin.infer_zipvoice_vc \
        --model-name zipvoice_vc \
        --model-dir ${exp_dir} \
        --checkpoint-name epoch-${num_epochs}-avg-5.pt \
        --test-list test_vc.tsv \
        --res-dir results/test_vc \
        --num-step 16 \
        --guidance-scale 1.0
fi

echo "ZipVoice Voice Conversion training completed!"
echo "Model saved in: ${exp_dir}"
echo "To test the model, run inference with your own audio files."
