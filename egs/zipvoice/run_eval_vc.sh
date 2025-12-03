#!/bin/bash

# Voice Conversion Evaluation Pipeline
# This script orchestrates the complete VC evaluation process:
# 1. Prepare metadata from LibriTTS test-clean
# 2. Generate test pairs
# 3. Run VC inference
# 4. Evaluate with Speaker Similarity, WER, and UTMOS
# 5. Analyze results by gender


# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Bash调试模式
set -e
set -u
set -o pipefail

# ====================================================================
# Configuration
# ====================================================================

stage=1
stop_stage=6

# Paths
download_dir="download/LibriTTS"
test_clean_dir="${download_dir}/test-clean"
speakers_file="${download_dir}/speakers.tsv"
eval_model_dir="download/tts_eval_models"

# Data preparation
metadata_file="data/test_clean_metadata.tsv"
test_data_dir="data/vc_test"

# ====================================================================
# Inference Hyperparameters (Modify these for different experiments)
# ====================================================================
num_step=16                           # Sampling steps: 8, 16, 32, etc.
guidance_scale=5.0                    # Guidance scale: 0.5, 1.0, 1.5, 2.0, etc.
checkpoint_name="best-valid-loss.pt"  # Model checkpoint to use

# Other inference parameters (fixed, not included in experiment name)
t_shift=0.5
target_rms=0.1

# Model settings
model_dir="exp/zipvoice_s3_libritts_1gpu_epoch60_bs1000s_lr0.02"
vocoder_path="/dkucc/home/tp286/ZipVC/models/vocos-mel-24khz"

# ====================================================================
# Auto-generate experiment name from hyperparameters
# ====================================================================
# Remove .pt extension from checkpoint name
ckpt_short=$(echo $checkpoint_name | sed 's/.pt$//')
# Format guidance scale (replace . with p, e.g., "1.0" -> "1p0")
scale_str=$(echo $guidance_scale | sed 's/\./p/')
# Generate experiment name
exp_name="step${num_step}_scale${scale_str}_cp${ckpt_short}"

# Set result directories based on experiment name
results_dir="results/${exp_name}/vc_converted"
eval_results_dir="results/${exp_name}/vc_eval"

# Test pair generation parameters
num_pairs=500
source_min_duration=5.0
source_max_duration=20.0
target_min_duration=3.0
target_max_duration=15.0
balance_gender=true  # Set to false to disable gender balancing

# Random seed for reproducibility
seed=42

# Parse command line arguments
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
        --num-pairs)
            num_pairs="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--stage N] [--stop-stage N] [--num-pairs N] [--seed N]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "Voice Conversion Evaluation Pipeline"
echo "========================================================================"
echo "Experiment: ${exp_name}"
echo ""
echo "Hyperparameters:"
echo "  num_step:        ${num_step}"
echo "  guidance_scale:  ${guidance_scale}"
echo "  checkpoint:      ${checkpoint_name}"
echo "  t_shift:         ${t_shift}"
echo "  target_rms:      ${target_rms}"
echo ""
echo "Settings:"
echo "  Stage:           ${stage} - ${stop_stage}"
echo "  Test pairs:      ${num_pairs}"
echo "  Random seed:     ${seed}"
echo ""
echo "Output directories:"
echo "  Results:         ${results_dir}"
echo "  Evaluation:      ${eval_results_dir}"
echo "========================================================================"
echo ""

# ====================================================================
# Stage 1: Prepare metadata from LibriTTS test-clean
# ====================================================================
# NOTE: Stage 1-2 only need to run ONCE for all hyperparameter experiments.
#       After generating test pairs, you can modify hyperparameters and
#       run only Stage 3+ to test different configurations.
# ====================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Prepare metadata from LibriTTS test-clean"
    
    if [ ! -d "${test_clean_dir}" ]; then
        echo "Error: test-clean directory not found: ${test_clean_dir}"
        echo "Please download LibriTTS dataset first"
        exit 1
    fi
    
    if [ ! -f "${speakers_file}" ]; then
        echo "Error: speakers.tsv not found: ${speakers_file}"
        exit 1
    fi
    
    python3 local/prepare_vc_metadata.py \
        --test-clean-dir ${test_clean_dir} \
        --speakers-file ${speakers_file} \
        --output-path ${metadata_file}
    
    echo "Stage 1 completed"
    echo ""
fi

# ====================================================================
# Stage 2: Generate test pairs
# ====================================================================
# NOTE: This stage generates test_vc.tsv which is shared across all
#       hyperparameter experiments. Only run once!
# ====================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Generate test pairs"
    
    if [ ! -f "${metadata_file}" ]; then
        echo "Error: metadata file not found: ${metadata_file}"
        echo "Please run stage 1 first"
        exit 1
    fi
    
    balance_arg=""
    if [ "${balance_gender}" = true ]; then
        balance_arg="--balance-gender"
    fi
    
    python3 local/generate_vc_test_pairs.py \
        --metadata-file ${metadata_file} \
        --output-dir ${test_data_dir} \
        --num-pairs ${num_pairs} \
        --source-min-duration ${source_min_duration} \
        --source-max-duration ${source_max_duration} \
        --target-min-duration ${target_min_duration} \
        --target-max-duration ${target_max_duration} \
        --seed ${seed} \
        ${balance_arg}
    
    echo "Stage 2 completed"
    echo ""
fi

# ====================================================================
# Stage 3: Run VC inference
# ====================================================================
# NOTE: Stage 3-7 can be run multiple times with different hyperparameters.
#       After running Stage 1-2 once, modify hyperparameters at the top
#       of this script and run: bash run_eval_vc.sh --stage 3
# ====================================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Run Voice Conversion inference"
    
    test_vc_file="${test_data_dir}/test_vc.tsv"
    
    if [ ! -f "${test_vc_file}" ]; then
        echo "Error: test_vc.tsv not found: ${test_vc_file}"
        echo "Please run stage 2 first"
        exit 1
    fi
    
    if [ ! -d "${model_dir}" ]; then
        echo "Error: model directory not found: ${model_dir}"
        exit 1
    fi
    
    mkdir -p ${results_dir}
    
    # Create a simplified test list for inference (3-field format)
    test_vc_infer="${test_data_dir}/test_vc_infer.tsv"
    cut -f1,2,3 ${test_vc_file} > ${test_vc_infer}
    
    python3 -m zipvoice.bin.s3_infer_zipvoice \
        --model-dir ${model_dir} \
        --checkpoint-name ${checkpoint_name} \
        --vocoder-path ${vocoder_path} \
        --tokenizer s3 \
        --test-list ${test_vc_infer} \
        --res-dir ${results_dir} \
        --num-step ${num_step} \
        --guidance-scale ${guidance_scale} \
        --t-shift ${t_shift} \
        --target-rms ${target_rms}
    
    # Create soft links for source audio files (for baseline WER evaluation)
    echo "Creating soft links for source audio files..."
    while IFS=$'\t' read -r wav_name source_wav target_wav src_gender tgt_gender; do
        source_filename="${wav_name}_source.wav"
        ln -sf "$(realpath ${source_wav})" "${results_dir}/${source_filename}"
    done < "${test_vc_file}"
    echo "Created $(wc -l < ${test_vc_file}) source audio soft links"
    
    echo "Stage 3 completed"
    echo ""
fi

# ====================================================================
# Stage 4: Evaluate Speaker Similarity
# ====================================================================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Evaluate Speaker Similarity"
    
    if [ ! -d "${eval_model_dir}" ]; then
        echo "Error: evaluation model directory not found: ${eval_model_dir}"
        echo "Please download evaluation models from:"
        echo "https://huggingface.co/k2-fsa/TTS_eval_models"
        exit 1
    fi
    
    test_vc_file="${test_data_dir}/test_vc.tsv"
    similarity_output="${eval_results_dir}/similarity_details.tsv"
    
    mkdir -p ${eval_results_dir}
    
    python3 -m zipvoice.eval.speaker_similarity.s3_sim \
        --wav-path ${results_dir} \
        --test-list ${test_vc_file} \
        --model-dir ${eval_model_dir} \
        --output-path ${similarity_output}
    
    echo "Stage 4 completed"
    echo ""
fi

# ====================================================================
# Stage 5: Evaluate WER (Converted & Source Baseline)
# ====================================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Evaluate Word Error Rate (WER)"
    
    transcript_converted="${test_data_dir}/transcript_converted.tsv"
    transcript_source="${test_data_dir}/transcript_source.tsv"
    wer_converted_output="${eval_results_dir}/wer_converted_details.tsv"
    wer_source_output="${eval_results_dir}/wer_source_details.tsv"
    
    if [ ! -f "${transcript_converted}" ]; then
        echo "Error: transcript_converted.tsv not found: ${transcript_converted}"
        exit 1
    fi
    
    if [ ! -f "${transcript_source}" ]; then
        echo "Error: transcript_source.tsv not found: ${transcript_source}"
        exit 1
    fi
    
    mkdir -p ${eval_results_dir}
    
    # 5a) Evaluate WER on converted audio
    echo "  5a) Evaluating WER on converted audio..."
    python3 -m zipvoice.eval.wer.hubert \
        --wav-path ${results_dir} \
        --test-list ${transcript_converted} \
        --model-dir ${eval_model_dir} \
        --decode-path ${wer_converted_output}
    
    # 5b) Evaluate WER on source audio (baseline)
    echo "  5b) Evaluating WER on source audio (baseline)..."
    python3 -m zipvoice.eval.wer.hubert \
        --wav-path ${results_dir} \
        --test-list ${transcript_source} \
        --model-dir ${eval_model_dir} \
        --decode-path ${wer_source_output}
    
    echo "Stage 5 completed"
    echo ""
fi

# ====================================================================
# Stage 6: Evaluate UTMOS
# ====================================================================
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6: Evaluate UTMOS"
    
    utmos_output="${eval_results_dir}/utmos_details.tsv"
    
    mkdir -p ${eval_results_dir}
    
    python3 -m zipvoice.eval.mos.s3_utmos \
        --wav-path ${results_dir} \
        --model-dir ${eval_model_dir} \
        --output-path ${utmos_output}
    
    echo "Stage 6 completed"
    echo ""
fi

# ====================================================================
# Stage 7: Analyze results by gender
# ====================================================================
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Analyze results by gender combination"
    
    test_vc_file="${test_data_dir}/test_vc.tsv"
    similarity_file="${eval_results_dir}/similarity_details.tsv"
    utmos_file="${eval_results_dir}/utmos_details.tsv"
    wer_converted_file="${eval_results_dir}/wer_converted_details.tsv"
    wer_source_file="${eval_results_dir}/wer_source_details.tsv"
    analysis_output_dir="${eval_results_dir}/analysis"
    
    mkdir -p ${analysis_output_dir}
    
    # Save hyperparameter information
    hyperparam_file="${eval_results_dir}/hyperparameters.txt"
    cat > ${hyperparam_file} << EOF
exp_name=${exp_name}
num_step=${num_step}
guidance_scale=${guidance_scale}
checkpoint_name=${checkpoint_name}
t_shift=${t_shift}
target_rms=${target_rms}
model_dir=${model_dir}
num_pairs=${num_pairs}
seed=${seed}
EOF
    echo "Hyperparameters saved to: ${hyperparam_file}"
    
    python3 local/analyze_vc_results_by_gender.py \
        --test-vc-file ${test_vc_file} \
        --similarity-file ${similarity_file} \
        --utmos-file ${utmos_file} \
        --wer-file ${wer_converted_file} \
        --wer-source-file ${wer_source_file} \
        --output-dir ${analysis_output_dir}
    
    echo "Stage 7 completed"
    echo ""
fi

# ====================================================================
# Summary
# ====================================================================
echo ""
echo "========================================================================"
echo "Voice Conversion Evaluation Completed"
echo "========================================================================"
echo "Experiment: ${exp_name}"
echo ""
echo "Hyperparameters:"
echo "  num_step:        ${num_step}"
echo "  guidance_scale:  ${guidance_scale}"
echo "  checkpoint:      ${checkpoint_name}"
echo ""
echo "Results saved in: ${eval_results_dir}"
echo ""
echo "Files generated:"
echo "  - Converted audio:   ${results_dir}/"
echo "  - Similarity scores: ${eval_results_dir}/similarity_details.tsv"
echo "  - WER scores:        ${eval_results_dir}/wer_details.tsv"
echo "  - UTMOS scores:      ${eval_results_dir}/utmos_details.tsv"
echo "  - Hyperparameters:   ${eval_results_dir}/hyperparameters.txt"
echo "  - Gender analysis:   ${eval_results_dir}/analysis/gender_analysis.csv"
echo "========================================================================"

