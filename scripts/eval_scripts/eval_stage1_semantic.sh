#!/bin/bash
#SBATCH --job-name=eval_stage1_sem
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/eval_stage1_sem_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/eval_stage1_sem_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=rtx_a5000:1
#SBATCH --partition=stud
#SBATCH --qos=batch
#SBATCH --time=04:00:00

set -e
set -u

echo "========================================="
echo "Stage 1 Evaluation: Semantic Prediction Quality"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-localhost}"
echo "Started at: $(date)"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

cd /usrhomes/s1492/Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"
python -c "import ctrlv, os; print('✓ ctrlv path:', os.path.dirname(ctrlv.__file__))"

echo "GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# ============================================================================
# Evaluation Configuration
# ============================================================================

# Stage 1 checkpoint directory (will auto-detect latest checkpoint inside)
CHECKPOINT_DIR="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae"

# Output directory for evaluation results
OUTPUT_DIR="/no_backups/s1492/Ctrl-V/outputs/eval_stage1_semantic"

# Number of video clips to evaluate (each clip = 25 frames)
NUM_SAMPLES=170

# Number of videos to save visualized frames for
NUM_SAVE_VIDEOS=10

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Checkpoint dir:    $CHECKPOINT_DIR"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Num eval samples:  $NUM_SAMPLES"
echo "  Num save videos:   $NUM_SAVE_VIDEOS"
echo "  Clip length:       25"
echo "  Resolution:        192x704"
echo ""

# ============================================================================
# Run Evaluation
# ============================================================================

echo "Starting evaluation..."
START_TIME=$(date +%s)

python tools/eval_stage1_semantic.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --output_dir $OUTPUT_DIR \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --dataset_name kitti360 \
    --data_root "" \
    --clip_length 25 \
    --train_H 192 \
    --train_W 704 \
    --num_workers 4 \
    --num_samples $NUM_SAMPLES \
    --num_inference_steps 30 \
    --min_guidance_scale 3.0 \
    --max_guidance_scale 7.0 \
    --noise_aug_strength 0.01 \
    --fps 7 \
    --seed 1234 \
    --num_cond_bbox_frames 1 \
    --save_frames \
    --num_save_videos $NUM_SAVE_VIDEOS

# ============================================================================
# Post-Evaluation Summary
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================="
echo "Evaluation Summary"
echo "========================================="
echo "Job ID:           ${SLURM_JOB_ID:-interactive}"
echo "Started:          $(date -d @${START_TIME})"
echo "Finished:         $(date)"
echo "Duration:         ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results:"
echo "  JSON:              ${OUTPUT_DIR}/eval_results.json"
echo "  Summary:           ${OUTPUT_DIR}/eval_summary.txt"
echo "  Confusion matrix:  ${OUTPUT_DIR}/confusion_matrix.png"
echo "  Frames:            ${OUTPUT_DIR}/frames/"
echo "  Legend:             ${OUTPUT_DIR}/class_legend.png"
echo ""

# Print the summary if it exists
if [ -f "${OUTPUT_DIR}/eval_summary.txt" ]; then
    echo "--- Results ---"
    cat "${OUTPUT_DIR}/eval_summary.txt"
fi

echo "========================================="
echo "✓ Evaluation complete!"
