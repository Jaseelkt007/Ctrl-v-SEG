#!/bin/bash
#SBATCH --job-name=eval_stage2_rgb
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/eval_stage2_rgb_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/eval_stage2_rgb_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=rtx_a5000:1
#SBATCH --partition=stud
#SBATCH --qos=batch
#SBATCH --time=24:00:00

set -e
set -u

echo "========================================="
echo "Stage 2 Evaluation: Semantic-to-RGB"
echo "Metrics: DRN-mIoU, FID, FVD-I3D,"
echo "         FVD-VideoMAE, LPIPS, SSIM, PSNR"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node:   ${SLURM_NODELIST:-localhost}"
echo "Start:  $(date)"
echo ""

# ============================================================================
# Environment
# ============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

cd /usrhomes/s1492/Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"
python -c "import ctrlv, os; print('✓ ctrlv:', os.path.dirname(ctrlv.__file__))"

echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv

# ============================================================================
# Configuration  —  edit these two lines to switch checkpoints
# ============================================================================

CHECKPOINT_DIR="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_sem2video_unet_unfreeze"
OUTPUT_DIR="/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb_unet_unfreeze"

# Evaluate the full non-overlapping val set (487 clips from 9 sequences).
# Reduce NUM_SAMPLES for a quick smoke-test (e.g. 20).
NUM_SAMPLES=487

DRN_DIR="/usrhomes/s1492/drn"
DRN_CHECKPOINT="/usrhomes/s1492/drn/KITTI360_checkpoints/checkpoint_030.pth.tar"
DRN_INFO_JSON="/usrhomes/s1492/drn/CTRLV_BBOX/info.json"

mkdir -p "$OUTPUT_DIR"
mkdir -p /no_backups/s1492/Ctrl-V/logs

echo ""
echo "Configuration:"
echo "  Checkpoint : $CHECKPOINT_DIR"
echo "  Output     : $OUTPUT_DIR"
echo "  Num clips  : $NUM_SAMPLES  (val split, non-overlapping)"
echo "  Resolution : 192x704  |  clip_length=25"
echo ""

# ============================================================================
# Run evaluation  (all metrics in one pass)
# ============================================================================

echo "Starting evaluation..."
START_TIME=$(date +%s)

python tools/eval_stage2_rgb.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --dataset_name kitti360 \
    --data_root "" \
    --clip_length 25 \
    --train_H 192 \
    --train_W 704 \
    --num_workers 4 \
    --num_samples $NUM_SAMPLES \
    --num_inference_steps 30 \
    --min_guidance_scale 1.0 \
    --max_guidance_scale 3.0 \
    --conditioning_scale 1.0 \
    --noise_aug_strength 0.01 \
    --fps 7 \
    --seed 1234 \
    --drn_dir        "$DRN_DIR" \
    --drn_checkpoint "$DRN_CHECKPOINT" \
    --drn_info_json  "$DRN_INFO_JSON" \
    --drn_arch drn_d_105

# ============================================================================
# Post-run summary
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECS=$((DURATION % 60))

echo ""
echo "========================================="
echo "Job finished"
echo "  Duration : ${HOURS}h ${MINUTES}m ${SECS}s"
echo "  Outputs  :"
echo "    JSON             : ${OUTPUT_DIR}/eval_results.json"
echo "    Summary          : ${OUTPUT_DIR}/eval_summary.txt"
echo "    Confusion matrix : ${OUTPUT_DIR}/confusion_matrix_drn.png"
echo "========================================="

if [ -f "${OUTPUT_DIR}/eval_summary.txt" ]; then
    echo ""
    echo "--- eval_summary.txt ---"
    cat "${OUTPUT_DIR}/eval_summary.txt"
fi

echo ""
echo "✓ Evaluation complete!"
