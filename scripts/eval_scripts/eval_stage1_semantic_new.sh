#!/bin/bash
#SBATCH --job-name=eval_stage1_sem_new
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/eval_stage1_sem_new_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/eval_stage1_sem_new_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=rtx_a5000:1
#SBATCH --partition=stud
#SBATCH --qos=batch
#SBATCH --time=48:00:00

# =============================================================================
# eval_stage1_semantic_new.sh  [NOT USED - IGNORED]
#
# Stage 1 (RGB → Semantic) evaluation on the same fixed clip set used by the
# Stage 2 DRN mIoU evaluation (drn_eval/CTRLV_STAGE2/val_labels.txt).
# Enables direct comparison between Stage 1 and Stage 2 results.
#
# Two modes (set USE_CONFIDENCE_WEIGHTING below):
#   OFF  — standard unweighted mIoU (same as eval_stage1_semantic.sh)
#   ON   — confidence-weighted mIoU (matches Stage 2 DRN methodology)
# =============================================================================

set -e
set -u

echo "============================================================"
echo " Stage 1 Semantic Evaluation  (fixed clip set)"
echo " Clip set : drn_eval/CTRLV_STAGE2/val_labels.txt"
echo "============================================================"
echo " Job ID  : ${SLURM_JOB_ID:-interactive}"
echo " Node    : ${SLURM_NODELIST:-localhost}"
echo " Start   : $(date)"
echo ""

# ============================================================================
# Environment
# ============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

PROJECT_DIR="/usrhomes/s1492/Ctrl-V-seg"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
python -c "import ctrlv, os; print('✓ ctrlv:', os.path.dirname(ctrlv.__file__))"

echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv

# ============================================================================
# Configuration  —  edit these variables to switch checkpoints / modes
# ============================================================================

CHECKPOINT_DIR="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae"
OUTPUT_DIR="/no_backups/s1492/Ctrl-V/outputs/eval_stage1_semantic_new"

DRN_EVAL_DIR="${PROJECT_DIR}/drn_eval"

# Set to "--use_confidence_weighting" to enable, or "" to disable
# USE_CONFIDENCE_WEIGHTING=""
USE_CONFIDENCE_WEIGHTING="--use_confidence_weighting"

# Set to "--save_frames" to save GT vs predicted visualisations, or "" to skip
SAVE_FRAMES=""
# SAVE_FRAMES="--save_frames"

# Inference settings
CLIP_LENGTH=25
TRAIN_H=192
TRAIN_W=704
NUM_INFERENCE_STEPS=30
MIN_GUIDANCE=3.0
MAX_GUIDANCE=7.0
NOISE_AUG=0.01
FPS=7
SEED=1234

mkdir -p "$OUTPUT_DIR"
mkdir -p /no_backups/s1492/Ctrl-V/logs

echo ""
echo "Configuration:"
echo "  Checkpoint         : $CHECKPOINT_DIR"
echo "  Output             : $OUTPUT_DIR"
echo "  Clip set           : ${DRN_EVAL_DIR}/CTRLV_STAGE2/val_labels.txt"
echo "  Clips              : 19 groups × ${CLIP_LENGTH} frames = $((19 * CLIP_LENGTH)) frames"
echo "  Resolution         : ${TRAIN_H}×${TRAIN_W}"
echo "  Inference steps    : $NUM_INFERENCE_STEPS"
echo "  Confidence weight  : ${USE_CONFIDENCE_WEIGHTING:-OFF}"
echo ""

# ============================================================================
# Run Evaluation
# ============================================================================

START_TIME=$(date +%s)

python tools/eval_stage1_semantic_new.py \
    --checkpoint_dir               "$CHECKPOINT_DIR" \
    --output_dir                   "$OUTPUT_DIR" \
    --drn_eval_dir                 "$DRN_EVAL_DIR" \
    --kitti360_root                /misc/data/public/kitti-360/KITTI-360 \
    --clip_length                  $CLIP_LENGTH \
    --train_H                      $TRAIN_H \
    --train_W                      $TRAIN_W \
    --num_inference_steps          $NUM_INFERENCE_STEPS \
    --min_guidance_scale           $MIN_GUIDANCE \
    --max_guidance_scale           $MAX_GUIDANCE \
    --noise_aug_strength           $NOISE_AUG \
    --fps                          $FPS \
    --seed                         $SEED \
    $USE_CONFIDENCE_WEIGHTING \
    $SAVE_FRAMES

# ============================================================================
# Summary
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))
SECS=$((DURATION % 60))

echo ""
echo "============================================================"
echo " Evaluation complete"
echo "   Duration      : ${HOURS}h ${MINS}m ${SECS}s"
echo "   Report        : ${OUTPUT_DIR}/eval_report.txt"
echo "   JSON          : ${OUTPUT_DIR}/eval_results.json"
echo "   Confusion mat : ${OUTPUT_DIR}/confusion_matrix.png"
echo "============================================================"

if [ -f "${OUTPUT_DIR}/eval_report.txt" ]; then
    echo ""
    echo "--- Results ---"
    grep -E "mIoU|Pixel Accuracy|Mean Accuracy" "${OUTPUT_DIR}/eval_report.txt" | head -10
fi

echo ""
echo "✓ Done"
