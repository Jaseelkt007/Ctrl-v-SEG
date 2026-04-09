#!/bin/bash
#SBATCH --job-name=eval_stage2_drn_ms
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/eval_stage2_drn_ms_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/eval_stage2_drn_ms_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=48:00:00

# =============================================================================
# eval_stage2_drn_ms.sh
#
# Multi-scale DRN mIoU evaluation for Stage 2 (Semantic → RGB).
# Matches the drn_d_105_MIoU evaluation methodology:
#   - Same 19 clip groups from the reference val_labels.txt
#   - Multi-scale DRN inference (6 scales: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
#   - Confidence-weighted confusion matrix
#
# Two-phase pipeline:
#   Phase 1  tools/generate_stage2_frames_for_drn.py
#            Runs Stage 2 ControlNet and saves predicted RGB frames to disk.
#            Writes val_images.txt + val_labels.txt for DRN consumption.
#
#   Phase 2  drn_eval/segment.py test --ms
#            Runs multi-scale DRN on the saved frames and computes mIoU.
#
# DO NOT modify eval_stage2_rgb.sh — this script provides the
# reference-matching evaluation as a separate, self-contained pipeline.
# =============================================================================

set -e
set -u

echo "============================================================"
echo " Stage 2 DRN Multi-Scale mIoU Evaluation"
echo " Methodology: drn_d_105_MIoU (multi-scale + confidence-weighted)"
echo " Metrics:     DRN-mIoU (multi-scale, confidence-weighted)"
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
# Configuration  —  edit these variables to switch checkpoints / output dirs
# ============================================================================

CHECKPOINT_DIR="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_sem2video_unet_unfreeze_reinject/checkpoint-32700"

# All Phase 1 outputs go here.  Phase 2 uses:  ${OUTPUT_DIR}/CTRLV_STAGE2
OUTPUT_DIR="/no_backups/s1492/Ctrl-V/outputs/eval_stage2_drn_ms_unet_unfreeze_with_reinject"

# GT labels: drn_eval/CTRLV_STAGE2/val_labels.txt (static, checked into project)
# Phase 1 reads this file directly via --drn_eval_dir; no separate argument needed.

# DRN paths — use the project's drn_eval/ directory
DRN_EVAL_DIR="${PROJECT_DIR}/drn_eval"
DRN_CHECKPOINT="${DRN_EVAL_DIR}/KITTI360_checkpoints/checkpoint_030.pth.tar"

# Stage 2 inference settings  (match eval_stage2_rgb.sh)
CLIP_LENGTH=25
TRAIN_H=192
TRAIN_W=704
NUM_INFERENCE_STEPS=30
MIN_GUIDANCE=1.0
MAX_GUIDANCE=3.0
CONDITIONING_SCALE=1.0
NOISE_AUG=0.01
FPS=7
SEED=1234

mkdir -p "$OUTPUT_DIR"
mkdir -p /no_backups/s1492/Ctrl-V/logs

echo ""
echo "Configuration:"
echo "  Checkpoint       : $CHECKPOINT_DIR"
echo "  Output           : $OUTPUT_DIR"
echo "  DRN eval dir     : $DRN_EVAL_DIR"
echo "  GT val labels    : ${DRN_EVAL_DIR}/CTRLV_STAGE2/val_labels.txt"
echo "  Clips            : 19 groups × ${CLIP_LENGTH} frames = ~$((19 * CLIP_LENGTH)) frames"
echo "  Resolution       : ${TRAIN_H}×${TRAIN_W}"
echo "  Inference steps  : $NUM_INFERENCE_STEPS"
echo ""

# ============================================================================
# Phase 1 — Generate Stage 2 RGB frames and write txt manifests
# ============================================================================

echo "============================================================"
echo " Phase 1: Stage 2 inference → save frames to disk"
echo "============================================================"
PHASE1_START=$(date +%s)

python tools/generate_stage2_frames_for_drn.py \
    --checkpoint_dir               "$CHECKPOINT_DIR" \
    --output_dir                   "$OUTPUT_DIR" \
    --kitti360_root                /misc/data/public/kitti-360/KITTI-360 \
    --drn_eval_dir                 "$DRN_EVAL_DIR" \
    --clip_length                  $CLIP_LENGTH \
    --train_H                      $TRAIN_H \
    --train_W                      $TRAIN_W \
    --num_inference_steps          $NUM_INFERENCE_STEPS \
    --min_guidance_scale           $MIN_GUIDANCE \
    --max_guidance_scale           $MAX_GUIDANCE \
    --conditioning_scale           $CONDITIONING_SCALE \
    --noise_aug_strength           $NOISE_AUG \
    --fps                          $FPS \
    --seed                         $SEED

PHASE1_END=$(date +%s)
PHASE1_DUR=$(( PHASE1_END - PHASE1_START ))
echo ""
echo "✓ Phase 1 done in $(( PHASE1_DUR / 60 ))m $(( PHASE1_DUR % 60 ))s"

# Verify outputs exist
CTRLV_STAGE2_DIR="${OUTPUT_DIR}/CTRLV_STAGE2"
if [ ! -f "${CTRLV_STAGE2_DIR}/val_images.txt" ]; then
    echo "ERROR: val_images.txt not found at ${CTRLV_STAGE2_DIR}"
    exit 1
fi
if [ ! -f "${CTRLV_STAGE2_DIR}/val_labels.txt" ]; then
    echo "ERROR: val_labels.txt not found at ${CTRLV_STAGE2_DIR}"
    exit 1
fi

N_IMAGES=$(wc -l < "${CTRLV_STAGE2_DIR}/val_images.txt")
N_LABELS=$(wc -l < "${CTRLV_STAGE2_DIR}/val_labels.txt")
echo "  val_images.txt : $N_IMAGES entries"
echo "  val_labels.txt : $N_LABELS entries"

# Quick sanity check: verify first generated frame and GT label exist
FIRST_IMG=$(head -1 "${CTRLV_STAGE2_DIR}/val_images.txt")
FIRST_LBL=$(head -1 "${CTRLV_STAGE2_DIR}/val_labels.txt")
echo ""
echo "  Sanity check:"
echo "    First image  : ${CTRLV_STAGE2_DIR}/${FIRST_IMG}"
echo "    First label  : ${FIRST_LBL}"
if [ ! -f "${CTRLV_STAGE2_DIR}/${FIRST_IMG}" ]; then
    echo "ERROR: First generated frame not found!"
    exit 1
fi
if [ ! -f "${FIRST_LBL}" ]; then
    echo "ERROR: First GT label not found!"
    exit 1
fi

# Derive and check confidence map for first label
FIRST_CONF="${FIRST_LBL//semantic/confidence}"
echo "    First confidence: ${FIRST_CONF}"
if [ ! -f "${FIRST_CONF}" ]; then
    echo "WARNING: First confidence map not found — confidence weighting will be disabled."
    echo "         segment.py will fall back to standard (unweighted) mIoU."
else
    echo "    ✓ Confidence maps accessible"
fi

echo ""

# ============================================================================
# Phase 2 — Multi-scale DRN inference and mIoU computation
# ============================================================================

echo "============================================================"
echo " Phase 2: Multi-scale DRN mIoU  (segment.py test --ms)"
echo "============================================================"
PHASE2_START=$(date +%s)

cd "$DRN_EVAL_DIR"

# segment.py writes its log to a file named:
#   drn_d_105_000_val_ms/   (a subdirectory created by segment.py itself)
# We also tee stdout to our own log file for convenience.
DRN_LOG="${OUTPUT_DIR}/drn_ms_eval.log"

python segment.py test \
    -d  "${CTRLV_STAGE2_DIR}" \
    -c  19 \
    --arch     drn_d_105 \
    --pretrained "${DRN_CHECKPOINT}" \
    --phase    val \
    --batch-size 1 \
    --ms \
    2>&1 | tee "$DRN_LOG"

cd "$PROJECT_DIR"

PHASE2_END=$(date +%s)
PHASE2_DUR=$(( PHASE2_END - PHASE2_START ))
echo ""
echo "✓ Phase 2 done in $(( PHASE2_DUR / 60 ))m $(( PHASE2_DUR % 60 ))s"

# ============================================================================
# Phase 3 — Generate structured report + confusion matrix
# ============================================================================

echo "============================================================"
echo " Phase 3: Generate eval report + confusion matrix"
echo "============================================================"

TOTAL_END=$(date +%s)
TOTAL_DUR=$(( TOTAL_END - PHASE1_START ))
HOURS=$(( TOTAL_DUR / 3600 ))
MINS=$(( (TOTAL_DUR % 3600) / 60 ))
SECS=$(( TOTAL_DUR % 60 ))

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

python tools/generate_drn_ms_report.py \
    --output_dir   "$OUTPUT_DIR" \
    --job_id       "${SLURM_JOB_ID:-interactive}" \
    --node         "${SLURM_NODELIST:-localhost}" \
    --gpu          "$GPU_NAME" \
    --phase1_dur   "$(( PHASE1_DUR / 60 ))m $(( PHASE1_DUR % 60 ))s" \
    --phase2_dur   "$(( PHASE2_DUR / 60 ))m $(( PHASE2_DUR % 60 ))s" \
    --total_dur    "${HOURS}h ${MINS}m ${SECS}s"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo " Evaluation complete"
echo "   Duration      : ${HOURS}h ${MINS}m ${SECS}s"
echo "   Generated RGB : ${CTRLV_STAGE2_DIR}/generated_frames/"
echo "   val_images    : ${CTRLV_STAGE2_DIR}/val_images.txt  ($N_IMAGES frames)"
echo "   val_labels    : ${CTRLV_STAGE2_DIR}/val_labels.txt  ($N_LABELS frames)"
echo "   DRN log       : ${DRN_LOG}"
echo "   Report        : ${OUTPUT_DIR}/eval_report.txt"
echo "   CM npy        : ${OUTPUT_DIR}/confusion_matrix_drn.npy"
echo "   CM png        : ${OUTPUT_DIR}/confusion_matrix_drn.png"
echo "   Metadata      : ${OUTPUT_DIR}/metadata.json"
echo "============================================================"

echo ""
echo "✓ Done"
