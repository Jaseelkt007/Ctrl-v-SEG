#!/bin/bash
#SBATCH --job-name=k360_sem2video_eval_tf
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/eval_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/eval_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=36:00:00

# Teacher-Forced evaluation for KITTI-360 Semantic->Video (Ctrl-V-seg)
# Uses GT semantic maps as control to generate videos with the trained ControlNet.
# This script auto-selects the latest checkpoint under kitti360_semantic2video.

set -e
set -u

echo "========================================="
echo "Starting KITTI-360 Semantic2Video Teacher-Forced Evaluation"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-localhost}"
echo "Started at: $(date)"
echo ""

# Activate env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti

echo "✓ Conda environment 'kitti' activated"

# Prefer Ctrl-V-seg package
cd /usrhomes/s1492/Ctrl-V-seg
if ! python -c "import ctrlv" 2>/dev/null; then
  echo "Installing ctrlv (seg) package..."
  pip install -e . --no-deps
fi

# Ensure we import from Ctrl-V-seg/src first
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"
python -c "import ctrlv, os; print('✓ ctrlv path:', os.path.dirname(ctrlv.__file__))"

echo "GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATASET='kitti360'
DATASET_PATH="/no_backups/s1492/"   # parent containing kitti360_ctrlv/

# Trained semantic2video checkpoints live in Ctrl-V (shared checkpoints root)
CKPT_ROOT="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video"

# Resolve latest checkpoint subfolder (checkpoint-XXXX)
if [ -d "$CKPT_ROOT" ]; then
  LATEST_CKPT=$(ls -1 "$CKPT_ROOT" | grep -E '^checkpoint-[0-9]+' | sed 's/checkpoint-//' | sort -n | tail -n1 || true)
else
  echo "✗ ERROR: Checkpoint root not found: $CKPT_ROOT"; exit 1
fi

if [ -z "${LATEST_CKPT:-}" ]; then
  echo "✗ ERROR: No checkpoint-* folders found in $CKPT_ROOT"; exit 1
fi
BOX2VIDEO_DIR="$CKPT_ROOT/checkpoint-${LATEST_CKPT}"

# Sanity check required subfolders
if [ ! -f "$BOX2VIDEO_DIR/unet/config.json" ]; then
  echo "✗ ERROR: Missing $BOX2VIDEO_DIR/unet/config.json"; exit 1
fi
if [ ! -f "$BOX2VIDEO_DIR/control_net/config.json" ]; then
  echo "✗ ERROR: Missing $BOX2VIDEO_DIR/control_net/config.json"; exit 1
fi

echo "Dataset:              $DATASET"
echo "Dataset Path:         $DATASET_PATH"
echo "Semantic2Video Ckpt:  $BOX2VIDEO_DIR"

OUT_DIR="/no_backups/s1492/Ctrl-V/outputs/kitti360_sem2video_eval_tf"
LOG_DIR="/no_backups/s1492/Ctrl-V/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"
PROJECT_NAME='ctrl_v_kitti360_seg_eval'

# WandB
export WANDB_ENTITY="jaseelkt1-university-of-stuttgart"
export WANDB_MODE=online

echo "Starting evaluation..."
START_TIME=$(date +%s)

# ----------------------------------------------------------------------------
# Run teacher-forced evaluation with segmentation conditioning
# ----------------------------------------------------------------------------
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  tools/eval_video_controlnet.py \
  --run_name kitti360-sem2video-tf-eval \
  --data_root $DATASET_PATH \
  --project_name $PROJECT_NAME \
  --pretrained_model_name_or_path $BOX2VIDEO_DIR \
  --output_dir $OUT_DIR \
  --variant fp16 \
  --dataset_name $DATASET \
  --report_to wandb \
  --seed 123 \
  --mixed_precision fp16 \
  --clip_length 25 \
  --min_guidance_scale 1.0 \
  --max_guidance_scale 3.0 \
  --noise_aug_strength 0.01 \
  --bbox_dropout_prob 0.1 \
  --num_demo_samples 200 \
  --num_inference_steps 30 \
  --conditioning_scale 1.0 \
  --train_batch_size 1 \
  --resume_from_checkpoint latest \
  --use_segmentation \
  --train_H 128 \
  --train_W 512

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
echo "Results saved to: ${OUT_DIR}/"
echo "WandB Project:    ${PROJECT_NAME}"
echo "Generated videos: ${OUT_DIR}/wandb/run-*/files/media/videos/"
echo "========================================="
