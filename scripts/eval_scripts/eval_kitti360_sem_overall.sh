#!/bin/bash
#SBATCH --job-name=k360_sem_overall
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/eval_sem%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/eval_sem%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=48:00:00

# Overall evaluation for KITTI-360 Semantic pipeline (predict semantics + sem2video)
# - Loads the semantic predictor (Stage 1) from parent dir (auto-detects latest checkpoint)
# - Loads the semantic2video ControlNet (Stage 2) from specific checkpoint
# - Runs selection and video generation, logs to WandB
# steps completed : Steps:  46%|████▌     | 55489/121010 [66:57:30<68:59:43,  3.79s/it, lr=1e-5, step_loss=0.265] 
set -e
set -u

echo "========================================="
echo "Starting KITTI-360 Semantic Overall Evaluation"
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

# Stage 1 (semantic predictor) parent dir — auto-detect latest checkpoint inside
SEM_PRED_DIR_PARENT="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict"

# Stage 2 (semantic2video) specific checkpoint
SEM2VID_DIR="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video/checkpoint-66900"


echo "Dataset:              $DATASET"
echo "Dataset Path:         $DATASET_PATH"
echo "Semantic Predict dir: $SEM_PRED_DIR_PARENT (auto-detect latest)"
echo "Semantic2Video ckpt:  $SEM2VID_DIR"

OUT_DIR="/no_backups/s1492/Ctrl-V/outputs/kitti360_sem_overall_eval"
LOG_DIR="/no_backups/s1492/Ctrl-V/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"
PROJECT_NAME='ctrl_v_kitti360_seg_eval'

# WandB
export WANDB_ENTITY="jaseelkt1-university-of-stuttgart"
export WANDB_MODE=online

echo "Starting overall evaluation..."
START_TIME=$(date +%s)

# ----------------------------------------------------------------------------
# Run overall evaluation: predict semantics (Stage 1) → select → generate video (Stage 2)
# ----------------------------------------------------------------------------
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  tools/eval_overall.py \
  --data_root $DATASET_PATH \
  --project_name $PROJECT_NAME \
  --pretrained_model_name_or_path $SEM2VID_DIR \
  --output_dir $OUT_DIR \
  --variant fp16 \
  --dataset_name $DATASET \
  --report_to wandb \
  --seed 123 \
  --mixed_precision fp16 \
  --clip_length 25 \
  --min_guidance_scale 1.0 \
  --max_guidance_scale 5.0 \
  --noise_aug_strength 0.01 \
  --train_batch_size 1 \
  --num_demo_samples 200 \
  --resume_from_checkpoint latest \
  --num_inference_steps 50 \
  --pretrained_bbox_model $SEM_PRED_DIR_PARENT \
  --num_cond_bbox_frames 1 \
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
