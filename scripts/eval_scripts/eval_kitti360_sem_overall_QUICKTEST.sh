#!/bin/bash
#SBATCH --job-name=kitti360_sem_eval_QUICKTEST
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=highperf
#SBATCH --time=00:30:00

###############################################################################
# QUICK TEST VERSION - Evaluates only 1-2 clips to verify fix
# This is a test version of eval_kitti360_sem_overall.sh
# Once verified, run the full version
###############################################################################

# Create output directory FIRST (before SLURM tries to write logs)
OUT_DIR="/no_backups/s1492/Ctrl-V/outputs/kitti360_sem_overall_eval_quicktest"
mkdir -p "$OUT_DIR"

# Now redirect output/error to the directory we just created
exec 1>"$OUT_DIR/slurm_${SLURM_JOB_ID}.out"
exec 2>"$OUT_DIR/slurm_${SLURM_JOB_ID}.err"

echo "=========================================="
echo "QUICK TEST - Evaluating 2 clips only"
echo "=========================================="

# HuggingFace cache (needed for model loading)
export HF_HOME=/no_backups/s1492/.cache/huggingface
export HF_HUB_CACHE=/no_backups/s1492/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/no_backups/s1492/.cache/huggingface/transformers

# Activate env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti

echo "✓ Conda environment 'kitti' activated"


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

# Paths
DATASET_PATH="/no_backups/s1492/"
DATASET="kitti360"
PROJECT_NAME="kitti360_sem_overall_eval_QUICKTEST"

# Model paths
SEM_PRED_DIR_PARENT="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict"
SEM2VID_DIR="/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video/checkpoint-77300"

# WandB
export WANDB_ENTITY="jaseelkt1-university-of-stuttgart"
export WANDB_MODE=online

echo "Using semantic predictor: $SEM_PRED_DIR_PARENT"
echo "Using semantic2video: $SEM2VID_DIR"

echo "Starting evaluation..."
START_TIME=$(date +%s)

###############################################################################
# QUICK TEST: Only 2 samples 
###############################################################################
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
  --num_demo_samples 1 \
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
echo "=========================================="
echo "QUICK TEST COMPLETE!"
echo "=========================================="
echo ""
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
echo "========================================="
echo "Check outputs in: $OUT_DIR/wandb/latest-run/files/media/images/"
echo ""
echo "Verify the fix worked:"
echo "  cd /usrhomes/s1492/Ctrl-V-seg"
echo "  python scripts/eval_scripts/verify_fix.py $OUT_DIR"
echo ""
echo "If successful, run full evaluation:"
echo "  sbatch scripts/eval_scripts/eval_kitti360_sem_overall.sh"
echo ""
