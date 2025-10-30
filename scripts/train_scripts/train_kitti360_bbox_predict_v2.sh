#!/bin/bash
#SBATCH --job-name=kitti360_bbox_train
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/train_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/train_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse19

# Training script for KITTI-360 Bbox Prediction (BDD100K format)
# Usage:
#   Interactive: bash scripts/train_scripts/train_kitti360_bbox_predict_v2.sh
#   Batch:       sbatch scripts/train_scripts/train_kitti360_bbox_predict_v2.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Environment Setup
# ============================================================================

echo "========================================="
echo "Starting KITTI360 Bbox Prediction Training"
echo "========================================="
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job ID: $SLURM_JOB_ID"
    echo "Node: $SLURM_NODELIST"
fi
echo "Started at: $(date)"
echo ""

# Remove old CUDA 10.x paths
export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'cuda-10' | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$(echo ${LD_LIBRARY_PATH:-} | tr ':' '\n' | grep -v 'cuda-10' | tr '\n' ':' | sed 's/:$//')
echo "✓ Cleaned old CUDA 10.x paths from environment"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

# Install ctrlv package if needed
cd /usrhomes/s1492/Ctrl-V
if ! python -c "import ctrlv" 2>/dev/null; then
    echo "Installing ctrlv package..."
    pip install -e . --no-deps
    echo "✓ ctrlv package installed"
else
    echo "✓ ctrlv package already installed"
fi

# Verify PyTorch CUDA
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✓ PyTorch CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))" 2>/dev/null; then
    echo "✗ ERROR: PyTorch cannot access GPU!"
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    exit 1
fi

echo "GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
echo ""

timestamp=$(date +%y%m%d_%H%M%S)

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET="kitti360"  # Uses new KITTI-360 BDD format loader
DATASET_PATH="/no_backups/s1492/"  # Parent directory of kitti360_ctrlv/

# ============================================================================
# Experiment Configuration
# ============================================================================
NAME="kitti360_bbox_predict_${timestamp}"
CHECKPOINT_DIR="/no_backups/s1492/Ctrl-V/checkpoints/${NAME}"
OUT_DIR="/no_backups/s1492/Ctrl-V/outputs/${NAME}"
LOG_DIR="/no_backups/s1492/Ctrl-V/logs"

mkdir -p $CHECKPOINT_DIR
mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

PROJECT_NAME='ctrl_v_kitti360'

# Save this script for reproducibility
SCRIPT_PATH=$0
SAVE_SCRIPT_PATH="${OUT_DIR}/train_script.sh"
cp $SCRIPT_PATH $SAVE_SCRIPT_PATH
echo "Saved script to ${SAVE_SCRIPT_PATH}"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
echo "Outputs:     ${OUT_DIR}/"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "SLURM logs:  ${LOG_DIR}/train_${SLURM_JOB_ID}.{out,err}"
fi
echo ""

# ============================================================================
# Training Execution
# ============================================================================

echo "Working directory: $(pwd)"
echo ""

# Accelerate and WandB Configuration
export ACCELERATE_USE_FSDP=0
export ACCELERATE_MIXED_PRECISION=fp16
export WANDB_ENTITY="jaseelkt1-university-of-stuttgart"
export WANDB_MODE=online
echo "✓ WandB configured (entity: jaseelkt1-university-of-stuttgart)"

echo "Starting training..."
START_TIME=$(date +%s)

CUDA_LAUNCH_BLOCKING=1 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --dynamo_backend no \
    tools/train_video_diffusion.py \
    --run_name $NAME \
    --data_root $DATASET_PATH \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --output_dir $CHECKPOINT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --train_batch_size 1 \
    --learning_rate 5e-6 \
    --checkpoints_total_limit 2 \
    --checkpointing_steps 200 \
    --gradient_accumulation_steps 6 \
    --dataloader_num_workers 4 \
    --validation_steps 100 \
    --enable_gradient_checkpointing \
    --lr_scheduler constant \
    --report_to wandb \
    --seed 1234 \
    --mixed_precision fp16 \
    --clip_length 8 \
    --min_guidance_scale 3 \
    --max_guidance_scale 7 \
    --noise_aug_strength 0.01 \
    --bbox_dropout_prob 0.1 \
    --conditioning_dropout_prob 0.0 \
    --num_demo_samples 10 \
    --backprop_temporal_blocks_start_iter -1 \
    --num_train_epochs 1 \
    --predict_bbox \
    --num_inference_steps 30 \
    --num_cond_bbox_frames 3 \
    --train_H 376 \
    --train_W 1408
    # --resume_from_checkpoint latest  # Uncomment to resume training
    # --if_last_frame_trajectory  # Uncomment to use trajectory instead of last bbox

# ============================================================================
# Post-Training Cleanup
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "Training complete!"
echo "Copying validation plots to outputs directory..."
if [ -d "${CHECKPOINT_DIR}/plots" ]; then
    cp -r "${CHECKPOINT_DIR}/plots" "${OUT_DIR}/"
    echo "✓ Validation plots copied to: ${OUT_DIR}/plots/"
fi

echo ""
echo "========================================="
echo "Training Summary"
echo "========================================="
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job ID:           ${SLURM_JOB_ID}"
fi
echo "Run Name:         ${NAME}"
echo "Started:          $(date -d @${START_TIME} 2>/dev/null || date -r ${START_TIME})"
echo "Finished:         $(date)"
echo "Duration:         ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Checkpoints:      ${CHECKPOINT_DIR}/"
echo "Outputs & Plots:  ${OUT_DIR}/"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "SLURM Logs:       ${LOG_DIR}/train_${SLURM_JOB_ID}.{out,err}"
fi
echo ""
echo "WandB Project:    ${PROJECT_NAME}"
echo "WandB URL:        https://wandb.ai/jaseelkt1-university-of-stuttgart/${PROJECT_NAME}/runs/${NAME}"
echo "========================================="
echo ""
echo "✓ Training completed successfully!"
