#!/bin/bash
#SBATCH --job-name=S2_unet_unfreeze_reinject
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/train_s2_unet_unfreeze_reinject_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/train_s2_unet_unfreeze_reinject_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=96:00:00


set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Environment Setup
# ============================================================================

echo "========================================="
echo "Stage 2: Partial UNet Unfreeze + Semantic VAE Scaling"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Remove all CUDA 10.x paths that were added multiple times
export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'cuda-10' | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'cuda-10' | tr '\n' ':' | sed 's/:$//')
echo "Cleaned old CUDA 10.x paths from environment"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "Conda environment 'kitti' activated"

# Install ctrlv package in editable mode (if not already installed)
cd /usrhomes/s1492/Ctrl-V-seg
if ! python -c "import ctrlv" 2>/dev/null; then
    echo "Installing ctrlv package..."
    pip install -e . --no-deps
    echo "ctrlv package installed"
else
    echo "ctrlv package already installed"
fi

echo "GPUs in use: $(nvidia-smi --query-gpu=index --format=csv | grep -v index | tr '\n' ',' | sed 's/,$//')"

# PyTorch bundles CUDA 12.1 runtime
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('PyTorch CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))" 2>/dev/null; then
    echo "ERROR: PyTorch cannot access GPU!"
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    exit 1
fi

# Verify GPU
echo "GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# ============================================================================
# Training Configuration
# ============================================================================

DATASET="kitti360"
NAME="kitti360_sem2video_unet_unfreeze_reinject"

FINETUNED_SVD_PATH=""
PRETRAINED_MODEL_NAME_OR_PATH="stabilityai/stable-video-diffusion-img2vid-xt"

CHECKPOINT_DIR="/no_backups/s1492/Ctrl-V/checkpoints/${NAME}"
OUT_DIR="/no_backups/s1492/Ctrl-V/outputs/${NAME}"
LOG_DIR="/no_backups/s1492/Ctrl-V/logs"

mkdir -p $CHECKPOINT_DIR
mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

PROJECT_NAME='ctrl_v_kitti360'

SCRIPT_PATH=$0
SAVE_SCRIPT_PATH="${OUT_DIR}/train_script.sh"
cp $SCRIPT_PATH $SAVE_SCRIPT_PATH
echo "Saved script to ${SAVE_SCRIPT_PATH}"

echo "Checkpoints will be saved to: ${CHECKPOINT_DIR}"
echo "Logs and outputs will be saved to: ${OUT_DIR}"
echo ""
echo "--- Experiment Config ---"
echo "  ControlNet LR: 1e-5"
echo "  UNet LR:       1e-6 (mid_block + up_blocks unfrozen)"
echo "  Semantic VAE:  latents scaled by vae.config.scaling_factor"
echo "-------------------------"
echo ""

# ============================================================================
# Training Execution
# ============================================================================

cd /usrhomes/s1492/Ctrl-V-seg
echo "Working directory: $(pwd)"
echo ""

# Force Python to import ctrlv from Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"
echo "PYTHONPATH set to prioritize Ctrl-V-seg"

# Verify which ctrlv will be imported
python -c "import ctrlv, os; print('ctrlv will be imported from:', os.path.dirname(ctrlv.__file__))"
echo ""

# Create minimal accelerate config to avoid warnings
export ACCELERATE_USE_FSDP=0
export ACCELERATE_MIXED_PRECISION=fp16

# WandB Configuration
export WANDB_ENTITY="jaseelkt1-university-of-stuttgart"
export WANDB_MODE=online
echo "WandB configured (entity: jaseelkt1-university-of-stuttgart)"

echo "Starting training..."
START_TIME=$(date +%s)

CUDA_LAUNCH_BLOCKING=1 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --dynamo_backend no \
    tools/train_video_controlnet.py \
    --run_name $NAME \
    --data_root "" \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_NAME_OR_PATH \
    --output_dir $CHECKPOINT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --unet_learning_rate 1e-6 \
    --checkpoints_total_limit 1 \
    --checkpointing_steps 100 \
    --gradient_accumulation_steps 4 \
    --validation_steps 300 \
    --enable_gradient_checkpointing \
    --lr_scheduler constant \
    --report_to wandb \
    --seed 1234 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 1.0 \
    --max_guidance_scale 3.0 \
    --noise_aug_strength 0.01 \
    --bbox_dropout_prob 0.1 \
    --num_demo_samples 3 \
    --num_train_epochs 10 \
    --max_train_steps 32700 \
    --use_segmentation \
    --num_inference_steps 30 \
    --train_H 192 \
    --train_W 704 \
    --dataloader_num_workers 8 \
    --early_stop_patience 0 \
    --use_multiscale_injection \
    --resume_from_checkpoint latest \
    $( [ -n "$FINETUNED_SVD_PATH" ] && echo "--finetuned_svd_path $FINETUNED_SVD_PATH" )

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
    echo "Validation plots copied to: ${OUT_DIR}/plots/"
fi

echo ""
echo "========================================="
echo "Training Summary"
echo "========================================="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Run Name:         ${NAME}"
echo "Duration:         ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Changes vs baseline:"
echo "  1. Semantic VAE latents scaled by vae.config.scaling_factor"
echo "  2. UNet mid_block + up_blocks unfrozen at LR=1e-6"
echo "  3. ControlNet at LR=1e-5 (unchanged)"
echo ""
echo "Checkpoints:      ${CHECKPOINT_DIR}/"
echo "Outputs & Plots:  ${OUT_DIR}/"
echo "========================================="
