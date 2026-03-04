#!/bin/bash
#SBATCH --job-name=verify_semantic_vae
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/verify_semantic_vae_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/verify_semantic_vae_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=00:30:00

# Quick verification run: confirm semantic VAE is being used correctly
# Runs only 5 steps, no checkpoint saving, no WandB

set -e
set -u

echo "========================================="
echo "VERIFICATION RUN: Semantic VAE Usage Check"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""

# Clean CUDA paths
export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'cuda-10' | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'cuda-10' | tr '\n' ':' | sed 's/:$//')

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti

cd /usrhomes/s1492/Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"

python -c "import ctrlv, os; print('ctrlv from:', os.path.dirname(ctrlv.__file__))"

# Minimal accelerate config
export ACCELERATE_USE_FSDP=0
export ACCELERATE_MIXED_PRECISION=fp16

# Disable WandB for verification
export WANDB_MODE=disabled

VERIFY_DIR="/no_backups/s1492/Ctrl-V/verify_semantic_vae"
mkdir -p $VERIFY_DIR

echo "Starting verification (max_train_steps=5, no checkpoints)..."
echo ""

CUDA_LAUNCH_BLOCKING=1 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --dynamo_backend no \
    tools/train_video_diffusion.py \
    --run_name verify_semantic_vae \
    --data_root "" \
    --project_name verify \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --output_dir $VERIFY_DIR \
    --variant fp16 \
    --dataset_name kitti360 \
    --train_batch_size 1 \
    --learning_rate 5e-6 \
    --checkpointing_steps 99999 \
    --gradient_accumulation_steps 1 \
    --validation_steps 99999 \
    --enable_gradient_checkpointing \
    --lr_scheduler constant \
    --report_to tensorboard \
    --seed 1234 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 3 \
    --max_guidance_scale 7 \
    --noise_aug_strength 0.01 \
    --bbox_dropout_prob 0.1 \
    --conditioning_dropout_prob 0.0 \
    --num_demo_samples 0 \
    --backprop_temporal_blocks_start_iter -1 \
    --max_train_steps 5 \
    --predict_bbox \
    --use_segmentation \
    --num_inference_steps 30 \
    --num_cond_bbox_frames 1 \
    --train_H 192 \
    --train_W 704 \
    --dataloader_num_workers 4

echo ""
echo "========================================="
echo "VERIFICATION COMPLETE"
echo "========================================="
echo "Check the output above for:"
echo "  1. 'Using Semantic VAE encoding for target latents' log message"
echo "  2. No assertion errors about missing semantic_ids"
echo "  3. Training ran 5 steps without errors"
echo ""
echo "If all checks pass, proceed with full training."
