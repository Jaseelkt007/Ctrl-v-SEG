#!/bin/bash
#SBATCH --job-name=test_train_short
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/test_train_short_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/test_train_short_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G 
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=01:00:00

set -e

echo "================================================================================"
echo "SHORT TRAINING TEST - Semantic VAE Integration"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started at: $(date)"
echo ""
echo "This test will:"
echo "  1. Run training for ~50 steps (not full training)"
echo "  2. Verify semantic VAE is being used"
echo "  3. Check WandB logging shows 'gt_semantic_frames'"
echo "  4. Confirm grayscale semantic IDs are encoded"
echo "  5. Save checkpoint at step 50 for inspection"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

cd /usrhomes/s1492/Ctrl-V-seg
echo "✓ Working directory: $(pwd)"

# Force Python to import ctrlv from Ctrl-V-seg (with fixes), not old Ctrl-V
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"
echo "✓ PYTHONPATH set to prioritize Ctrl-V-seg"

# Verify which ctrlv will be imported
python -c "import ctrlv, os; print('✓ ctrlv will be imported from:', os.path.dirname(ctrlv.__file__))"
echo ""

# ============================================================================
# Training Configuration (SHORT TEST)
# ============================================================================

DATASET="kitti360"
# DATASET_PATH no longer needed - KITTI360OfficialDataset uses official paths internally
NAME="test_semantic_vae_short"
CHECKPOINT_DIR="/no_backups/s1492/Ctrl-V/test_checkpoints/test_semantic_short"
OUT_DIR="/no_backups/s1492/Ctrl-V/test_outputs/test_semantic_short"
PROJECT_NAME='ctrl_v_kitti360'

mkdir -p $CHECKPOINT_DIR
mkdir -p $OUT_DIR

echo "Checkpoints: $CHECKPOINT_DIR"
echo "Outputs: $OUT_DIR"
echo "Training for: 50 steps (1 validation at step 50)"
echo ""

# WandB Configuration
export WANDB_ENTITY="jaseelkt1-university-of-stuttgart"
export WANDB_MODE=online
echo "✓ WandB configured"

# ============================================================================
# Run Short Training
# ============================================================================

echo ""
echo "Starting short training test..."
echo ""

CUDA_LAUNCH_BLOCKING=1 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --dynamo_backend no \
    tools/train_video_diffusion.py \
    --run_name $NAME \
    --data_root "" \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --output_dir $CHECKPOINT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --train_batch_size 1 \
    --learning_rate 5e-6 \
    --checkpoints_total_limit 1 \
    --checkpointing_steps 50 \
    --gradient_accumulation_steps 6 \
    --validation_steps 50 \
    --max_train_steps 50 \
    --enable_gradient_checkpointing \
    --lr_scheduler constant \
    --report_to wandb \
    --seed 1234 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 3 \
    --max_guidance_scale 7 \
    --noise_aug_strength 0.01 \
    --bbox_dropout_prob 0.1 \
    --conditioning_dropout_prob 0.0 \
    --num_demo_samples 2 \
    --backprop_temporal_blocks_start_iter -1 \
    --num_train_epochs 1 \
    --predict_bbox \
    --use_segmentation \
    --num_inference_steps 30 \
    --num_cond_bbox_frames 1 \
    --train_H 192 \
    --train_W 704 \
    --dataloader_num_workers 4

# ============================================================================
# Verify Results
# ============================================================================

echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"

# Check if checkpoint was saved
if [ -d "$CHECKPOINT_DIR/checkpoint-50" ]; then
    echo "✅ Checkpoint saved at step 50"
    ls -lh $CHECKPOINT_DIR/checkpoint-50/
else
    echo "❌ No checkpoint found at step 50"
fi

# Check logs for semantic VAE usage
echo ""
echo "Checking logs for semantic VAE usage..."
if grep -q "Semantic VAE" /no_backups/s1492/Ctrl-V/logs/test_train_short_${SLURM_JOB_ID}.out 2>/dev/null || \
   grep -q "semantic_ids" /no_backups/s1492/Ctrl-V/logs/test_train_short_${SLURM_JOB_ID}.out 2>/dev/null; then
    echo "✅ Logs mention semantic VAE usage"
else
    echo "⚠️  Could not confirm semantic VAE usage from logs"
fi

# Check WandB logs
echo ""
echo "================================================================================"
echo "WandB Verification"
echo "================================================================================"
echo ""
echo "Please check WandB dashboard:"
echo "  URL: https://wandb.ai/jaseelkt1-university-of-stuttgart/ctrl_v_kitti360/runs/$NAME"
echo ""
echo "Verify:"
echo "  ✓ Log name shows 'gt_semantic_frames' (NOT 'gt_bbox_frames')"
echo "  ✓ Validation images show semantic predictions"
echo "  ✓ Training loss is logged"
echo "  ✓ No errors in training"
echo ""

echo "================================================================================"
echo "TEST COMPLETE"
echo "================================================================================"
echo "Completed at: $(date)"
echo ""
echo "Next steps:"
echo "  1. Check WandB logs for 'gt_semantic_frames'"
echo "  2. Verify checkpoint was saved correctly"
echo "  3. If all looks good, proceed with full training"
echo ""
echo "To start full training:"
echo "  sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh"
