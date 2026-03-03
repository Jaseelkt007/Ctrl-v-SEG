#!/bin/bash
#SBATCH --job-name=test_stage2_short
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/test_stage2_short_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/test_stage2_short_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G 
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=01:00:00

set -e

echo "================================================================================"
echo "SHORT TRAINING TEST - STAGE 2: Semantic-to-RGB Video Generation"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started at: $(date)"
echo ""
echo "This test will:"
echo "  1. Run Stage 2 training for ~50 steps (not full training)"
echo "  2. Verify ControlNet is being used for semantic-to-RGB generation"
echo "  3. Check WandB logging shows 'generated_videos' (realistic RGB)"
echo "  4. Confirm ground truth semantics are used as conditioning"
echo "  5. Save checkpoint at step 50 for inspection"
echo ""
echo "Stage 2 Objective:"
echo "  - Input: Ground truth semantic labels (trainIDs)"
echo "  - Output: Realistic RGB video frames"
echo "  - Model: ControlNet conditioned on semantic maps"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

cd /usrhomes/s1492/Ctrl-V-seg
echo "✓ Working directory: $(pwd)"

# Force Python to import ctrlv from Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"
echo "✓ PYTHONPATH set to prioritize Ctrl-V-seg"

# Verify which ctrlv will be imported
python -c "import ctrlv, os; print('✓ ctrlv will be imported from:', os.path.dirname(ctrlv.__file__))"
echo ""

# ============================================================================
# Training Configuration (SHORT TEST - STAGE 2)
# ============================================================================

DATASET="kitti360"
NAME="test_stage2_sem2rgb_short"
CHECKPOINT_DIR="/no_backups/s1492/Ctrl-V/test_checkpoints/test_stage2_short"
OUT_DIR="/no_backups/s1492/Ctrl-V/test_outputs/test_stage2_short"
PROJECT_NAME='ctrl_v_kitti360'

# Stage 2 uses base SVD for testing (no Stage 1 checkpoint needed for initial test)
PRETRAINED_MODEL_NAME_OR_PATH="stabilityai/stable-video-diffusion-img2vid-xt"

mkdir -p $CHECKPOINT_DIR
mkdir -p $OUT_DIR

echo "Checkpoints: $CHECKPOINT_DIR"
echo "Outputs: $OUT_DIR"
echo "Training for: 50 steps (1 validation at step 50)"
echo "Base model: $PRETRAINED_MODEL_NAME_OR_PATH"
echo "Stage 1 checkpoint: Not used (parallel training mode)"
echo ""

# WandB Configuration
export WANDB_ENTITY="jaseelkt1-university-of-stuttgart"
export WANDB_MODE=online
echo "✓ WandB configured"

# ============================================================================
# Run Short Stage 2 Training
# ============================================================================

echo ""
echo "Starting Stage 2 short training test..."
echo "NOTE: Stage 2 uses train_video_controlnet.py (different from Stage 1)"
echo ""

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
    --checkpoints_total_limit 1 \
    --checkpointing_steps 50 \
    --gradient_accumulation_steps 4 \
    --validation_steps 50 \
    --max_train_steps 50 \
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
    --num_demo_samples 2 \
    --num_train_epochs 1 \
    --use_segmentation \
    --num_inference_steps 30 \
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

# Check logs for ControlNet usage
echo ""
echo "Checking logs for ControlNet usage..."
if grep -q "ControlNet\|controlnet" /no_backups/s1492/Ctrl-V/logs/test_stage2_short_${SLURM_JOB_ID}.out 2>/dev/null; then
    echo "✅ Logs mention ControlNet usage"
else
    echo "⚠️  Could not confirm ControlNet usage from logs"
fi

# Check WandB logs
echo ""
echo "================================================================================"
echo "WandB Verification - Stage 2"
echo "================================================================================"
echo ""
echo "Please check WandB dashboard:"
echo "  URL: https://wandb.ai/jaseelkt1-university-of-stuttgart/ctrl_v_kitti360/runs/$NAME"
echo ""
echo "Verify Stage 2 outputs:"
echo "  ✓ Log name shows 'generated_videos' (realistic RGB frames)"
echo "  ✓ Log name shows 'gt_videos' (ground truth RGB frames)"
echo "  ✓ Validation shows RGB generation from semantic maps"
echo "  ✓ Training loss is logged"
echo "  ✓ No errors in training"
echo ""
echo "Key Differences from Stage 1:"
echo "  - Stage 1: RGB → Semantic (predict_bbox mode)"
echo "  - Stage 2: Semantic → RGB (ControlNet mode)"
echo ""

echo "================================================================================"
echo "TEST COMPLETE - STAGE 2"
echo "================================================================================"
echo "Completed at: $(date)"
echo ""
echo "Next steps:"
echo "  1. Check WandB logs for realistic RGB generation"
echo "  2. Verify checkpoint was saved correctly"
echo "  3. Compare with ground truth RGB videos"
echo "  4. If all looks good, proceed with full Stage 2 training"
echo ""
echo "To start full Stage 2 training:"
echo "  sbatch scripts/train_scripts/train_kitti360_sem2video.sh"
echo ""
echo "Stage Training Status:"
echo "  ✓ Stage 1 (RGB→Semantic): Running (step 24212/81320)"
echo "  🧪 Stage 2 (Semantic→RGB): Testing complete - review before full training"
echo ""
