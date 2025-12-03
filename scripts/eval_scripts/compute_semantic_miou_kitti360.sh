#!/bin/bash
#SBATCH --job-name=semantic_miou_k360
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/semantic_miou_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/semantic_miou_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=24:00:00

# Compute Semantic Segmentation mIoU for KITTI-360 generated videos
# This evaluates the quality of generated frames against official KITTI-360 semantic labels

set -e

echo "========================================="
echo "Semantic Segmentation mIoU Evaluation"
echo "========================================="

# Generated frames directory (from semantic pipeline evaluation)
GENERATED_FRAMES_DIR="/no_backups/s1492/Ctrl-V/outputs/kitti360_sem_overall_eval/wandb/run-20251108_192456-nhl86lzj/files/media/images/"

# Ground truth semantic segmentation from official KITTI-360
GT_SEMANTIC_DIR="/data/public/kitti-360/KITTI-360/data_2d_semantics/"
VAL_SPLIT_FILE="/data/public/kitti-360/KITTI-360/val_images.txt"

# Check if directories exist
if [ ! -d "$GENERATED_FRAMES_DIR" ]; then
    echo "Error: Generated frames directory not found: $GENERATED_FRAMES_DIR"
    exit 1
fi

if [ ! -d "$GT_SEMANTIC_DIR" ]; then
    echo "Error: GT semantic directory not found: $GT_SEMANTIC_DIR"
    exit 1
fi

if [ ! -f "$VAL_SPLIT_FILE" ]; then
    echo "Error: Validation split file not found: $VAL_SPLIT_FILE"
    exit 1
fi

echo "Generated frames: $GENERATED_FRAMES_DIR"
echo "GT semantic dir:  $GT_SEMANTIC_DIR"
echo "Val split file:   $VAL_SPLIT_FILE"
echo ""

# Count generated frames
FRAME_COUNT=$(ls "$GENERATED_FRAMES_DIR" | grep "^frames_with_" | wc -l)
echo "Generated frames found: $FRAME_COUNT"
echo ""

if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "Error: No generated frames found!"
    echo "Expected files: frames_with_*.png"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment 'kitti'..."
eval "$(conda shell.bash hook)"
conda activate kitti
echo "✓ Conda environment 'kitti' activated"
echo ""

# Check for DRN repository
DRN_PATH="/usrhomes/s1492/drn"
echo "Checking for DRN repository..."
if [ ! -d "$DRN_PATH" ]; then
    echo "Error: DRN repository not found at: $DRN_PATH"
    echo "Please clone it with: git clone https://github.com/fyu/drn.git"
    echo "To your home directory: /usrhomes/s1492/"
    exit 1
fi
echo "✓ DRN repository found at: $DRN_PATH"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python -c "import torch, torchvision, PIL, cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    pip install torch torchvision pillow opencv-python
    echo "✓ Dependencies installed"
else
    echo "✓ All dependencies available"
fi
echo ""

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "Error: CUDA not available. This evaluation requires GPU for segmentation model."
    exit 1
fi

echo "✓ GPU available"
echo ""

echo "========================================="
echo "Running semantic segmentation evaluation..."
echo "Model: DRN-D-105 (pretrained on Cityscapes)"
echo "DRN path: $DRN_PATH"
echo "This will take 10-30 minutes depending on number of frames..."
echo "========================================="
echo ""

# Change to project root directory
cd /usrhomes/s1492/Ctrl-V

# Run evaluation
python tools/compute_semantic_miou.py \
    --generated_frames_dir "$GENERATED_FRAMES_DIR" \
    --gt_semantic_dir "$GT_SEMANTIC_DIR" \
    --val_split_file "$VAL_SPLIT_FILE" \
    --drn_path "$DRN_PATH" \
    --model_name drn_d_105 \
    --device cuda:0 \
    --save_predictions

echo ""
echo "========================================="
echo "Semantic Segmentation mIoU Evaluation Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  ${GENERATED_FRAMES_DIR}/semantic_miou_results.txt"
echo ""
echo "Predicted segmentation maps saved to:"
echo "  ${GENERATED_FRAMES_DIR}/predicted_segmentation/"
echo ""
echo "Interpretation:"
echo "  - mIoU > 0.70:  Excellent segmentation quality"
echo "  - mIoU 0.60-0.70: Good quality"
echo "  - mIoU 0.50-0.60: Moderate quality"
echo "  - mIoU 0.40-0.50: Fair quality"
echo "  - mIoU < 0.40:  Poor quality"
echo ""
echo "========================================="