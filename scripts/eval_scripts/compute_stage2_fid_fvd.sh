#!/bin/bash
#SBATCH --job-name=stage2_fvd_freezeunet
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/stage2_fvd_freezeunet_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/stage2_fvd_freezeunet_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_a5000:1
#SBATCH --partition=stud
#SBATCH --qos=batch
#SBATCH --time=08:00:00

set -e
set -u

echo "========================================="
echo "Stage 2: FID / FVD / LPIPS / SSIM / PSNR"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-localhost}"
echo "Started at: $(date)"
echo ""

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

cd /usrhomes/s1492/Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# ============================================================================
# Configuration
# ============================================================================

FRAMES_DIR="/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb/frames"
OUTPUT_DIR="/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb"

echo ""
echo "Frames directory: $FRAMES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# Run
# ============================================================================

echo "Starting FID/FVD/LPIPS/SSIM/PSNR evaluation..."
START_TIME=$(date +%s)

python tools/compute_stage2_fid_fvd.py \
    --frames_dir "$FRAMES_DIR" \
    --num_frames 25 \
    --fid_batch_size 64 \
    --output_file "${OUTPUT_DIR}/fid_fvd_results.txt" \
    --output_json "${OUTPUT_DIR}/fid_fvd_results.json"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "Duration: $((DURATION / 60))m $((DURATION % 60))s"
echo ""

if [ -f "${OUTPUT_DIR}/fid_fvd_results.txt" ]; then
    echo "--- Results ---"
    cat "${OUTPUT_DIR}/fid_fvd_results.txt"
fi

echo ""
echo "✓ Complete!"
