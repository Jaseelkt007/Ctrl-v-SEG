#!/bin/bash
#SBATCH --job-name=eval_ctrlv_fid
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/eval_ctrlv_fid_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/eval_ctrlv_fid_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=12:00:00

set -e
set -u

echo "========================================="
echo "Ctrl-V (Original) Evaluation: FID"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node:   ${SLURM_NODELIST:-localhost}"
echo "Start:  $(date)"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "✓ Conda environment 'kitti' activated"

cd /usrhomes/s1492/Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"

echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR="/misc/data/public/kitti-360/CTRL_V_Semantic_to_video"
OUTPUT_DIR="/no_backups/s1492/Ctrl-V/outputs/eval_ctrlv_fid"

mkdir -p "$OUTPUT_DIR"
mkdir -p /no_backups/s1492/Ctrl-V/logs

echo ""
echo "Configuration:"
echo "  Data dir  : $DATA_DIR"
echo "  Output    : $OUTPUT_DIR"
echo "  GT frames : real_B_*.png co-located with frame_*.png"
echo ""

START_TIME=$(date +%s)

python tools/eval_ctrlv_fid.py \
    --data_dir    "$DATA_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --batch_size  64

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECS=$((DURATION % 60))

echo ""
echo "========================================="
echo "Job finished"
echo "  Duration : ${HOURS}h ${MINUTES}m ${SECS}s"
echo "  Outputs  :"
echo "    JSON    : ${OUTPUT_DIR}/eval_results.json"
echo "    Summary : ${OUTPUT_DIR}/eval_summary.txt"
echo "========================================="

if [ -f "${OUTPUT_DIR}/eval_summary.txt" ]; then
    echo ""
    echo "--- eval_summary.txt ---"
    cat "${OUTPUT_DIR}/eval_summary.txt"
fi

echo ""
echo "✓ Ctrl-V FID evaluation complete!"
