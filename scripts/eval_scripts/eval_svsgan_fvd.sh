#!/bin/bash
#SBATCH --job-name=eval_svsgan_fvd
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/eval_svsgan_fvd_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/eval_svsgan_fvd_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=highperf
#SBATCH --time=12:00:00

set -e
set -u

echo "========================================="
echo "SVS-GAN Evaluation: FVD-I3D"
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

SVS_DIR="/data/public/kitti-360/Kitti360_512_g1_DP45_DPDisc/val_latest"
OUTPUT_DIR="/no_backups/s1492/Ctrl-V/outputs/eval_svsgan_fvd"

mkdir -p "$OUTPUT_DIR"
mkdir -p /no_backups/s1492/Ctrl-V/logs

echo ""
echo "Configuration:"
echo "  SVS-GAN output : $SVS_DIR"
echo "  Output         : $OUTPUT_DIR"
echo "  GT source      : real_B_*.png co-located with fake_B_*.jpg"
echo ""

START_TIME=$(date +%s)

python tools/eval_svsgan_fvd.py \
    --svs_dir     "$SVS_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --clip_length 25

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
echo "✓ SVS-GAN FVD evaluation complete!"
