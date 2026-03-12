#!/bin/bash
#SBATCH --job-name=ctrlv_api
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/api_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/api_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=rtx_a5000:1
#SBATCH --partition=stud
#SBATCH --qos=batch
#SBATCH --time=08:00:00

set -e
set -u

echo "========================================="
echo "Ctrl-V-Seg Backend API Server"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Started at: $(date)"
echo ""

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti
echo "Conda environment 'kitti' activated"

cd /usrhomes/s1492/Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"

# Install API dependencies if needed
pip install fastapi uvicorn python-multipart websockets imageio 2>/dev/null || true

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Get node hostname for frontend connection
NODE=$(hostname)
PORT=${API_PORT:-8000}

echo "========================================="
echo "API Server starting on: http://${NODE}:${PORT}"
echo "========================================="
echo ""
echo "To connect the frontend, set the API URL to:"
echo "  http://${NODE}:${PORT}"
echo ""
echo "Or from the login node, use SSH tunnel:"
echo "  ssh -L ${PORT}:${NODE}:${PORT} $(whoami)@login-node"
echo ""

# Run the API server
cd /usrhomes/s1492/Ctrl-V-seg/backend
python -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1

echo ""
echo "API Server stopped at: $(date)"
