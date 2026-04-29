#!/bin/bash
#SBATCH --job-name=stage2_fps
#SBATCH --output=/no_backups/s1492/Ctrl-V/logs/stage2_fps_%j.out
#SBATCH --error=/no_backups/s1492/Ctrl-V/logs/stage2_fps_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=rtx_a5000:1
#SBATCH --partition=stud
#SBATCH --qos=batch
#SBATCH --time=00:30:00

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kitti

cd /usrhomes/s1492/Ctrl-V-seg
export PYTHONPATH="/usrhomes/s1492/Ctrl-V-seg/src:${PYTHONPATH:-}"
mkdir -p /no_backups/s1492/Ctrl-V/logs

echo "Node: ${SLURM_NODELIST:-localhost}  |  GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python tools/time_stage2_fps.py
