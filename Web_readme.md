# run backend : 
sbatch backend/run_backend.sh

# check ip of backend compute node
squeue -j <job_id> -h -o "%N" → getent hosts <node> → update .env.local

# run frontend : 
cd frontend && npm run dev -- -p 3000 -H 0.0.0.0

# GPU configs 
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

