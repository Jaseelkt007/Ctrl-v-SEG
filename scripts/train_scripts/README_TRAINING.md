# KITTI360 Training Guide

## Quick Start

### Option 1: Submit Batch Job (Recommended)
```bash
cd /usrhomes/s1492/Ctrl-V
sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh
```

**Check job status:**
```bash
squeue -u $USER
```

**Monitor logs in real-time:**
```bash
tail -f /no_backups/s1492/Ctrl-V/logs/train_<JOB_ID>.out
```

### Option 2: Interactive Mode (For Testing)
```bash
# Allocate GPU node
salloc --gpus=1 --mem=32G --qos=interactive --nodelist=linse19

# Run script
cd /usrhomes/s1492/Ctrl-V
bash scripts/train_scripts/train_kitti360_bbox_predict.sh
```

---

## SBATCH Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--job-name` | `kitti360_bbox_train` | Job name in queue |
| `--gpus` | `1` | One A5000 GPU (24GB) |
| `--mem` | `32G` | System RAM |
| `--qos` | `batch` | No time limit |
| `--nodelist` | `linse19` | Specific node with A5000 |
| `--partition` | `studentbatch` | Student partition |

---

## Output Locations

### Checkpoints
```
/no_backups/s1492/Ctrl-V/checkpoints/kitti360_preprocessed_bbox_predict_<timestamp>/
├── checkpoint-200/
├── checkpoint-400/
├── checkpoint-600/
└── plots/
```

### Logs & Outputs
```
/no_backups/s1492/Ctrl-V/outputs/kitti360_preprocessed_bbox_predict_<timestamp>/
├── train_script.sh  # Backup of training script
└── plots/           # Copy of validation plots
```

### SLURM Logs
```
/no_backups/s1492/Ctrl-V/logs/
├── train_<JOB_ID>.out  # Standard output
└── train_<JOB_ID>.err  # Error output
```

---

## Training Configuration

### Current Settings (1 Epoch Test)
- **Dataset:** KITTI360 Preprocessed (5,993 clips)
- **Model:** Stable Video Diffusion (SVD-XT)
- **Resolution:** 320×512
- **Batch size:** 1 (effective: 5 with gradient accumulation)
- **Learning rate:** 5e-6
- **Epochs:** 1 (~1,199 steps)
- **Checkpoints:** Every 200 steps
- **Validation:** Every 100 steps

### For Full Training
Edit `train_kitti360_bbox_predict.sh` line 116:
```bash
--num_train_epochs 10 \  # Change from 1 to 10
```

---

## Monitor Training

### WandB Dashboard
```
https://wandb.ai/<your_username>/ctrl_v_kitti360
```

**First time setup:**
```bash
wandb login
# Paste API key from: https://wandb.ai/authorize
```

### Check Logs
```bash
# Real-time monitoring
tail -f /no_backups/s1492/Ctrl-V/logs/train_<JOB_ID>.out

# View errors
tail -f /no_backups/s1492/Ctrl-V/logs/train_<JOB_ID>.err

# Full log
less /no_backups/s1492/Ctrl-V/logs/train_<JOB_ID>.out
```

### Check Validation Images
```bash
# Latest run
ls -lh /no_backups/s1492/Ctrl-V/checkpoints/kitti360_*/plots/

# Specific step
ls /no_backups/s1492/Ctrl-V/checkpoints/kitti360_*/plots/step_100/
```

---

## Resume Training

Edit `train_kitti360_bbox_predict.sh` line 123:
```bash
--resume_from_checkpoint latest  # Uncomment this line
```

Then resubmit:
```bash
sbatch scripts/train_scripts/train_kitti360_bbox_predict.sh
```

---

## Troubleshooting

### Out of Memory (OOM)
Reduce memory usage:
```bash
# Line 97: Reduce batch size (already at minimum 1)
# Line 101: Reduce gradient accumulation
--gradient_accumulation_steps 2  # From 5 to 2

# Or reduce resolution (lines 120-121)
--train_H 256 \
--train_W 448 \
```

### Training Too Slow
Speed up:
```bash
# Line 122: Reduce data workers
--dataloader_num_workers 2  # From 4 to 2

# Line 102: Reduce validation frequency
--validation_steps 200  # From 100 to 200
```

### Job Killed/Failed
Check error log:
```bash
tail -100 /no_backups/s1492/Ctrl-V/logs/train_<JOB_ID>.err
```

Common issues:
- CUDA out of memory → Reduce batch size/resolution
- Dataset not found → Check `/no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv/`
- Import errors → Check conda environment: `conda activate kitti`

---

## File Structure After Training

```
/no_backups/s1492/Ctrl-V/
│
├── checkpoints/
│   └── kitti360_preprocessed_bbox_predict_251014_194500/
│       ├── checkpoint-1000/      # Last checkpoint before final
│       │   ├── unet/              # Model weights
│       │   ├── optimizer.bin      # Optimizer state
│       │   └── scheduler.bin      # LR scheduler
│       ├── checkpoint-1199/       # Final checkpoint
│       └── plots/                 # Validation images
│           ├── step_100/
│           ├── step_200/
│           └── ...
│
├── outputs/
│   └── kitti360_preprocessed_bbox_predict_251014_194500/
│       ├── train_script.sh        # Script backup
│       └── plots/                 # Validation copy
│
└── logs/
    ├── train_12345.out            # Training logs
    └── train_12345.err            # Error logs
```

---

## Useful Commands

### Check job queue
```bash
squeue -u $USER
```

### Cancel job
```bash
scancel <JOB_ID>
```

### Check GPU usage
```bash
nvidia-smi
```

### Disk usage
```bash
du -sh /no_backups/s1492/Ctrl-V/checkpoints/*
```

### List all runs
```bash
ls -lht /no_backups/s1492/Ctrl-V/checkpoints/
```

---

## Expected Timeline (1 Epoch)

| Time | Event |
|------|-------|
| 0-3 min | Model download & initialization |
| 3-5 min | Dataset loading |
| 5-10 min | First 100 steps + validation |
| 10-15 min | Checkpoint-200 saved |
| 60-120 min | Complete 1 epoch (~1,199 steps) |

---

## Next Steps After Test Run

1. ✅ Review WandB dashboard for loss curve
2. ✅ Check validation images in `plots/` directory
3. ✅ Verify checkpoints saved correctly
4. ✅ If successful, increase to 10 epochs
5. ✅ Monitor first few hours of full training

---

## Support

For issues, check:
- Training logs: `/no_backups/s1492/Ctrl-V/logs/`
- WandB dashboard: https://wandb.ai
- GPU status: `nvidia-smi`
- Disk space: `df -h /no_backups/s1492/`
