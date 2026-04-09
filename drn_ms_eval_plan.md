# Stage 2 Multi-Scale DRN mIoU Evaluation — Implementation Plan

**Goal:** Replicate the `drn_d_105_MIoU` evaluation methodology on our
Stage 2 ControlNet (Semantic → RGB) generated frames, enabling a fair apples-to-apples
comparison of DRN mIoU scores.

---

## Background

Our existing `eval_stage2_rgb.py` already computes DRN mIoU, but uses:
- **Single-scale** DRN inference (1 forward pass per frame)
- **Unweighted** confusion matrix (every valid pixel counts equally)
- **150 non-overlapping clips** from the official KITTI-360 val split

The `drn_d_105_MIoU` evaluation uses:
- **Multi-scale** DRN inference (6 scales: 0.5×, 0.75×, 1.0×, 1.25×, 1.5×, 1.75×)
- **Confidence-weighted** confusion matrix (KITTI-360 per-pixel confidence maps)
- **19 specific clip groups** × 30 frames = 570 frames (from their baseline model)

These three differences are expected to raise the reported mIoU by 1–5 points.
To compare our model against the baseline on equal footing, we need
to apply the same evaluation protocol to our generated frames.

---

## Repository Changes Made

### New directory: `drn_eval/`
```
drn_eval/
├── segment.py               ← enhanced version (from ~/drn/): handles 16-bit I;16
│                               confidence PNGs via cv2.resize, supports val_confidence.txt
├── drn.py                   ← DRN-D-105 architecture
├── data_transforms.py       ← image transforms for DRN
├── lib/                     ← required C extension modules
├── KITTI360_checkpoints/
│   ├── checkpoint_030.pth.tar → symlink → ~/drn/KITTI360_checkpoints/checkpoint_030.pth.tar
│   └── checkpoint_029.pth.tar → symlink
└── CTRLV_STAGE2/
    └── info.json            ← KITTI-360 normalisation stats: mean/std
                                (same as ~/drn/CTRLV_BBOX/info.json)
```

**Key property of the enhanced `segment.py`:**
- `SegList` and `SegListMS` auto-derive confidence map paths from `val_labels.txt` by
  replacing `"semantic"` → `"confidence"` in every label path.
- If the first confidence file exists on disk, `confidence_list` is populated and
  `fast_hist_weighted()` is used; otherwise falls back to `fast_hist()`.

### New script: `tools/generate_stage2_frames_for_drn.py`
Phase 1 of the pipeline. See §Phase 1 below.

### New script: `scripts/eval_scripts/eval_stage2_drn_ms.sh`
Orchestrates both phases via SLURM. See §Running the Evaluation below.

---

## Data Paths Used

| Resource | Path |
|---|---|
| Public KITTI-360 root | `/misc/data/public/kitti-360/KITTI-360/` |
| GT semantic PNGs | `…/data_2d_semantics/train/{seq}/image_00/semantic/{frame:010d}.png` |
| GT confidence PNGs | `…/data_2d_confidences/train/{seq}/image_00/confidence/{frame:010d}.png` |
| RGB raw frames | `…/data_2d_raw/{seq}/image_00/data_rect/{frame:010d}.png` |
| reference | `/data/public/kitti-360/drn_d_105_MIoU/CTRLV_BBOX/val_labels.txt` |
| DRN checkpoint | `drn_eval/KITTI360_checkpoints/checkpoint_030.pth.tar` |
| DRN info.json | `drn_eval/CTRLV_STAGE2/info.json` |

---

## The 19 Clip Groups

Parsed from the `val_labels.txt`. Each is a run of 30 consecutive frames.
We use the **first 25 frames** of each group (matching our `clip_length=25`).

| # | Sequence | Frame IDs | Notes |
|---|---|---|---|
| 00 | 2013_05_28_drive_0000_sync | 0447–0471 | |
| 01 | 2013_05_28_drive_0000_sync | 2768–2792 | |
| 02 | 2013_05_28_drive_0000_sync | 3295–3319 | |
| 03 | 2013_05_28_drive_0002_sync | 4452–4476 | |
| 04 | 2013_05_28_drive_0002_sync | 15261–15285 | |
| 05 | 2013_05_28_drive_0003_sync | 0068–0092 | |
| 06 | 2013_05_28_drive_0004_sync | 2975–2999 | |
| 07 | 2013_05_28_drive_0004_sync | 4442–4466 | |
| 08 | 2013_05_28_drive_0005_sync | 4849–4873 | |
| 09 | 2013_05_28_drive_0006_sync | 0158–0182 | |
| 10 | 2013_05_28_drive_0006_sync | 2351–2375 | |
| 11 | 2013_05_28_drive_0006_sync | 9285–9309 | |
| 12 | 2013_05_28_drive_0007_sync | 0063–0087 | |
| 13 | 2013_05_28_drive_0009_sync | 0849–0873 | |
| 14 | 2013_05_28_drive_0009_sync | 4551–4575 | |
| 15 | 2013_05_28_drive_0009_sync | 5241–5265 | |
| 16 | 2013_05_28_drive_0009_sync | 6347–6371 | |
| 17 | 2013_05_28_drive_0010_sync | 1941–1965 | |
| 18 | 2013_05_28_drive_0010_sync | 2668–2692 | |

**Total: 19 groups × 25 frames = 475 generated RGB frames evaluated**
(The evaluates 19 × 30 = 570 frames; we use 19 × 25 = 475 due to `clip_length=25`.)

---

## Phase 1: Frame Generation (`generate_stage2_frames_for_drn.py`)

### What it does
1. Parses the 19 clip groups from the `val_labels.txt`
2. For each group, loads:
   - **image_init**: first raw RGB frame from KITTI-360 (for SVD image conditioning)
   - **semantic_ids**: 25 GT semantic maps → remapped to trainIDs (0–18, 255=ignore)
   - **sem_rgb**: semantic RGB visualization tensor (for `cond_images` argument)
3. Runs the Stage 2 `StableVideoControlPipeline` with `use_semantic_vae=True`
4. Saves 25 generated RGB PNGs per group to:
   ```
   {output_dir}/CTRLV_STAGE2/generated_frames/{seq}_{group_idx:02d}/frame_{t:04d}.png
   ```
5. Writes:
   - `val_images.txt` — relative paths to generated PNGs (from `CTRLV_STAGE2/`)
   - `val_labels.txt` — absolute paths to GT semantic PNGs (`data_2d_semantics/…`)
   - `metadata.json` — clip metadata (checkpoint step, frame IDs, etc.)

### Confidence map handling
`segment.py` auto-derives confidence paths:
```python
potential_conf_list = [line.replace("semantic", "confidence") for line in self.label_list]
```
This transforms each `val_labels.txt` entry from:
```
/misc/data/public/kitti-360/KITTI-360/data_2d_semantics/train/{seq}/image_00/semantic/{f}.png
```
to:
```
/misc/data/public/kitti-360/KITTI-360/data_2d_confidences/train/{seq}/image_00/confidence/{f}.png
```
Since these files exist on the cluster, confidence weighting **will be active**.

### Label format compatibility
KITTI-360 semantic PNGs store **raw label IDs (0–44)**. `segment.py`'s `id2label_kitti()`
converts them to trainIDs (0–18, 255) — the same mapping used by `KITTI360_LABEL_MAPPING`
in our training pipeline. No additional conversion is needed in the generation script.

### Output directory layout
```
{output_dir}/
├── CTRLV_STAGE2/
│   ├── info.json                  ← copied from drn_eval/CTRLV_STAGE2/
│   ├── val_images.txt             ← relative paths to generated frames
│   ├── val_labels.txt             ← absolute paths to GT semantic PNGs
│   └── generated_frames/
│       ├── 2013_05_28_drive_0000_sync_00/
│       │   ├── frame_0001.png … frame_0025.png
│       ...
│       └── 2013_05_28_drive_0010_sync_18/
│           └── frame_0001.png … frame_0025.png
└── metadata.json
```

---

## Phase 2: Multi-Scale DRN Evaluation (`segment.py test --ms`)

### Command executed
```bash
cd drn_eval/
python segment.py test \
    -d  {output_dir}/CTRLV_STAGE2 \
    -c  19 \
    --arch     drn_d_105 \
    --pretrained KITTI360_checkpoints/checkpoint_030.pth.tar \
    --phase    val \
    --batch-size 1 \
    --ms
```

### What `segment.py test --ms` does step-by-step

**Step 1 — Load txt manifests:**
```
{data_dir}/val_images.txt  →  SegListMS.image_list
{data_dir}/val_labels.txt  →  SegListMS.label_list
derived from labels         →  SegListMS.confidence_list  (if files exist)
```

**Step 2 — Per-sample data loading (`SegListMS.__getitem__`):**
```
data[0]    : original generated RGB image (at native resolution)
data[1]    : GT semantic label   (id2label_kitti applied → trainIDs 0-18)
data[2]    : confidence map      (16-bit I;16 PNG, loaded via cv2, normalised)
data[3]    : image name (string)
data[4..8] : same RGB image resized to 5 scales [0.5, 0.75, 1.25, 1.50, 1.75]
```

**Step 3 — Multi-scale inference (`test_ms()`):**
```
For each scale s in [0.5, 0.75, 1.0, 1.25, 1.50, 1.75]:
    forward_pass(image_at_scale_s) → logit_map [1, 19, H_s, W_s]  (log_softmax)

bilinear_resize(all 6 logit maps) → [1, 19, H, W]
sum 6 logit maps
pred = argmax over 19 classes  →  [H, W] trainID prediction
```

**Step 4 — Confidence-weighted confusion matrix update:**
```python
confidence_np = confidence[0].numpy() / 255.0
hist += fast_hist_weighted(
    pred.flatten(), label.flatten(), confidence_np.flatten(), 19
)
```
The confidence PNG is 16-bit (uint16). `ToTensor16Bit` normalises as `/ 65535 + 0.5`,
then `test_ms()` divides again by `/ 255.0`. This matches the behaviour.

**Step 5 — Final mIoU:**
```python
ious  = per_class_iu(hist) * 100   # per-class IoU (%)
mIoU  = round(np.nanmean(ious), 2)  # mean over non-NaN classes
```
Printed as `mAP: XX.XX` at the end of the log.

---

## Running the Evaluation

### Full SLURM job
```bash
sbatch scripts/eval_scripts/eval_stage2_drn_ms.sh
```

### Manual (interactive)
```bash
cd /usrhomes/s1492/Ctrl-V-seg
conda activate kitti
export PYTHONPATH="src:${PYTHONPATH:-}"

# Phase 1 — generate frames
python tools/generate_stage2_frames_for_drn.py \
    --checkpoint_dir /no_backups/s1492/Ctrl-V/checkpoints/kitti360_sem2video_unet_unfreeze_reinject \
    --output_dir     /no_backups/s1492/Ctrl-V/outputs/eval_stage2_drn_ms_test

# Phase 2 — multi-scale DRN
cd drn_eval/
python segment.py test \
    -d  /no_backups/s1492/Ctrl-V/outputs/eval_stage2_drn_ms_test/CTRLV_STAGE2 \
    -c 19 --arch drn_d_105 \
    --pretrained KITTI360_checkpoints/checkpoint_030.pth.tar \
    --phase val --batch-size 1 --ms
```

---

## Changing the Checkpoint

In `eval_stage2_drn_ms.sh`, edit only:
```bash
CHECKPOINT_DIR="..."   # path to Stage 2 checkpoint
OUTPUT_DIR="..."       # change suffix to avoid overwriting previous results
```

---

## Expected Outputs

After the job completes:
```
{OUTPUT_DIR}/
├── CTRLV_STAGE2/
│   ├── info.json
│   ├── val_images.txt         (475 entries: 19 groups × 25 frames)
│   ├── val_labels.txt         (475 entries: absolute GT paths)
│   └── generated_frames/      (19 subdirectories, 25 PNGs each)
├── metadata.json              (checkpoint info, clip metadata)
└── drn_ms_eval.log            (segment.py stdout: per-image mAP + final mIoU)

drn_eval/drn_d_105_000_val_ms/   (created by segment.py, colourised prediction images)
```

The **final mIoU** is printed as `mAP: XX.XX` at the end of `drn_ms_eval.log`.

---

## Key Differences vs `eval_stage2_rgb.sh`

| Aspect | `eval_stage2_rgb.py` | This pipeline |
|---|---|---|
| DRN inference | Single-scale (1×) | Multi-scale (6 scales) |
| Confusion matrix | Unweighted | Confidence-weighted |
| Frame count | 150 clips × 25 = 3,750 | 19 clips × 25 = 475 |
| Clip selection | Official val split | Same clips  |
| Comparability | Standard academic | Directly comparable to baseline |
| Additional metrics | FID, FVD, LPIPS, SSIM, PSNR | DRN mIoU only |

Both use: `drn_d_105`, `checkpoint_030.pth.tar`, same normalisation stats, same GT
source, same mIoU formula `nanmean(TP / (TP + FP + FN))`.

---

## Troubleshooting

**Phase 1 crashes mid-way (CUDA OOM / pipeline error)**
The txt files are written only at the end of Phase 1. Re-run after fixing the issue.
If OOM: reduce `--num_inference_steps 20` or request A6000 GPU (48 GB).

**"val_images.txt not found" after Phase 1**
Check `drn_ms_eval.log` for the error. Phase 1 must complete successfully.

**"Confidence maps not found" — mIoU silently uses unweighted fallback**
`segment.py` checks `exists(join(data_dir, potential_conf_list[0]))`.
Since confidence paths are absolute, this resolves to the full path.
Verify: `ls /misc/data/public/kitti-360/KITTI-360/data_2d_confidences/train/`
If the mount is unavailable on the compute node, confidence weighting is disabled.

**`ModuleNotFoundError: lib.dense.batch_norm`**
The `lib/` C extensions may need recompiling for the node's CUDA version.
`cd drn_eval/lib && make` (requires `nvcc`).

**segment.py log directory conflict**
`segment.py` creates a subdirectory `drn_d_105_000_val_ms/` inside `drn_eval/`.
If this already exists from a previous run, it is overwritten harmlessly.
