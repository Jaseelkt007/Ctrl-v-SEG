# Multi-Scale Semantic Re-Injection for ControlNet: Design Document

---

## Research Statement

Standard ControlNet introduces semantic conditioning at a single point — the encoder input — and relies on the encoder's own downsampling pipeline to propagate that signal to deeper feature levels. We observe that semantic information degrades substantially through successive stride-2 downsampling stages: objects that occupy 5–50 pixels in the latent space lose their discriminative boundaries by the time the encoder reaches its deepest representations, producing residuals that are geometrically diffuse rather than semantically precise.

To address this, we propose a **multi-scale semantic re-injection mechanism** that adds fresh semantic conditioning to the ControlNet encoder's intermediate representations at each spatial scale, after each down block completes its computation. Crucially, the injections are zero-initialized and additive, preserving the pretrained block computations entirely. The encoder blocks are never disturbed; only the inter-block communication is enriched with semantics. This approach follows the ControlNet design philosophy of additive, zero-initialized residuals, extended from the output level (UNet skip connections) to the encoder's internal feature propagation.

---

## 1. Vanilla ControlNet: Current Implementation

### 1.1 Architecture Overview

The `ControlNetModel` in this codebase is a structural replica of the UNet encoder half — it mirrors `conv_in`, all four `down_blocks`, and `mid_block`. The core idea from the original ControlNet paper is:

1. Create a trainable copy of the frozen UNet encoder
2. Inject a conditioning signal into it once at the input
3. Feed the resulting intermediate features back into the UNet decoder as additive residuals via zero-initialized projectors

### 1.2 Semantic Signal Entry: Single Point

The semantic conditioning enters the ControlNet at **exactly one location** — the very first layer, before any block processes the signal:

```python
# controlnet.py  forward(), lines 297–299
sample     = self.conv_in(sample)           # noisy latents  [B*F, 8, H, W] → [B*F, 320, H, W]
control_cond = self.control_conv_in(control_cond)  # semantic       [B*F, 4, H, W] → [B*F, 320, H, W]
sample     = sample + control_cond          # ← ONLY injection point
```

After this single addition, the loop over down blocks runs with **no further semantic input**:

```python
# controlnet.py  forward(), lines 303–319
down_block_res_samples = (sample,)
for downsample_block in self.down_blocks:
    sample, res_samples = downsample_block(
        hidden_states=sample,
        temb=emb,
        encoder_hidden_states=encoder_hidden_states,
        image_only_indicator=image_only_indicator,
    )
    down_block_res_samples += res_samples
```

### 1.3 What the Blocks Actually Receive

At our training resolution (192×704, latent 24×88), the spatial dimensions of `sample` entering each block are:

| Stage | Channels | Spatial Size | Semantic info quality |
|---|---|---|---|
| After `conv_in` + injection | 320 | 24 × 88 | **Strong** — fresh from `control_conv_in` |
| After `down_block[0]` | 320 | 12 × 44 | Diluted through stride-2 + 2 ResNet layers |
| After `down_block[1]` | 640 | 6 × 22 | Further diluted |
| After `down_block[2]` | 1280 | 3 × 11 | Severely diluted — 8× total downsampling |
| After `down_block[3]` | 1280 | 3 × 11 | Near-semantic-free |
| After `mid_block` | 1280 | 3 × 11 | Weakest semantic content |

### 1.4 What the UNet Receives

After the blocks run, 13 feature tensors pass through zero-initialized 1×1 convolutions and are added to the UNet's skip connections and mid-block:

```python
# controlnet.py  forward(), lines 331–344
controlnet_down_block_res_samples = ()
for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
    down_block_res_sample = controlnet_block(down_block_res_sample)   # zero-init conv
    controlnet_down_block_res_samples += (down_block_res_sample,)

mid_block_res_sample = self.controlnet_mid_block(sample)              # zero-init conv

# Scale and return — UNet adds these to its own skip connections
down_block_res_samples = [s * conditioning_scale for s in down_block_res_samples]
mid_block_res_sample   = mid_block_res_sample * conditioning_scale
```

**The output interface is multi-scale** (13 residuals spanning full to quarter resolution), but the **semantic content within those residuals** is single-scale — only the first few residuals carry meaningful semantic signal. The deeper residuals (from `down_block[2]`, `down_block[3]`, `mid_block`) are geometrically diffuse.

### 1.5 Observed Failure Pattern

This architectural bottleneck directly explains the per-class IoU pattern from evaluation (checkpoint-22700, 20 samples):

| Class tier | IoU range | Reason |
|---|---|---|
| road, car, vegetation, sky | 80–91% | Large spatial footprint — survives 8× latent downsampling |
| sidewalk, pole, fence, sign | 30–55% | Medium-small — partially lost through downsampling stages |
| traffic light, bicycle, motorcycle | 0–6% | Sub-pixel or near-sub-pixel in 3×11 deepest feature map |

The pattern is unambiguous: semantic controllability degrades monotonically with object size, because the conditioning signal enters once and dilutes through successive downsampling.

---

## 2. Proposed Change: Multi-Scale Semantic Re-Injection

### 2.1 Core Idea

Keep the pretrained ControlNet blocks completely undisturbed. After each block completes its computation, add a fresh semantic signal — downsampled to match the current spatial resolution and projected to match the current channel count — to the block's output before it propagates to the next block.

This is a **post-block additive residual**, not a pre-block input modification.

### 2.2 Why Post-Block, Not Pre-Block

The distinction is critical and directly derives from ControlNet's design philosophy.

**Pre-block injection (incorrect)**:
```python
sample = sample + sem_signal(downsampled)   # ← modifies block input
sample, res = block(sample)                  # block sees out-of-distribution input
```
The pretrained block's internal ResNet layers and temporal attention mechanisms now operate on an input that deviates from what they were pretrained to process. As the projector learns non-zero values, the input distribution drifts. This risks instability and degrades the pretrained alignment.

**Post-block injection (correct)**:
```python
sample, res = block(sample)                  # block always sees its natural input
sample = sample + sem_signal(downsampled)    # semantic enrichment added to OUTPUT
```
The pretrained block is never disturbed — it always processes the same distribution it saw during SVD pretraining. The semantic signal is added as an additive residual to the block's output. Zero-initialization guarantees that at training step 0, this is byte-for-byte identical to the baseline. Gradients flow through the zero-initialized projector and it gradually learns to enrich the inter-block representation.

This is philosophically identical to how ControlNet conditions the UNet: **zero-initialized additive residuals that preserve pretrained behavior**.

### 2.3 Proposed Forward Pass (Pseudocode)

```
Vanilla ControlNet:

  sample = conv_in(noisy) + control_conv_in(semantic)   ← injection at input
  for i, block in enumerate(down_blocks):
      sample, res = block(sample)                        ← no semantic refresh
      collect res
  mid = mid_block(sample)


Proposed (multi-scale re-injection):

  control_cond_orig = semantic                           ← save original [B*F, 4, H, W]
  sample = conv_in(noisy) + control_conv_in(semantic)   ← level-0 injection (unchanged)
  for i, block in enumerate(down_blocks):
      sample, res = block(sample)                        ← block UNCHANGED, sees natural input

      ↓ POST-BLOCK ↓
      sem_i = interpolate(control_cond_orig, size=sample.shape[-2:])   ← downsample to current H, W
      sample = sample + sem_scale_projectors[i](sem_i)                  ← add semantic residual

      collect res
  mid = mid_block(sample)
```

### 2.4 What Each Injector Does

`sem_scale_projectors` is a `nn.ModuleList` of 4 lightweight `Conv2d(4 → C_i, kernel_size=1)` layers, where `C_i = block_out_channels[i]` is the output channel count of `down_block[i]`:

| Injector | Input channels | Output channels | Applied spatial size | After block |
|---|---|---|---|---|
| `sem_scale_projectors[0]` | 4 | 320 | 12 × 44 | `down_block[0]` |
| `sem_scale_projectors[1]` | 4 | 640 | 6 × 22 | `down_block[1]` |
| `sem_scale_projectors[2]` | 4 | 1280 | 3 × 11 | `down_block[2]` |
| `sem_scale_projectors[3]` | 4 | 1280 | 3 × 11 | `down_block[3]` |

Each projector is **zero-initialized** (`zero_module` wrapper), consistent with ControlNet's existing convention for stable training.

The semantic latents are downsampled with `F.interpolate(..., mode='bilinear', align_corners=False)` — simple and differentiable, matching the spatial size of `sample` after each block's internal downsampler runs.

### 2.5 Why This Works

At deeper blocks, the ControlNet now processes `sample` that has been explicitly enriched with a spatially-downsampled but semantically-fresh version of the conditioning. The block's own computations (ResNet + temporal attention) then transform this into a semantically-grounded feature map at the appropriate depth. The resulting residuals injected into the UNet carry genuine semantic content at every level — not just the first.

---

## 3. Side-by-Side Comparison

| Aspect | Vanilla ControlNet | Proposed (Multi-Scale) |
|---|---|---|
| Semantic injection points | 1 (input only) | 5 (input + after each of 4 blocks) |
| Injection mechanism | `control_conv_in` (3×3 conv, 4→320) | `control_conv_in` + 4 × `Conv2d(4→C_i, 1×1)` |
| Pretrained block inputs | Natural (undisturbed) | Natural (undisturbed) ✓ |
| Semantic content at `mid_block` | Very weak (8× diluted) | Refreshed (explicit injection after block 3) |
| ControlNet output interface | 13 residuals + 1 mid | 13 residuals + 1 mid (unchanged) |
| New parameters | 0 | 4 × Conv2d(4→C_i, 1×1) ≈ 18K params |
| Zero-init guarantee | ✓ (controlnet_down_blocks) | ✓ (both existing + new projectors) |
| Backward compatible | N/A | ✓ (flag off = identical to baseline) |
| Training stability | Baseline | Identical at step 0, smooth learning |

---

## 4. Code Changes Required

### 4.1 `src/ctrlv/models/controlnet.py`

**Change 1 — Add `use_multiscale_conditioning` parameter to `__init__`:**

```python
@register_to_config
def __init__(
    self,
    # ... existing params ...
    num_frames: int = 25,
    use_multiscale_conditioning: bool = False,   # ← NEW
):
```

**Change 2 — Instantiate projectors when flag is True (inside `__init__`, after `mid_block`):**

```python
# Multi-scale semantic re-injection projectors (post-block, zero-initialized)
# sem_scale_projectors[i]: Conv2d(4 → block_out_channels[i], kernel_size=1)
# Applied AFTER down_block[i], so output channels match block[i]'s output
if use_multiscale_conditioning:
    self.sem_scale_projectors = nn.ModuleList([
        zero_module(nn.Conv2d(in_channels // 2, ch, kernel_size=1))
        for ch in block_out_channels
    ])
```

Note: `in_channels // 2 = 4` — the semantic latent channel count, same as `control_conv_in` uses.

**Change 3 — Modify `forward()` to re-inject after each block:**

```python
# Save original semantic conditioning at full resolution before any processing
control_cond_orig = control_cond  # [B*F, 4, H, W]

# Pre-process (unchanged)
sample = self.conv_in(sample)
control_cond = self.control_conv_in(control_cond)
sample = sample + control_cond

image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

down_block_res_samples = (sample,)
for i, downsample_block in enumerate(self.down_blocks):
    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )
    else:
        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
            image_only_indicator=image_only_indicator,
        )

    # Post-block semantic re-injection: add fresh semantic at current spatial scale
    if self.use_multiscale_conditioning:
        sem_at_scale = F.interpolate(
            control_cond_orig,
            size=sample.shape[-2:],          # match current H, W after block's downsampler
            mode='bilinear',
            align_corners=False
        )
        sample = sample + self.sem_scale_projectors[i](sem_at_scale)

    down_block_res_samples += res_samples
```

**Change 4 — Update `from_unet` to accept and forward the flag:**

```python
@classmethod
def from_unet(cls,
    unet: UNetSpatioTemporalConditionModel,
    load_weights_from_unet: bool = True,
    use_multiscale_conditioning: bool = False,   # ← NEW
):
    ctrlnet = cls(
        # ... existing params ...
        use_multiscale_conditioning=use_multiscale_conditioning,  # ← NEW
    )
    # weight copying logic unchanged
    return ctrlnet
```

### 4.2 `tools/train_video_controlnet.py`

**Change 5 — Pass flag to `from_unet` (line ~154):**

```python
ctrlnet = ControlNetModel.from_unet(
    unet,
    use_multiscale_conditioning=args.use_multiscale_conditioning,  # ← NEW
)
```

**Change 6 — Load existing checkpoint with strict=False when flag is True:**

When loading from a checkpoint trained without multi-scale (strict=True would fail because `sem_scale_projectors` is new). The `load_model_hook` in the accelerator hook should handle this:

```python
def load_model_hook(models, input_dir):
    for _ in range(len(models)):
        model = models.pop()
        if isinstance(model, ControlNetModel):
            load_model = ControlNetModel.from_pretrained(input_dir, subfolder="control_net")
            model.register_to_config(**load_model.config)
            # Use strict=False so new sem_scale_projectors (not in old checkpoint)
            # initialize from scratch while all existing weights load normally
            missing, unexpected = model.load_state_dict(load_model.state_dict(), strict=False)
            if missing:
                logger.info(f"New layers (initialized from scratch): {missing}")
```

### 4.3 `src/ctrlv/utils/parser.py`

**Change 7 — Add `--use_multiscale_conditioning` argument:**

```python
parser.add_argument(
    "--use_multiscale_conditioning",
    action="store_true",
    default=False,
    help="(ControlNet only). Inject fresh semantic features after each down_block "
         "at the matching spatial scale. Zero-initialized projectors ensure backward "
         "compatibility with existing checkpoints when loading with strict=False.",
)
```

### 4.4 `scripts/train_scripts/train_kitti360_sem2video.sh`

**Change 8 — Add flag to training script:**

```bash
# Add to the accelerate launch command:
--use_multiscale_conditioning \
# Recommended: start from best UNet-unfreeze checkpoint
--resume_from_checkpoint /path/to/checkpoint-22700 \
```

---

## 5. Backward Compatibility

The design guarantees full backward compatibility:

- **Flag off** (`use_multiscale_conditioning=False`, the default): `sem_scale_projectors` is not instantiated. Architecture is byte-for-byte identical to the current implementation. Any existing checkpoint loads with `strict=True`.

- **Flag on, loading old checkpoint**: `sem_scale_projectors` are new layers not present in the old checkpoint. Load with `strict=False` — existing weights load normally, new layers initialize from scratch (effectively zero because of `zero_module`). Training continues from the checkpoint with the new layers warming up.

- **Flag on, loading new checkpoint**: All weights including `sem_scale_projectors` load normally with `strict=True`. The `use_multiscale_conditioning=True` flag is stored in `config.json` by `@register_to_config`, so `from_pretrained` reconstructs the correct architecture automatically.

---

## 6. Expected Gains

Based on the analysis in `controlnet_injection_analysis.md` and the observed per-class failure pattern:

| Metric | Expected change | Reasoning |
|---|---|---|
| mIoU (overall) | +4–8% | Mid-scale classes gain from semantically-refreshed deep features |
| Tier 2 classes (pole, fence, sign) | Larger improvement | These fail specifically because deep residuals lack spatial precision |
| Tier 3 classes (traffic light, motorcycle) | Minimal change | Bottleneck is latent spatial resolution, not feature semantics |
| FVD | -80 to -150 | Frame-to-frame semantic consistency improves when UNet receives better residuals at all depths |
| FID | -5 to -15 | Better semantic adherence → generated frames closer to real semantic layout |

The gains compound with the UNet-unfreezing already in place: the unfrozen UNet mid+output blocks will receive better-grounded residuals from all ControlNet levels, not just the shallow ones.

---

## 7. Training Strategy

1. **Start from**: Best UNet-unfreeze checkpoint (`checkpoint-22700` or latest)
2. **Initial phase** (first ~3K steps): The `sem_scale_projectors` are zero-initialized — training is identical to baseline, no instability
3. **Learning phase**: Projectors gradually learn to re-inject semantic signal; UNet unfrozen blocks continue adapting
4. **LR**: Same as current (`1e-5` for ControlNet, `1e-6` or current for UNet)
5. **Evaluation**: Run full 150-sample eval and compare against `eval_stage2_rgb_unet_unfreeze/` baseline
