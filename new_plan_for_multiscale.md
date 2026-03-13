---                                     
  Analysis: Multi-Scale Semantic Conditioning                                                                                                                                   
                                
  Current Metrics (What We're Working With)                                                                                                                                     
                                                                                                                                                                                
  ┌───────────┬────────────────────────────────┬──────────────────────────────────────────────┐                                                                                 
  │  Metric   │ Frozen UNet (checkpoint-96100) │ Unfrozen UNet (checkpoint-22700, 20 samples) │                                                                                 
  ├───────────┼────────────────────────────────┼──────────────────────────────────────────────┤                                                                                 
  │ mIoU      │ 23.20%                         │ 42.92% (+18.72pp)                            │                                                                                 
  ├───────────┼────────────────────────────────┼──────────────────────────────────────────────┤                                                                                 
  │ Pixel Acc │ 67.63%                         │ 89.04% (+21.41pp)                            │
  ├───────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ FID       │ 72.18                          │ N/A (not computed yet)                       │
  ├───────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ FVD       │ 861.70                         │ N/A (not computed yet)                       │
  ├───────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ PSNR      │ 10.72 dB                       │ N/A                                          │
  └───────────┴────────────────────────────────┴──────────────────────────────────────────────┘

  The UNet unfreezing was a massive win. Now let me assess what multi-scale conditioning would do.

  ---
  Why the Current Single-Scale Injection is Limiting

  The current flow in controlnet.py is:
  semantic_latents [B*F, 4, H, W]
      → control_conv_in (single 3×3 conv)  →  [B*F, 320, H, W]
      → ADD to conv_in(noisy_latents)          (injected once, at full resolution)
      → down_block[0]  [320→640, H → H/2]     (semantic signal diluted here)
      → down_block[1]  [640→1280, H/2 → H/4]  (further diluted)
      → down_block[2]  [1280→1280, H/4 → H/4] (further diluted)
      → mid_block                              (semantic signal survives only weakly)

  The semantic signal must survive 3 downsampling stages after a single-point injection. By the mid-block, the original semantic structure is severely diluted. The ControlNet
  residuals injected into the UNet carry weaker semantic signal at each level below the input.

  ---
  Would Multi-Scale Conditioning Help?

  Short answer: Yes, meaningfully — but not for all problems equally.

  Where it WILL help:

  1. Mid-scale objects (cars, buildings, poles, vegetation edges): These structures span 5-50 pixels in the latent space (24×88). A multi-scale injection keeps semantic
  features alive at the spatial resolution where these objects matter. Currently, by the time the ControlNet's mid-block residual reaches the UNet, a car boundary that was
  sharp at H×W has been averaged across 4 or 8 pixels. Estimated mIoU gain: +4-8% on these classes.
  2. FVD improvement: FVD measures video temporal coherence. Multi-scale injection means the UNet's up-blocks receive more semantically grounded residuals. The up-blocks
  reconstruct spatial detail — if they get better semantic signals, frame-to-frame semantic layout consistency improves. Estimated FVD reduction: -80 to -200 (moderate,
  meaningful).
  3. FID improvement: Better semantic adherence = generated frames closer to real semantic structure = marginally better FID. Estimated: -5 to -15 (small but positive).

  Where it WILL NOT help much:

  - Traffic lights (0% IoU), bicycle: The fundamental bottleneck is latent spatial resolution — 24×88 means a 2px traffic light in latent space is 1 pixel or less. Multi-scale
  conditioning cannot recover sub-pixel objects. This requires higher input resolution, not better conditioning architecture.
  - Large static regions (road, sky): Already well-controlled even with single-scale injection (70-91% IoU). These survive down-sampling easily.

  Honest expected gains after UNet unfreezing:

  ┌───────────┬───────────────────────────────────────────┐
  │  Metric   │              Expected Change              │
  ├───────────┼───────────────────────────────────────────┤
  │ mIoU      │ +4–8% (more on mid-scale classes)         │
  ├───────────┼───────────────────────────────────────────┤
  │ FVD       │ -80–200 (meaningful temporal improvement) │
  ├───────────┼───────────────────────────────────────────┤
  │ FID       │ -5–15 (marginal)                          │
  ├───────────┼───────────────────────────────────────────┤
  │ SSIM/PSNR │ Small positive improvement                │
  └───────────┴───────────────────────────────────────────┘

  ---
  Backward Compatibility Design

  The key constraint: existing checkpoint must still load and run identically when the flag is off.

  The solution is straightforward:

  - All new layers (hint encoder + scale projectors) only instantiated when use_multiscale_conditioning=True
  - When False: architecture is byte-for-byte identical to current; checkpoint loads with strict=True
  - When True: new layers initialize from scratch, checkpoint loads with strict=False (only new layers are missing — existing weights load normally)

  This means you can:
  1. Load the UNet-unfreeze checkpoint and evaluate baseline
  2. Load same checkpoint + new multi-scale layers (randomly initialized) and fine-tune further
  3. Compare both on same evaluation set

  ---
  Implementation Plan

  Step 1: Add MultiScaleHintEncoder to controlnet.py

  A 4-level feature pyramid that progressively downsamples semantic latents:
  Level 0: [4 → 32]  at H×W (full resolution)
  Level 1: [32 → 64]  at H/2×W/2 (stride-2 conv)
  Level 2: [64 → 128] at H/4×W/4 (stride-2 conv)
  Level 3: [128 → 256] at H/8×W/8 (stride-2 conv)
  Each level projected to match the corresponding down_block channel count via a 1×1 conv + zero_module (to match ControlNet's zero-init convention for stable training).

  Step 2: Modify ControlNetModel.__init__

  Add use_multiscale_conditioning: bool = False parameter. When True, instantiate the hint encoder and scale projectors alongside the existing control_conv_in (keep
  control_conv_in — it handles the base injection at level 0, multi-scale layers add on top).

  Step 3: Modify ControlNetModel.forward

  # Existing base injection (unchanged):
  sample = conv_in(sample) + control_conv_in(control_cond)  # Level 0

  if self.use_multiscale_conditioning:
      sem_features = self.hint_encoder(control_cond)  # Precompute pyramid

  for i, down_block in enumerate(self.down_blocks):
      if self.use_multiscale_conditioning and i < len(sem_features):
          sample = sample + self.scale_projectors[i](sem_features[i])
      sample, res_samples = down_block(sample, ...)
      down_block_res_samples += res_samples

  Step 4: Modify train_video_controlnet.py

  Add --use_multiscale_conditioning flag. Pass to ControlNetModel at init. When loading a checkpoint trained without this flag, use strict=False for the ControlNet only (so new
   hint encoder layers initialize from scratch while all existing weights load).

  Step 5: Training strategy

  - Start from the best UNet-unfreeze checkpoint (checkpoint-22700 or later)
  - Freeze the UNet layers that are currently unfrozen initially for ~5K steps to let hint encoder warm up
  - Then unfreeze UNet mid+output layers again for joint training
  - Use same LR as before (1e-5 for ControlNet, lower for UNet)

  Step 6: Evaluation

  Run full 150-sample eval on the new checkpoint and compare with:
  - eval_stage2_rgb/ (frozen UNet baseline)
  - eval_stage2_rgb_unet_unfreeze/ (UNet unfreeze, no multi-scale)
  - New: eval_stage2_rgb_multiscale/

  ---
  Recommendation

  Proceed with implementation. The expected +4-8% mIoU gain plus FVD improvement makes it worthwhile. More importantly, the combination of (UNet unfreezing) + (multi-scale
  conditioning) should push mIoU into the 48-52% range and bring FVD down significantly. These together move the model closer to publishable controllability numbers.

  The implementation is also architecturally clean — it doesn't break any existing code path and the backward compatibility is guaranteed by a single flag.

  Shall I start implementing these changes now?