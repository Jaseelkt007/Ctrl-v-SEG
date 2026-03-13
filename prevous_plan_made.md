# Comprehensive Analysis: Stage 2 ControlNet Limitations & Improvement Strategy

## Performance Summary Before Unfreezing the UNet

  ┌────────────┬─────────────┬──────────────┬─────────────┐
  │   Metric   │ Your Result │ Typical SOTA │     Gap     │
  ├────────────┼─────────────┼──────────────┼─────────────┤
  │ mIoU (DRN) │ 45.60%      │ 60-70%+      │ Significant │
  ├────────────┼─────────────┼──────────────┼─────────────┤
  │ FID        │ 68.47       │ 20-40        │ Moderate    │
  ├────────────┼─────────────┼──────────────┼─────────────┤
  │ FVD        │ 595.52      │ 200-400      │ Large       │
  ├────────────┼─────────────┼──────────────┼─────────────┤
  │ SSIM       │ 0.433       │ 0.6-0.8      │ Large       │
  ├────────────┼─────────────┼──────────────┼─────────────┤
  │ PSNR       │ 14.51 dB    │ 18-22 dB     │ Large       │
  └────────────┴─────────────┴──────────────┴─────────────┘

  The mIoU directly measures semantic controllability — how well the ControlNet steers the generation to match the input semantic maps. At 45.60%, roughly half the semantic
  intent is lost. Let me explain exactly why.

  ---
  Root Cause Analysis

  ## 1. Frozen UNet — The ControlNet Cannot Override SVD's Strong Priors

  This is the single biggest limitation.

  In train_video_controlnet.py:138:
  unet.requires_grad_(False)

  The UNet is completely frozen. The ControlNet can only nudge the UNet via additive residuals to its down/mid blocks (controlnet.py:333-344):

  # ControlNet output is ADDED to UNet's internal features
  down_block_res_sample = controlnet_block(down_block_res_sample)  # zero-initialized conv
  # ...
  down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]

  The zero-initialized controlnet_down_blocks start from zero and must learn to inject the right signal. But the frozen UNet has a massively strong prior from SVD pretraining on
   web-scale video. This prior favors generating "plausible-looking driving video" rather than "semantic-faithful driving video." For small/rare objects (traffic lights,
  bicycles, riders), the SVD prior simply overwhelms the ControlNet signal.

  Why small classes fail: The ControlNet residuals are spatially uniform in scale (conditioning_scale=1.0 everywhere). For a 5-pixel traffic light, the ControlNet signal is
  dwarfed by the UNet's prior to generate "typical road scene texture" in that region.

  ## 2. Latent-Space Semantic Encoding — Information Bottleneck

  The semantic maps go through a severe information bottleneck:

  Semantic IDs [H, W] (19 classes, pixel-perfect)
      → One-hot [19, H, W]
      → Semantic stem [128, H, W]
      → VAE encoder [4, H/8, W/8]    ← 8x spatial downsampling!

  At 192×704 input resolution, your semantic latents are 24×88 with 4 channels. Fine-grained semantic boundaries (thin poles, narrow sidewalks, small pedestrians) are inherently
   blurred. The ControlNet then processes these blurred latents through control_conv_in (a 3×3 conv from 4→320 channels) and adds them to the UNet features.

  Critical issue: The control_conv_in takes in_channels//2 = 4 channels (controlnet.py:136-141):
  self.control_conv_in = nn.Conv2d(in_channels//2, block_out_channels[0], kernel_size=3, padding=1)
  This is a single 3×3 conv — very limited capacity to project semantic latents into the UNet feature space. Compare this to ControlNet for Stable Diffusion 2D, which typically
  uses multi-layer hint encoders.

  ## 3. No Multi-Scale Conditioning — Only Single-Resolution Injection

  In controlnet.py:297-299, the semantic conditioning is injected at a single point:
  sample = self.conv_in(sample)        # Process noisy latent
  control_cond = self.control_conv_in(control_cond)  # Process semantic latent
  sample = sample + control_cond       # Single additive injection

  Then the combined signal goes through the entire down-block + mid-block pipeline. The semantic signal must survive multiple downsampling stages and attention layers. By the
  time it reaches deeper layers, the semantic information is diluted.

  SOTA ControlNets (e.g., ControlNet 1.1, T2I-Adapter) use multi-scale hint encoders that inject conditioning at every resolution level, not just the input.

  ## 4. No Conditioning Dropout During Training

  In the training script (train_kitti360_sem2video.sh), there is no --conditioning_dropout_prob flag. Looking at the training code (train_video_controlnet.py:448-467):

  if args.conditioning_dropout_prob is not None:
      # ... dropout logic for encoder_hidden_states and conditional_latents

  But this only drops the image conditioning (CLIP + first-frame latent), not the ControlNet semantic conditioning. The ControlNet conditioning (bbox_em) is never dropped during
   training. This means:
  - The model never learns to work without semantic guidance
  - Classifier-free guidance at inference (min_guidance_scale=1.0→max_guidance_scale=3.0) amplifies both the image and ControlNet signals together — but the ControlNet signal
  isn't independently guidable
  - You cannot independently scale semantic guidance vs image guidance

  ## 5. UNet First-Frame Conditioning Conflicts with ControlNet

  The pipeline concatenates the first RGB frame's latent with the noisy video latent:

  # train_video_controlnet.py:470-472
  conditional_latents = unet.encode_bbox_frame(conditional_latents, None)
  concatenated_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

  This means the UNet receives 8 input channels (4 noise + 4 first-frame). The encode_bbox_frame with None just repeats the first frame across all timesteps (unet:28-29). The
  UNet has a strong tendency to simply propagate the first frame forward, making the ControlNet semantic signal redundant for large, static regions (road, sky, building — which
  already have high IoU).

  The first-frame conditioning dominates for static elements but provides no guidance for dynamic/small objects, exactly where the ControlNet fails.

  ## 6. Training Setup Limitations

  ┌──────────────────────┬───────────────┬─────────────────────────────────────┐
  │        Issue         │    Current    │             Recommended             │
  ├──────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Effective batch size │ 1×4=4         │ 8-16+                               │
  ├──────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Training steps       │ 34K-68K       │ 100K-200K+                          │
  ├──────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Learning rate        │ 1e-5 constant │ 1e-5 with warmup + cosine decay     │
  ├──────────────────────┼───────────────┼─────────────────────────────────────┤
  │ LR scheduler         │ constant      │ cosine or cosine_with_restarts      │
  ├──────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Resolution           │ 192×704 (low) │ 256×960 or higher                   │
  ├──────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Gradient clipping    │ Not visible   │ max_grad_norm=1.0 recommended       │
  ├──────────────────────┼───────────────┼─────────────────────────────────────┤
  │ No UNet fine-tuning  │ Fully frozen  │ Joint fine-tuning (low LR for UNet) │
  └──────────────────────┴───────────────┴─────────────────────────────────────┘

  ## 7. Semantic VAE Scaling Factor Mismatch (Potential)

  In training (train_video_controlnet.py:419):
  target_latents = latents = latents * vae.config.scaling_factor

  The RGB target latents are scaled by the SVD VAE's scaling factor (0.18215). But the semantic conditioning latents from vae_manager.encode_semantic_from_ids() are NOT scaled
  by this factor — they come from the semantic VAE encoder directly. This means the ControlNet sees semantic latents in a different magnitude range than the noisy latents it's
  processing. While the zero-initialized projections can theoretically learn to compensate, this scale mismatch makes optimization harder.

  ## 8. Temporal Modeling Weakness

  SVD processes temporal attention via image_only_indicator set to zeros, meaning it's in video mode. But the ControlNet processes each frame's semantic conditioning
  independently at the input level (control_conv_in is a 2D conv), and temporal mixing only happens inside the spatiotemporal transformer blocks. This means:
  - The ControlNet doesn't model temporal evolution of semantic layouts explicitly
  - Frame-to-frame consistency of semantic control is weak (contributing to high FVD)

## Per-Class Failure Analysis

  ┌────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────┬────────────┐
  │       Class        │                                              IoU                                              │ Root Cause │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ traffic light (0%) │ Zero or near-zero training examples; too small (few pixels) to survive 8× latent downsampling │            │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ bus/train (0%)     │ Extremely rare in KITTI-360; the model never learned these                                    │            │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ bicycle (3.66%)    │ Thin structures lost in latent space; confused with background                                │            │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ rider (19%)        │ Small, rare, confused with person                                                             │            │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ person (29.6%)     │ Small objects; ControlNet signal too weak vs UNet prior                                       │            │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ pole (37.8%)       │ Thin vertical structures; latent bottleneck destroys them                                     │            │
  └────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────┴────────────┘

  The pattern is clear: spatial resolution loss in latent space + frozen UNet prior = failure on small/thin/rare objects.

  ---
  Recommended Improvements (Ranked by Impact)

  High Impact

  1. Joint UNet + ControlNet Fine-Tuning (Partial UNet Unfreezing)
    - Unfreeze the UNet's up_blocks and mid_block while keeping down_blocks frozen
    - Use a lower LR for UNet (e.g., 1e-6) vs ControlNet (1e-5)
    - This lets the decoder adapt to semantic conditioning signals
  2. Multi-Scale Semantic Conditioning
    - Replace the single control_conv_in with a multi-scale hint encoder (4-5 conv layers with progressive downsampling)
    - Inject semantic features at each resolution level of the ControlNet, not just the input
    - This is how ControlNet 1.1 and T2I-Adapter achieve high controllability
  3. Scale Semantic VAE Latents
    - Apply vae.config.scaling_factor to semantic latents during training (same as RGB latents), or normalize semantic latents to match the noise latent distribution
    - This ensures the ControlNet sees inputs in a consistent magnitude range

  Medium Impact

  4. Increase Training Duration + Schedule
    - Train to 150K-200K steps
    - Use cosine LR schedule with warmup (e.g., 1000 steps warmup)
    - Consider increasing effective batch size to 8 (gradient_accumulation_steps=8)
  5. ControlNet Conditioning Dropout
    - Add explicit dropout for the ControlNet semantic conditioning (bbox_em) during training (e.g., 10% of the time, replace bbox_em with zeros)
    - This enables independent classifier-free guidance for semantic control at inference
    - At inference, use separate guidance scales for image vs semantic conditioning
  6. Higher Conditioning Scale at Inference
    - Try conditioning_scale=1.5 or 2.0 at inference to amplify ControlNet signal
    - This is a free improvement that doesn't require retraining
  7. Increase Resolution
    - Train at 256×960 instead of 192×704
    - This gives 32×120 latent resolution (vs 24×88), preserving more semantic detail
    - Requires more VRAM — may need to reduce clip length to 14 frames

  Lower Impact (But Still Valuable)

  8. Class-Balanced Loss or Focal Loss
    - Weight the MSE loss by per-class frequency inverse to give rare classes more gradient signal
    - This requires computing class presence per-pixel in latent space (approximate)
  9. Pixel-Space Semantic Loss
    - Add an auxiliary loss that decodes generated latents and computes semantic cross-entropy against GT semantic maps
    - This directly optimizes for semantic fidelity, not just latent-space MSE
  10. Temporal Semantic Conditioning
    - Process semantic conditioning with temporal convolutions before injection
    - Use a small temporal attention module on semantic latents before feeding to ControlNet