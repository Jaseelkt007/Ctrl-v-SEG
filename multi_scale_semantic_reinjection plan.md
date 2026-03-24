                                                                                                                                                  
  ---                                                                                                                                               
  Implementation Plan: Multi-Scale Semantic Re-Injection into ControlNet                                                                            
                                                                                                                                                    
  Problem Recap                                                                                                                                     
                                                                                                                                                    
  After sample = sample + control_cond (line 299), semantic information is fused once and never refreshed. Through 3 stride-2 downsamplers the      
  hidden state drifts from h₀ = f(RGB, noise, semantic) toward h₃ ≈ f(RGB, noise) — the RGB/noise high-entropy signal dominates, erasing
  fine-grained semantic structure before the ControlNet can form useful skip-connection residuals for the UNet decoder.                             
                  
  ---
  Core Idea: Fresh Semantic Re-injection After Each Downsampling Block
                                                                                                                                                    
  After each block that spatially downsizes the trunk (blocks 0, 1, 2), downsample the original 4-channel semantic latent to the new spatial
  resolution and project it back into the feature dimension. Because the injectors start zero-initialized, training begins from checkpoint-22700's  
  exact behavior and gradually learns the re-injection signal.
                                                                                                                                                    
  Input level   [B*F, 320, 24×88]   control_conv_in (existing, trained)                                                                             
                      ↓                                                                                                                             
  down_block[0]                                                                                                                                     
    res_samples (idx 1,2,3) collected                                                                                                               
    trunk → [B*F, 320, 12×44]                                                                                                                       
                      ↓                                                                                                                             
            + projector[0](bilinear↓(semantic_orig, size=12×44))                                                                                    
              → [B*F, 320, 12×44]   ← FRESH INJECT #1                                                                                               
                      ↓                                                                                                                             
  down_block[1]                                                                                                                                     
    res_samples (idx 4,5,6) computed from semantically-refreshed trunk                                                                              
    trunk → [B*F, 640, 6×22]                                                                                                                        
                      ↓                                                                                                                             
            + projector[1](bilinear↓(semantic_orig, size=6×22))                                                                                     
              → [B*F, 640, 6×22]    ← FRESH INJECT #2                                                                                               
                      ↓                                                                                                                             
  down_block[2]
    res_samples (idx 7,8,9) computed from refreshed trunk                                                                                           
    trunk → [B*F, 1280, 3×11]
                      ↓                                                                                                                             
            + projector[2](bilinear↓(semantic_orig, size=3×11))
              → [B*F, 1280, 3×11]   ← FRESH INJECT #3                                                                                               
                      ↓
  down_block[3]  (no downsample, same resolution)                                                                                                   
  mid_block                                                                                                                                         
    → zero-conv residuals now carry genuine semantic context at every depth
                                                                                                                                                    
  ---                                                                                                                                               
  File to Modify: src/ctrlv/models/controlnet.py                                                                                                    
                                                                                                                                                    
  Only this one file needs changes. The UNet, training scripts, and pipelines are untouched.
                                                                                                                                                    
  ---
  Change 1: __init__ — Add semantic_scale_projectors                                                                                                
                                                                                                                                                    
  Location: After line 141 (after self.control_conv_in).
                                                                                                                                                    
  # Multi-scale semantic re-injection projectors.
  # One per downsampling boundary (after blocks 0, 1, 2).                                                                                           
  # Zero-initialized: training starts from pretrained checkpoint behavior                                                                           
  # and gradually learns the re-injection signal.                                                                                                   
  semantic_channels = in_channels // 2  # = 4  (same as control_conv_in input)                                                                      
  self.semantic_scale_projectors = nn.ModuleList([                                                                                                  
      zero_module(nn.Conv2d(semantic_channels, block_out_channels[0], kernel_size=3, padding=1)),  # 4→320                                          
      zero_module(nn.Conv2d(semantic_channels, block_out_channels[1], kernel_size=3, padding=1)),  # 4→640                                          
      zero_module(nn.Conv2d(semantic_channels, block_out_channels[2], kernel_size=3, padding=1)),  # 4→1280                                         
  ])                                                                                                                                                
                                                                                                                                                    
  Why block_out_channels[0] for injector 0 (not block_out_channels[1])?                                                                             
  After block 0's downsampler, the trunk is still at block_out_channels[0]=320 channels (the downsampler only changes spatial resolution, not
  channels). After block 1's downsampler, trunk is at block_out_channels[1]=640. Confirmed from the shape table in the docs.                        
                  
  Why zero-initialization?                                                                                                                          
  Matches the ControlNet design philosophy (all zero-convs start at zero output). Ensures that loading checkpoint-22700 gives identical initial
  behavior — the new injectors contribute nothing on step 0 and training discovers the injections organically.                                      
  
  ---                                                                                                                                               
  Change 2: forward — Save Original Semantic Latent
                                                                                                                                                    
  Location: After line 289 (control_cond = control_cond.flatten(0, 1)), before line 297.
                                                                                                                                                    
  # Save original semantic latent [B*F, 4, H, W] before it gets projected.
  # This is used for multi-scale re-injection after each downsampling block.                                                                        
  control_cond_original = control_cond                                                                                                              
                                                                                                                                                    
  The existing lines 297–299 stay exactly as-is:                                                                                                    
  sample = self.conv_in(sample)
  control_cond = self.control_conv_in(control_cond)                                                                                                 
  sample = sample + control_cond                                                                                                                    
  
  ---                                                                                                                                               
  Change 3: forward — Re-injection Inside the Down Block Loop
                                                                                                                                                    
  Location: Replace lines 303–319 (the down block loop). The hasattr conditional branches are preserved exactly; only the loop counter and
  post-block injection are new.                                                                                                                     
                  
  down_block_res_samples = (sample,)                                                                                                                
  inject_idx = 0  
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
                  
      down_block_res_samples += res_samples                                                                                                         
                  
      # Re-inject fresh semantic conditioning after each block that has a downsampler.                                                              
      # Blocks 0, 1, 2 have downsamplers (is_final_block=False).
      # Block 3 (final) does not — no injection there.                                                                                              
      if i < len(self.down_blocks) - 1:                                                                                                             
          semantic_rescaled = F.interpolate(                                                                                                        
              control_cond_original,                                                                                                                
              size=sample.shape[-2:],   # dynamically matches 12×44, 6×22, 3×11                                                                     
              mode='bilinear',                                                                                                                      
              align_corners=False,
          )                                                                                                                                         
          sample = sample + self.semantic_scale_projectors[inject_idx](semantic_rescaled)
          inject_idx += 1

  Key detail: The injection happens AFTER down_block_res_samples += res_samples. This means:                                                        
  - Block N's own internal residuals (already collected) are unmodified.
  - The refreshed sample enters block N+1 and shapes block N+1's internal residuals — which are the semantically weaker ones we care most about     
  improving.                                                                                                                                   
                                                                                                                                                    
  Why sample.shape[-2:] instead of hardcoded sizes?
  Defensive: works correctly regardless of training resolution and doesn't break if resolution changes.                                             
                                                                                                                                                    
  ---                                                                                                                                               
  from_unet — No Changes Needed                                                                                                                     
                                                                                                                                                    
  The from_unet classmethod copies keys that appear in both the UNet and ControlNet state dicts (via intersection_keys). semantic_scale_projectors
  has no UNet counterpart, so it gets no copied weights and remains zero-initialized. Correct.                                                      
                  
  ---                                                                                                                                               
  New Parameter Count
                     
  ┌──────────────────────────────┬─────────────────────┬──────────────────────────────────┐
  │            Layer             │        Shape        │              Params              │                                                         
  ├──────────────────────────────┼─────────────────────┼──────────────────────────────────┤                                                         
  │ semantic_scale_projectors[0] │ Conv2d(4→320, k=3)  │ 4 × 320 × 3 × 3 + 320 = 11,840   │                                                         
  ├──────────────────────────────┼─────────────────────┼──────────────────────────────────┤                                                         
  │ semantic_scale_projectors[1] │ Conv2d(4→640, k=3)  │ 4 × 640 × 3 × 3 + 640 = 23,680   │                                                         
  ├──────────────────────────────┼─────────────────────┼──────────────────────────────────┤                                                         
  │ semantic_scale_projectors[2] │ Conv2d(4→1280, k=3) │ 4 × 1280 × 3 × 3 + 1280 = 47,360 │                                                         
  ├──────────────────────────────┼─────────────────────┼──────────────────────────────────┤                                                         
  │ Total new params             │                     │ ~82K                             │
  └──────────────────────────────┴─────────────────────┴──────────────────────────────────┘                                                         
                  
  Negligible relative to the ~1.5B ControlNet. No memory budget impact.                                                                             
                  
  ---                                                                                                                                               
  Training Strategy
                                                                                                                                                    
  Recommended: Fine-tune from checkpoint-22700 with the new zero-init projectors.
                                                                                                                                                    
  sbatch scripts/train_scripts/train_kitti360_sem2video.sh \
      --resume_from_checkpoint /no_backups/s1492/Ctrl-V/checkpoints/kitti360_sem2video_unet_unfreeze/checkpoint-22700                               
                                                                                                                                                    
  Because the projectors are zero-initialized, day-0 loss and behavior is identical to checkpoint-22700. The training loop will discover gradient   
  signal through the new injection points and begin using them automatically.                                                                       
                                                                                                                                                    
  The existing UNet unfreeze configuration (mid+output blocks unfrozen) should be kept — those unfrozen UNet blocks were already adapting to the    
  single-scale ControlNet signal and will now adapt to the richer multi-scale signal.
                                                                                                                                                    
  ---             
  What Will Improve and Why
                           
  ┌─────────────┬──────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┐
  │    Tier     │         Classes          │                                          Expected Change                                          │    
  ├─────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Tier 1      │ road, car, sky,          │ Minimal — already well-served by single injection                                                 │    
  │ (>60% IoU)  │ vegetation, building     │                                                                                                   │
  ├─────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤    
  │ Tier 2      │ sidewalk, pole, fence,   │ Primary target — these are mid-scale objects whose semantic signal was surviving at H/2 but       │    
  │ (20–60%)    │ traffic sign, person     │ getting lost by H/4 or H/8. Fresh injection at those depths keeps them legible in the skip        │    
  │             │                          │ connections the UNet decoder uses                                                                 │    
  ├─────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Tier 3      │ traffic light,           │ No change from this fix alone — these are sub-pixel in the 24×88 latent, not a conditioning depth │
  │ (0–13%)     │ motorcycle, bicycle, bus │  problem                                                                                          │    
  └─────────────┴──────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                                                                                    
  Expected mIoU gain: +4–8% on Tier 2 classes, consistent with the analysis in controlnet_injection_analysis.md Section 6.                          
  
  ---                                                                                                                                               
  Complete Diff Summary
                                                                                                                                                    
  src/ctrlv/models/controlnet.py
    __init__:                                                                                                                                       
      + import F (already imported via torch.nn.functional as F)
      + self.semantic_scale_projectors = nn.ModuleList([...])  # 3 zero-init Conv2d                                                                 
                                                                                                                                                    
    forward:                                                                                                                                        
      + control_cond_original = control_cond            # save before projection                                                                    
      ~ down block loop: add inject_idx counter
      + after each non-final block: F.interpolate + semantic_scale_projectors[inject_idx]                                                           
                                                                                                                                                    
  No other files touched.