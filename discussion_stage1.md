# Discussion: Stage 1 Architecture, Current Limitations, and Recommended Next Updates

## Purpose

This note is intended as a thesis-oriented discussion of the current Stage 1 pipeline described in `pipeline_stage1.md`, cross-checked against the actual implementation in:

- `tools/train_video_diffusion.py`
- `src/ctrlv/pipelines/pipeline_video_diffusion.py`
- `src/ctrlv/models/dual_vae_manager.py`
- `scripts/train_scripts/train_kitti360_bbox_predict.sh`
- `tools/eval_stage1_semantic.py`

It also uses the current evaluation summary at:

- `/no_backups/s1492/Ctrl-V/outputs/eval_stage1_semantic/eval_summary.txt`
- `/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae/latent_statistics_report.txt`

The goal is not only to describe the architecture, but to explain why it behaves the way it does, which parts are likely fundamental bottlenecks, and which next updates are most justified.

---

## 1. What the current Stage 1 system is actually doing

Conceptually, Stage 1 is framed as:

> first RGB frame -> future 25-frame semantic segmentation sequence

However, the implementation is more specific than that summary:

1. The semantic target sequence is encoded by the Semantic VAE into a latent tensor of shape `[B, T, 4, 24, 88]` for `192 x 704` input.
2. The first semantic frame is encoded once and repeated across the intermediate temporal slots.
3. The conditioning tensor is built as:
   - frame 0: GT semantic latent
   - frames 1-23: repeated first-frame semantic latent
   - frame 24: GT semantic latent
4. This conditioning tensor is concatenated with the noisy target latents on the channel dimension and fed into the SVD UNet.
5. The first RGB frame is also encoded by CLIP, but only as a single global token.

This means the model is not simply asked to predict future semantics from RGB alone. It is asked to denoise a future semantic sequence while being given:

- a strong first-frame semantic anchor
- a repeated first-frame semantic prior in all intermediate slots
- a last-frame semantic anchor during training and evaluation
- a single global CLIP token from the first RGB frame

That design already suggests a strong bias toward temporal persistence and interpolation between anchors, rather than free future forecasting.

---

## 2. Important implementation clarifications versus `pipeline_stage1.md`

Several details in `pipeline_stage1.md` do not exactly match the current run used for the reported evaluation.

### 2.1 The current best run is not temporal-only training

`pipeline_stage1.md` states that only temporal transformer blocks are trained. That is not the current training configuration used by `scripts/train_scripts/train_kitti360_bbox_predict.sh`.

The script sets:

```bash
--backprop_temporal_blocks_start_iter -1
```

In `tools/train_video_diffusion.py`, this enters the `enable_grad(all=True)` branch, which means the full UNet is trainable from the beginning.

This is an important finding:

- the current Stage 1 result is already a full-UNet fine-tuning result
- therefore, the remaining failure modes cannot be explained only by "the network was too frozen"

They are more likely to be representational and conditioning bottlenecks.

### 2.2 The current evaluation is not pure first-frame-only forecasting

During evaluation, `tools/eval_stage1_semantic.py` passes the full semantic clip to the pipeline as `semantic_ids`.

Inside `src/ctrlv/pipelines/pipeline_video_diffusion.py`, the code:

- encodes frame 0 semantic IDs into the repeated base conditioning
- then encodes the full semantic clip through `_encode_vae_condition(...)`
- then copies GT conditioning into:
  - frame 0
  - frame 24

So the reported evaluation is boundary-conditioned: it uses future semantic information at frame 24.

This does not make the output frame 24 a hard copy, but it does make the task easier than true "first RGB frame only -> future semantic sequence" prediction, because future semantic information is injected into every denoising step.

### 2.3 The current run-specific hyperparameters differ from the markdown examples

The current training/evaluation scripts use:

- `min_guidance_scale = 3`
- `max_guidance_scale = 7`
- `noise_aug_strength = 0.01`
- `conditioning_dropout_prob = 0.0`

`pipeline_stage1.md` documents a milder example configuration in some sections. For thesis discussion, it is safer to distinguish:

- architectural design
- actual experimental configuration

---

## 3. What the current results say

From `/no_backups/s1492/Ctrl-V/outputs/eval_stage1_semantic/eval_summary.txt`:

- dataset-level mIoU: `35.45%`
- pixel accuracy: `80.13%`
- mean accuracy: `45.41%`
- FW-IoU: `67.71%`

Per-class IoU shows a very clear pattern:

- large/static classes are comparatively strong
  - road: `84.39%`
  - building: `67.05%`
  - vegetation: `72.02%`
  - sky: `69.43%`
- thin/small/rare classes collapse
  - pole: `10.20%`
  - traffic light: `0.00%`
  - rider: `5.60%`
  - bus: `6.63%`
  - train: `0.02%`
  - bicycle: `0.00%`

Two grouped summaries make the pattern even sharper:

- large static classes average: `65.86%`
  - road, sidewalk, building, wall, vegetation, terrain, sky
- thin/small classes average: `7.81%`
  - pole, traffic light, traffic sign, person, rider, bus, train, motorcycle, bicycle

At clip level:

- average clip mIoU: `42.85% +/- 10.55%`
- median clip mIoU: `43.24%`
- Q1 / Q3: `36.24% / 49.83%`
- range: `13.49%` to `71.19%`

An especially important observation is that some clips have very high pixel accuracy but poor mIoU, for example:

- `24.26%` mIoU with `99.25%` pixel accuracy
- `28.27%` mIoU with `97.24%` pixel accuracy

This indicates that the model is often getting dominant background classes right while failing minority classes. In other words, the system is learning scene persistence, not balanced semantic forecasting.

---

## 4. Strengths of the current design

Before discussing the bottlenecks, it is important to state what the architecture is already doing well.

### 4.1 It preserves global scene layout well

The strong performance on road, building, vegetation, terrain, and sky shows that the model can preserve coarse scene topology and static geometry.

This is not trivial. It means:

- the semantic VAE is producing a usable latent representation
- the SVD backbone can be adapted to semantic outputs at least at coarse scale
- the first-frame conditioning and CLIP token provide enough information to stabilize large regions

### 4.2 It can produce good clips under favorable conditions

The best clip mIoU reaches `71.19%`. This suggests the model is not globally broken. Instead, it succeeds when:

- scene geometry changes slowly
- large classes dominate
- there are fewer thin or rare objects
- future layout remains close to the initial frame prior

This matters because it argues against a total architectural mismatch. The issue is not that Stage 1 cannot work at all. The issue is that it works mainly in the easiest regime.

---

## 5. Main architectural limitations

### 5.1 First-frame semantic repetition creates a strong copy-forward bias

The core Stage 1 conditioning pattern is:

- frame 0 = GT semantic latent
- frames 1-23 = repeated frame-0 semantic latent
- frame 24 = GT semantic latent

This has a clear inductive bias: the easiest solution is to propagate the initial layout forward unless the network has strong evidence to change it.

That bias is particularly strong because the repeated conditioning enters through channel concatenation at the very input of the UNet. Every intermediate frame starts with a stale semantic prior that says:

> the scene still looks like frame 0

This naturally helps:

- road
- sky
- building
- vegetation

and hurts:

- pedestrians entering or leaving view
- riders, bicycles, motorcycles
- traffic lights and poles
- thin boundaries and small moving objects

The evaluation results match this expectation almost perfectly.

### Why skip connections make this worse

The UNet has standard encoder-decoder skip connections. These are useful for preserving detail, but here they also preserve stale first-frame structure. Future-specific changes must therefore be expressed as corrections against a strong copy-forward prior.

For large static regions this is fine.

For small dynamic structures this is much harder, because the model must erase or rewrite features that were injected as conditioning from the very beginning.

### 5.2 The semantic representation is too compressed for small or thin classes

The semantic maps are resized to `192 x 704`, then encoded by the Semantic VAE to `24 x 88`, i.e. `H/8, W/8`.

That means one latent cell corresponds to an `8 x 8` patch in the resized image.

This is already aggressive for a categorical signal such as semantic segmentation, where:

- class boundaries matter
- thin poles and traffic lights occupy very few pixels
- small agents may span only a handful of pixels at this resolution

The problem is then amplified inside the diffusion UNet:

- `24 x 88`
- `12 x 44`
- `6 x 22`
- `3 x 11`

So the deepest bottleneck has only `33` spatial locations per frame.

### Important nuance

The presence of skip connections means the model is not forced to reconstruct everything from `3 x 11` alone.

However, skip connections mostly preserve the early conditioned structure, which for frames 1-23 is just repeated frame-0 semantics. Therefore, truly future-specific changes still need to be carried through extremely low-resolution pathways.

This explains why:

- coarse static layout survives
- thin or newly appearing structures do not

### Relation to the Semantic VAE

Earlier project notes report that the Semantic VAE alone can reconstruct semantics much better than Stage 1 diffusion. If that still holds, then the VAE is not the sole bottleneck.

But it is still a meaningful bottleneck in the current system, because the diffusion model is not solving a pure reconstruction problem. It is solving future semantic forecasting while operating through this compressed latent space and while receiving stale first-frame conditioning.

So the correct interpretation is:

- the semantic VAE is not the only failure point
- but its `8x` compression makes the difficult part of the task substantially harder

### 5.3 The current scaling factor is not well matched to semantic latents

This concern is supported by direct evidence.

From `/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae/latent_statistics_report.txt`:

- unscaled semantic latent std: `9.497363`
- current scaling factor: `0.18215`
- scaled std after applying `0.18215`: `1.729945`
- suggested scaling factor from `1/std`: `0.105292`

This means the current semantic latent scaling is not aligned with the unit-variance assumption used by the RGB VAE and the original SVD noise schedule.

The report explicitly concludes:

> semantic latents are under-compressed by the current scaling factor

This matters because the effective signal-to-noise ratio is shifted:

- the signal remains too large relative to the noise schedule
- training is no longer happening in the distribution regime that the SVD backbone expects

This is no longer the old critical decode bug; that unscaling issue has already been fixed in `DualVAEManager.decode_semantic(...)`.

The remaining issue is calibration:

- RGB latent scaling was reused for semantic latents
- measured semantic latent statistics suggest that reuse is suboptimal

### Additional numeric mismatch inside the input tensor

The target latents are scaled before diffusion, but the semantic conditioning branch is assembled directly from semantic latents without an explicit semantic-specific normalization layer.

Even if this was inherited from the original design, it still means the two halves of the 8-channel input are not guaranteed to live in well-matched numeric regimes.

This is likely a secondary issue, but it adds friction to optimization.

### 5.4 The architecture still borrows an RGB-video prior for a dense semantic task

The UNet is loaded directly from the pretrained SVD checkpoint. No semantic-specific input stem is introduced.

So even though the current run fine-tunes the full UNet, the backbone still starts from a prior optimized for:

- RGB latent denoising
- first-frame image conditioning
- natural-video appearance synthesis

Stage 1 instead asks it to solve:

- semantic layout forecasting
- categorical boundary preservation
- small-object retention
- dense, class-balanced prediction

This mismatch is not fatal, but it is real.

The architecture is effectively reusing a model that was designed to propagate visual appearance, then asking it to become a structured semantic predictor.

That prior helps with global scene consistency, but it is not naturally aligned with minority semantic classes or boundary accuracy.

### 5.5 The conditioning bandwidth is too low for genuine motion reasoning

The RGB conditioning is only a single CLIP token from the first frame.

That token is useful for:

- scene type
- rough object presence
- global context

It is not a strong dense motion signal.

Stage 1 does not explicitly use:

- ego-motion
- camera pose
- depth
- optical flow
- multiple RGB frames
- object tracks

This is a problem because future semantics in driving scenes are not determined by appearance alone. They depend heavily on:

- camera motion
- occlusion/disocclusion
- moving agents
- geometry relative to ego-vehicle motion

Without explicit motion cues, the model tends to predict the most persistent semantic configuration compatible with the first frame.

That is exactly the behavior observed in the results.

### 5.6 The SVD micro-conditioning no longer matches the semantic path cleanly

The pipeline still uses SVD-style added-time-ids:

- `fps - 1`
- `motion_bucket_id`
- `noise_aug_strength`

However, in semantic mode the original SVD image-latent conditioning path is bypassed and replaced with semantic latents.

This creates a semantic mismatch:

- the model still receives SVD motion metadata
- but the conditioning modality is no longer the same one that SVD was trained to pair with those embeddings

The most obvious case is `noise_aug_strength`:

- in SVD, this value is tied to the conditioning image corruption level
- in Stage 1 semantic mode, the semantic conditioning latents are not produced from a noised RGB latent path

So the model inherits a micro-conditioning scheme that is no longer fully semantically grounded in the actual conditioning tensor.

This is probably not the primary bottleneck, but it is another sign that the current pipeline is a modality adaptation of SVD rather than a clean semantic forecasting architecture.

### 5.7 The current objective does not strongly protect minority classes

Training is done with sigma-weighted MSE in continuous latent space.

That has two consequences:

1. it optimizes latent similarity, not semantic class balance directly
2. large regions dominate the objective

This helps overall scene structure, but it does not strongly penalize failures on:

- traffic lights
- bicycles
- riders
- trains
- thin structures

The gap between pixel accuracy (`80.13%`) and mIoU (`35.45%`) is exactly what one expects when dominant classes are learned much better than minority classes.

---

## 6. A more precise interpretation of your specific concerns

The concerns you raised are well founded, but they do not all have the same status.

### 6.1 "The scaling factor multiplied to semantic encoder output may not fit for semantic"

This is confirmed by measured latent statistics.

It is no longer just a hypothesis. The current semantic latent distribution is not well matched by `0.18215`.

### 6.2 "Padding and concatenating first-frame conditioning causes the UNet to propagate the first frame forward"

This is strongly supported by both:

- the actual conditioning construction
- the per-class evaluation pattern

It is one of the central architectural biases of the current Stage 1 design.

### 6.3 "The latent bottleneck causes severe information loss"

This is also supported, but it should be stated carefully:

- the bottleneck is real
- the evidence is strongest for thin and small classes
- it is not the only bottleneck

The current failures are best explained by the combination of:

- stale first-frame conditioning
- compressed semantic latents
- weak motion observability
- class-agnostic latent loss

### 6.4 "Maybe a transformer-based diffusion model such as DiT could mitigate this"

Potentially yes, but not by itself.

A transformer-based diffusion backbone can help with:

- long-range global dependencies
- more flexible spatial-temporal reasoning
- better token-level conditioning integration

But a DiT running on the same `24 x 88` semantic latent grid will still inherit the same lost fine detail. If the representation has already discarded thin-object information, changing the denoiser alone cannot recover it.

So the right conclusion is:

- DiT may help
- DiT alone is not the main fix
- representation and conditioning should be fixed first, or at least jointly

---

## 7. Recommended next updates

The most defensible next steps are not all equally expensive. A sensible roadmap is to separate:

- protocol corrections
- low-risk architecture fixes
- medium-cost representation upgrades
- full redesign options

### 7.1 Priority 1: Fix the evaluation protocol and reporting

Before changing the backbone, the task definition should be made explicit.

Recommended reporting split:

1. `frame-0 only` forecasting
   - no GT frame-24 semantic anchor at inference
2. `frame-0 + frame-24 anchor` boundary-conditioned generation
   - current setup
3. metrics on frames `1-23` only
   - excludes the conditioned endpoints
4. full-sequence metrics
   - for comparability with prior runs

This is important for the thesis because the current score can otherwise be misread as pure future semantic prediction from RGB alone.

### 7.2 Priority 2: Calibrate semantic latent scaling properly

This is a relatively low-cost intervention with strong justification.

Recommended options:

1. replace `0.18215` with the measured semantic-specific scaling factor (`~0.1053`)
2. or add a learnable latent normalization/affine layer before diffusion
3. and normalize the conditioning branch explicitly as well

This should be done before claiming that the current backbone has reached its ceiling.

### 7.3 Priority 3: Replace concat-only conditioning with a semantic adapter or multi-scale hint encoder

This is likely the highest-value architecture update.

Instead of injecting semantic information only by raw channel concatenation at the input, introduce a semantic hint encoder that produces features at multiple resolutions, for example:

- `24 x 88`
- `12 x 44`
- `6 x 22`
- optional `3 x 11`

These features can then be injected into the UNet at matching stages, similar in spirit to:

- ControlNet
- T2I-Adapter
- multi-scale conditioning pyramids

Why this matters:

- small-object semantics degrade rapidly if injected only once at the input
- fresh semantic features at deeper resolutions preserve class identity longer
- the model no longer has to infer all semantic structure from a single stale first-frame latent

This is more targeted than a full backbone replacement and directly addresses the observed failure pattern.

### 7.4 Priority 4: Improve the semantic latent representation

If the main bottleneck is semantic compression, then the representation should be changed, not just the denoiser.

The most promising options are:

### Option A: higher-resolution semantic VAE

- move from `8x` to `4x` downsampling
- increase latent channels if needed

Pros:

- preserves more boundary detail
- directly addresses thin-object collapse

Cons:

- substantially higher memory cost

### Option B: hierarchical semantic VAE

Use a two-level representation, e.g.:

- low-resolution global latent
- higher-resolution residual or boundary latent

This is probably more practical than a pure `4x` latent if GPU memory is tight.

### Option C: discrete or vector-quantized semantic tokens

Semantic maps are categorical. A discrete token representation may be more natural than a Gaussian continuous latent for preserving crisp boundaries and rare classes.

This is especially attractive if the long-term plan includes a transformer denoiser.

### 7.5 Priority 5: Increase the conditioning information, not only the model size

A single RGB frame and one CLIP token are weak predictors of future semantics.

If the task definition allows it, larger gains may come from better conditioning rather than from a larger denoiser.

Recommended additions:

- first `K` RGB frames instead of one
- ego-motion or vehicle pose
- depth
- optical flow
- dense RGB features from a segmentation or self-supervised backbone

For driving scenes, ego-motion is especially important. It directly explains many future semantic changes that a static first-frame token cannot capture.

### 7.6 Priority 6: Add an auxiliary semantic-aware loss

This is not the first thing to change, but it is a reasonable follow-up once representation and scaling are cleaned up.

Possible additions:

- decoded cross-entropy loss on semantic logits
- Dice loss
- boundary-aware loss
- class-balanced weighting

The current latent MSE objective is good for coarse consistency, but it does not specifically defend minority classes.

### 7.7 Priority 7: Consider DiT or STDiT only after the above fixes

A transformer-based diffusion model is a reasonable long-term direction, especially for video semantics.

It may help with:

- global geometric consistency
- long-range temporal reasoning
- flexible conditioning fusion

But it should be framed correctly:

- if the latent representation remains `8x` compressed and stale-frame conditioned, DiT will still inherit those limitations
- if the task remains boundary-conditioned with GT frame 24, architectural gains may be confounded by protocol leakage

Therefore, a DiT-style redesign makes the most sense after:

1. protocol cleanup
2. semantic scaling calibration
3. improved conditioning injection
4. improved semantic latent representation

If a large redesign is pursued, the most coherent version would be:

- hierarchical or discrete semantic tokens
- dense RGB and motion conditioning
- factorized spatial-temporal transformer diffusion

rather than simply swapping the current UNet for a DiT while keeping everything else unchanged.

---

## 8. Recommended experiment order

If the goal is to improve Stage 1 efficiently and produce a strong thesis discussion, the following order is the most defensible:

1. Re-run evaluation in true `frame-0 only` mode and report frames `1-23` separately.
2. Retrain or fine-tune with semantic-specific latent scaling (`~0.1053`) or learnable semantic latent normalization.
3. Replace concat-only input conditioning with a multi-scale semantic adapter.
4. Add richer conditioning from RGB, preferably dense features and ego-motion cues.
5. Upgrade the semantic codec to a hierarchical or higher-resolution latent representation.
6. Only then evaluate a transformer-based diffusion backbone such as DiT or STDiT.

If only one architectural update can be tried next, the best candidate is:

> multi-scale semantic conditioning plus semantic-specific latent normalization

because it directly addresses the current failure modes without requiring a full redesign.

---

## 9. Final conclusion

The current Stage 1 model is not failing because it cannot learn anything. It clearly learns coarse scene structure and performs reasonably well on large static classes. The main issue is that the architecture is biased toward semantic persistence rather than balanced future semantic forecasting.

The strongest evidence points to four interacting bottlenecks:

1. repeated first-frame semantic conditioning encourages copy-forward behavior
2. `8x` semantic compression removes fine class detail too early
3. RGB-derived SVD priors are only partially aligned with dense semantic prediction
4. the current protocol uses a frame-24 semantic anchor, so the reported task is easier than pure first-frame-only forecasting

The scaling-factor concern is also valid and now empirically supported.

Therefore, the next Stage 1 update should not start with "use a bigger denoiser" as the only answer. The more principled order is:

- first fix protocol and representation
- then improve conditioning injection
- then consider larger transformer-based diffusion models

In short: the current architecture is good at preserving what is already visible, but weak at generating semantically precise future structure that is small, thin, rare, or dynamically changing.
