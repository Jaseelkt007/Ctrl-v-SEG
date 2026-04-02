# Discussion: Stage 2 Architecture, Current Limitations, and Recommended Next Updates

## Purpose

This note is intended as a discussion of the current Stage 2 pipeline described in `pipeline_stage2.md`, cross-checked against the actual implementation in:

- `src/ctrlv/models/controlnet.py`
- `src/ctrlv/models/unet_spatio_temporal_condition.py`
- `tools/train_video_controlnet.py`
- `src/ctrlv/models/dual_vae_manager.py`
- `scripts/eval_scripts/eval_stage2_rgb.sh`

It also uses the reported experiment results in:

- `/usrhomes/s1492/iss_rp_paper/iss_rp_paper/iss-thesis/chapters/ch4_experiments.tex`
- `/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb/eval_summary.txt`
- `/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb_unet_unfreeze/eval_summary.txt`
- `/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb_unet_unfreeze_reinject/eval_summary.txt`
- `/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb_unet_unfreeze/eval_results.json`
- `/no_backups/s1492/Ctrl-V/outputs/eval_stage2_rgb_unet_unfreeze_reinject/eval_results.json`

The goal is not to argue that the Stage 2 result is "good enough". The goal is to explain why the current metrics take the values they do, what the architecture is actually good at, where the failure modes come from, and which next modifications are most justified for a controllability-focused research thesis.

---

## 1. What the current Stage 2 system is actually doing

Conceptually, Stage 2 is framed as:

> semantic segmentation sequence + first RGB frame -> photorealistic RGB video

That summary is correct, but the actual implementation is more specific:

1. The ground-truth semantic sequence is encoded by the Semantic VAE into a latent tensor of shape `[B, T, 4, 24, 88]` for `192 x 704` input.
2. The first RGB frame is encoded in two different ways:
   - a single CLIP visual embedding for cross-attention
   - an RGB VAE latent that is repeated across all 25 frames
3. The target RGB video is encoded by the RGB VAE, diffused with noise, and fed to the UNet as noisy RGB latents.
4. The repeated first-frame RGB latent is concatenated with the noisy RGB latents, creating the 8-channel input seen by both ControlNet and UNet.
5. The semantic latents are not concatenated into the UNet input directly. Instead, they are processed through the ControlNet branch, whose outputs are added to the UNet encoder skip features and the UNet mid block.
6. The decoder never receives semantic tensors directly. It only receives semantically corrected skip connections and a semantically corrected bottleneck.

This means Stage 2 is not a pure "semantic map to RGB" renderer. It is a pretrained SVD video prior that is being steered by:

- dense semantic ControlNet residuals
- a repeated first-frame RGB appearance latent
- a single global CLIP token from the first frame
- fixed SVD micro-conditioning (`fps`, `motion_bucket_id`, `noise_aug_strength`)

That design already suggests an important inductive bias:

- the architecture is strong at preserving large-scale spatial structure from semantics
- but it is still anchored to the appearance and motion prior of the first RGB frame and the pretrained SVD backbone

In other words, the model is optimized to produce a plausible video that respects the semantic map as much as possible, not a semantically exact video renderer with native temporal control.

---

## 2. Important implementation clarifications versus the markdown description

Several details matter for thesis discussion because the actual code is slightly more specific than some higher-level descriptions.

### 2.1 The actual UNet unfreezing scope is narrower than some prose descriptions suggest

The current best Stage 2 run is usually described as "partial UNet unfreezing". That is correct, but the actual code in `tools/train_video_controlnet.py` only unfreezes:

- `mid_block`
- `conv_norm_out`
- `conv_act`
- `conv_out`

The decoder up-blocks remain frozen.

This matters because the current best result should not be interpreted as "the decoder adapted broadly to semantic control". The adaptation is concentrated in:

- the semantic bottleneck fusion point
- the final output projection layers

That explains why semantic controllability improved sharply after unfreezing, but also why there is still a strong residual bias toward the original SVD decoder behavior.

### 2.2 The actual reinjection scales in code are `12 x 44`, `6 x 22`, and `3 x 11`

In `src/ctrlv/models/controlnet.py`, the semantic latents are re-injected after each downsampling boundary using the current feature-map resolution. At `192 x 704`, the actual reinjection scales are:

- after down block 0: `12 x 44`
- after down block 1: `6 x 22`
- after down block 2: `3 x 11`

So the reinjection mechanism refreshes semantic context at increasingly coarse spatial scales. It does not provide a high-resolution boundary-preserving path into the decoder.

This is important because it means the reinjection design is primarily reinforcing global and mid-level structure, not recovering fine semantic edges.

### 2.3 The Stage 2 semantic fidelity metric is an external recoverability metric

The reported Stage 2 mIoU is not computed directly on generated semantic outputs. It is computed by:

1. generating RGB frames
2. re-segmenting those frames with DRN-D-105
3. comparing DRN predictions to the ground-truth semantic labels

So the Stage 2 semantic score measures:

> how recoverable the intended semantics are from the generated RGB appearance

This is a useful controllability metric, but it is not identical to direct semantic reconstruction accuracy. It includes the behavior of the external DRN segmenter.

### 2.4 The Stage 2 evaluation is still an upper-bound setting

The current Stage 2 evaluation uses ground-truth semantic maps as conditioning input. Therefore the reported Stage 2 semantic fidelity is an upper bound on end-to-end performance.

In the full pipeline:

- Stage 1 prediction errors will enter Stage 2
- Stage 2 must render from predicted, not ground-truth, semantics

So the `39.17%` mIoU of the best Stage 2 model should be discussed as:

> the semantic controllability ceiling of Stage 2 under ideal conditioning

not as the final end-to-end semantic fidelity of the full two-stage system.

### 2.5 The frozen-baseline versus unfreeze comparison is very informative, but not perfectly matched

The frozen UNet baseline in Chapter 4 is evaluated on `150` clips, while the UNet-unfreeze model is evaluated on `487` clips. The gain from `23.20%` to `39.17%` mIoU is large enough that the qualitative conclusion is very likely correct, but academically it is still worth noting that the two evaluations are not matched on sample count.

That means the comparison is strongly suggestive rather than perfectly controlled.

---

## 3. What the current results say

### 3.1 The main quantitative picture

The current reported Stage 2 results are:

| Configuration | Samples | mIoU | Pixel Acc. | Mean Acc. | FW-IoU | FID | FVD-I3D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen UNet baseline | 150 | 23.20% | 67.63% | 37.10% | 52.29% | not matched here | not matched here |
| UNet unfreeze | 487 | **39.17%** | **85.34%** | **50.88%** | **75.31%** | 21.9122 | 255.2064 |
| UNet unfreeze + reinjection | 487 | 35.62% | 83.41% | 46.63% | 72.61% | **20.9585** | **234.3064** |

Three immediate conclusions follow.

First, partial UNet unfreezing is not a small tweak. It is the single most important Stage 2 architectural correction currently implemented. The gain of `+15.97` percentage points in mIoU from the frozen baseline to the UNet-unfreeze model is too large to dismiss as noise.

Second, the reinjection variant creates a real trade-off rather than a universal improvement. It improves FID and FVD, but it reduces all DRN-based semantic fidelity metrics.

Third, Stage 2 has not yet solved semantic controllability even under ideal conditioning. The best model still loses substantial semantic information when semantic maps are translated into RGB video.

### 3.2 The per-class pattern is highly structured

For the best UNet-unfreeze configuration, the per-class IoU pattern is very uneven:

- very strong: road `85.17%`, sky `84.37%`, car `84.29%`, vegetation `80.62%`, building `75.25%`
- moderate: terrain `59.06%`, sidewalk `55.20%`, wall `47.63%`, fence `37.74%`, traffic sign `30.99%`, truck `29.40%`, pole `25.67%`, person `22.92%`
- weak to collapsed: rider `9.88%`, bus `7.34%`, motorcycle `5.17%`, bicycle `3.15%`, train `0.32%`, traffic light `0.00%`

This is not a random pattern. It is exactly the pattern expected from an architecture that is:

- strong on large contiguous regions
- acceptable on medium-size structures
- weak on thin, rare, or boundary-sensitive classes

### 3.3 Clip-level spread is large, which means the model is scene-regime dependent

From the saved `eval_results.json` files:

- UNet-unfreeze clip-level DRN mIoU has mean / median / std of `40.47 / 41.03 / 11.15`, with a range of `0.70` to `70.03`
- reinjection clip-level DRN mIoU has mean / median / std of `36.88 / 37.82 / 9.97`, with a range of `0.98` to `62.89`

These clip-level means are slightly higher than the dataset-level mIoU because they average per clip rather than aggregating a single confusion matrix over all pixels. That distinction itself is useful:

- dataset-level mIoU weights pixels
- mean clip mIoU weights clips

Both views indicate that Stage 2 is highly non-uniform across scenes.

### 3.4 Reinjection helps some clips, but hurts most clips semantically

On the 487 shared validation clips:

- reinjection improves DRN mIoU on only `115` clips
- reinjection worsens DRN mIoU on `372` clips

So the semantic cost of reinjection is not restricted to a small number of outliers. It is the dominant trend.

### 3.5 Failure cases cluster by sequence, which supports the idea of scene-dependent difficulty

Using the saved per-clip outputs:

- the worst 50 reinjection clips are dominated by sequences `0005`, `0006`, and `0007` (`17 + 11 + 10 = 38` clips)
- the worst 50 UNet-unfreeze clips are also dominated by `0005`, `0007`, and `0006`
- the best 50 clips are dominated by sequence `0000`, followed by `0004`, `0009`, and `0002`

This matters because it suggests the failure mode is not uniformly distributed over the validation set. It is tied to scene regime.

That is consistent with your qualitative observation from the saved worst clips:

- sudden turns and junction transitions
- vegetation-dominant or forest-like scenes
- clips with larger viewpoint change and stronger motion

Even without a full motion decomposition, the sequence clustering already supports the claim that Stage 2 is more fragile in temporally harder regimes.

---

## 4. Why mIoU, pixel accuracy, and FW-IoU look the way they do

### 4.1 Why mIoU stays relatively low

mIoU gives every class equal weight. In Stage 2, several classes remain near zero:

- traffic light `0.00%`
- bicycle `3.15%`
- motorcycle `5.17%`
- bus `7.34%`
- rider `9.88%`
- train `0.32%`

Therefore the mean is pulled down strongly even though the dominant classes are rendered well.

This is the correct interpretation of the `39.17%` mIoU:

> the model is good at coarse semantic rendering, but not balanced semantic rendering

### 4.2 Why pixel accuracy is much higher

Pixel accuracy is dominated by large classes:

- road
- building
- vegetation
- sky
- car

These occupy most of the image area. If the generator gets those right, pixel accuracy can be high even when minority classes are missing or merged into nearby regions.

That is why the best Stage 2 model reaches `85.34%` pixel accuracy while mIoU is only `39.17%`.

### 4.3 Why FW-IoU is also high

FW-IoU weights classes by frequency, so it behaves more like pixel accuracy than like mIoU. The reported `75.31%` FW-IoU therefore indicates that the generator is strong on the frequent visual mass of the scene, not that it is uniformly faithful across classes.

### 4.4 Why some clips still have high pixel accuracy but poor semantic controllability

The per-clip reinjection summary contains cases such as:

- `12.91%` mIoU with `84.20%` pixel accuracy
- `19.22%` mIoU with `84.87%` pixel accuracy
- `19.30%` mIoU with `79.75%` pixel accuracy

These are classic signs that the model is preserving dominant background structure while dropping minority classes and boundaries.

### 4.5 Why FID and FVD can improve while mIoU gets worse

FID and FVD reward global realism and temporal smoothness. DRN-based mIoU rewards recoverable semantic structure.

These are related but not identical goals.

If the model:

- smooths object boundaries
- suppresses thin or unstable classes
- prefers visually plausible textures over exact structural adherence

then FID and FVD can improve even while semantic recoverability decreases.

That is exactly what the reinjection results show:

- better distributional realism
- worse semantic control

So the Stage 2 trade-off is not contradictory. It is structurally plausible.

---

## 5. Strengths of the current Stage 2 design

Before focusing on limitations, it is important to state what the architecture already does well.

### 5.1 The semantic conditioning pathway is genuinely effective at large-scale structure transfer

The high IoU of road, sky, vegetation, building, and car shows that the ControlNet conditioning is not superficial. It clearly transfers large-scale scene layout into the RGB output.

### 5.2 Partial UNet unfreezing solved a real architectural bottleneck

The jump from `23.20%` to `39.17%` mIoU strongly suggests that the frozen SVD decoder was previously under-utilizing the conditioning signal. Allowing the bottleneck and output projection to adapt significantly improved the semantic-to-RGB translation.

### 5.3 The backbone still provides strong natural video priors

The generated frames achieve realistic textures, color distributions, and broad scene plausibility. That is one reason the Stage 2 outputs remain visually convincing even when semantic fidelity is imperfect.

So the current problem is not "poor image generation" in a general sense. The problem is:

> semantic controllability is still weaker than visual plausibility

---

## 6. Main architectural limitations

### 6.1 Temporal control is weak because the semantic branch is fundamentally frame-wise at entry

The semantic conditioning enters ControlNet as `[B*T, 4, H, W]` and is first processed by ordinary 2D convolutions. In other words:

- the semantic maps are encoded per frame
- they are flattened across time before the initial control processing
- there is no dedicated temporal semantic encoder before ControlNet fusion

Temporal mixing does exist later inside the shared spatio-temporal blocks, but by that point the semantic control signal has already been injected frame by frame.

This has an important consequence:

> the model is spatially conditioned frame-by-frame, and only later asked to make that conditioning temporally coherent

That is weaker than explicitly modeling temporal control at the conditioning side.

This limitation fits your qualitative observation of failures in:

- sudden turns
- junction transitions
- viewpoint changes
- scenes with rapidly changing visible structure

In such clips, the correct future RGB appearance is not determined only by per-frame semantic layout. It also depends on how the layout evolves through time.

### 6.2 Repeating the first RGB latent across all frames creates appearance inertia

Every generated frame receives the same first-frame RGB latent as conditioning.

That is useful when:

- the scene changes smoothly
- appearance remains stable
- viewpoint drift is small

But it becomes problematic when the camera turns, enters a new road, or reveals new geometry. In those cases the model receives two competing signals:

- semantics say that the future frame should change structurally
- the repeated first-frame RGB latent says that the appearance prior should stay tied to frame 0

This conflict is especially relevant in driving scenes, where ego-motion changes the visible content rapidly.

So one important Stage 2 weakness is not only limited temporal semantics. It is also:

> a persistent first-view appearance anchor that can oppose future semantic change

### 6.3 The decoder is still RGB-prior dominated

Even in the best configuration, semantics do not directly flow into the decoder. The decoder sees:

- ControlNet-corrected skip connections
- a corrected bottleneck

but the decoder weights themselves are still mostly frozen SVD weights.

This means the system still relies heavily on a backbone pretrained for:

- natural RGB video generation
- first-frame image conditioning
- plausible motion synthesis

not for exact semantic faithfulness.

This is useful for realism, but it is a limitation for controllability. When realism and semantic exactness conflict, the architecture is still biased toward realism.

### 6.4 The semantic representation is too compressed for small or thin classes

At `192 x 704` input, the semantic VAE produces `24 x 88` latents. That is already an `8x` spatial reduction. The ControlNet and UNet then go deeper:

- `24 x 88`
- `12 x 44`
- `6 x 22`
- `3 x 11`

At those scales:

- poles
- traffic lights
- bicycles
- thin sidewalk boundaries
- rider-vehicle interactions

are all extremely fragile.

This is the main reason the best Stage 2 model can still render road and building very well while traffic light remains at `0.00%`.

This limitation is shared with Stage 1, but in Stage 2 it appears in a slightly different form:

- the semantic condition may still contain the correct class
- but the RGB renderer loses that class during appearance synthesis because the class occupies too little effective latent support

### 6.5 Multi-scale reinjection currently reinforces coarse layout more than fine structure

The reinjection design adds resized semantic latents back into the ControlNet encoder at coarse scales. In principle, this is a good idea: refresh semantic information instead of injecting it only once.

However, the current implementation has two important characteristics:

1. reinjection occurs only in the encoder, not in a high-resolution decoder-side adapter path
2. the reinjection scales are progressively coarser (`12 x 44`, `6 x 22`, `3 x 11`)

This means the mechanism is particularly well suited to reinforcing:

- global region identity
- broad scene organization
- coarse temporal stability

It is less well suited to preserving:

- class boundaries
- thin vertical structures
- small dynamic objects
- instance-level separation

That is why the reinjection model can improve FID/FVD while decreasing mIoU. It is refreshing semantics in a way that benefits coarse plausibility more than fine semantic precision.

### 6.6 The current objective does not directly protect semantic fidelity

Stage 2 is trained with a latent denoising objective in RGB latent space. That objective does not explicitly optimize:

- semantic recoverability
- class balance
- boundary sharpness
- instance separation

Therefore the optimization pressure is strongest on the visually dominant parts of the scene. If a small object disappears but the global image still looks plausible, the loss may not penalize that error strongly enough.

This is one of the cleanest explanations for the observed metric pattern:

- FID/FVD can improve
- pixel accuracy can stay high
- rare-class IoU can still collapse

### 6.7 The semantic and RGB latent spaces are only numerically aligned, not jointly learned as one space

Applying the RGB VAE scaling factor to semantic latents was an important fix. It likely helped the conditioning magnitude become compatible with the SVD latent regime.

But that does not mean the two latent spaces are genuinely aligned semantically. They still come from:

- a frozen RGB VAE trained for photorealistic video
- a separate semantic VAE trained for segmentation maps

So the network still has to learn a bridge between two different latent manifolds. The strong benefit of partial UNet unfreezing is consistent with this interpretation: the model needed extra flexibility at the bottleneck and output projection to translate semantically corrected features back into the RGB latent manifold.

### 6.8 Motion micro-conditioning is generic rather than clip-specific

The current pipeline uses fixed added-time conditioning values such as:

- `fps = 7`
- `motion_bucket_id = 127`
- `noise_aug_strength`

These are inherited from SVD-style conditioning, but in the current Stage 2 pipeline they are not clip-specific motion descriptors.

That means the model receives:

- strong spatial conditioning from semantic maps
- but no explicit clip-specific control for actual ego-motion magnitude, turn rate, or scene transition

For a driving dataset, this is a real limitation. The hardest Stage 2 cases are exactly the cases where appearance depends strongly on motion:

- entering an intersection
- turning onto a new road
- rapid viewpoint change
- heavy occlusion/disocclusion

### 6.9 The current Stage 2 result is not robust to Stage 1 noise yet

Because Stage 2 is trained and evaluated with ground-truth semantics, the system has not yet been stress-tested under the actual semantic noise distribution produced by Stage 1.

For a controllability thesis, this matters because the current Stage 2 number answers:

> how well can the renderer follow ideal semantic control?

but not yet:

> how robust is the renderer when the control signal itself is imperfect?

That gap should be acknowledged explicitly in the thesis discussion.

---

## 7. A more precise interpretation of the reinjection result

It would be a mistake to interpret the reinjection experiment as:

> multi-scale conditioning does not work

That conclusion is too strong.

What the current result actually supports is more specific:

1. adding coarse multi-scale semantic refresh inside the encoder improves realism-oriented metrics
2. the present reinjection design does not preserve semantic boundaries as well as the plain UNet-unfreeze model
3. therefore the current reinjection mechanism is better at regularizing the video toward a smoother, more natural video manifold than at maximizing semantic recoverability

In other words, the reinjection experiment revealed a design tension:

- if conditioning is injected mainly at coarse internal scales, the model may become more globally stable
- but if there is no matching high-resolution or semantic-aware supervision path, the same mechanism can wash out the exact class evidence needed by DRN

So the result should be framed as:

> evidence for a realism-versus-controllability trade-off in the current implementation

not as a rejection of multi-scale conditioning as a research direction.

---

## 8. Relation to prior controllability design choices

The current Stage 2 architecture inherits several strengths and weaknesses that are well aligned with the broader controllable diffusion literature.

### 8.1 ControlNet-style residual control is conservative and stable

ControlNet is attractive because it preserves a strong pretrained generative backbone while adding zero-initialized control residuals. That is exactly why the current Stage 2 model can generate realistic video without retraining an entire video generator from scratch.

But the same conservatism means that if the pretrained backbone and the new control objective are not perfectly aligned, the backbone prior can remain stronger than the control signal.

### 8.2 Adapter-style high-resolution pathways are relevant here

Architectures such as T2I-Adapter motivate a complementary design principle:

- keep the backbone mostly frozen
- but feed lightweight condition features at shallow or high-resolution stages

That idea is directly relevant to Stage 2 because the present reinjection mechanism mainly strengthens coarse encoder stages. A shallow boundary-preserving adapter could target exactly the classes that are currently disappearing.

### 8.3 Video-specific controllability work motivates explicit temporal condition encoding

Recent controllable video architectures increasingly use explicit spatio-temporal condition encoders rather than relying only on per-frame control injection. This is highly relevant to the Stage 2 failure pattern. The current pipeline has strong spatial control but weak native temporal control, so a spatio-temporal condition encoder is a natural next step.

### 8.4 Stable Video Diffusion remains a strong prior, but it is still an RGB prior

SVD is a good starting point because it provides realistic temporal and appearance priors. However, the current thesis should make clear that SVD was not pretrained for semantic faithfulness. Stage 2 therefore inherits both:

- SVD's realism strength
- SVD's mismatch with dense semantic controllability

---

## 9. Recommended next updates

The most defensible next steps are not all equally expensive. A sensible roadmap is to separate:

- diagnostic improvements
- control-path architecture fixes
- representation upgrades
- larger redesigns

### 9.1 Priority 1: Add native temporal modeling to the semantic conditioning path

This is the most justified architectural change.

Recommended options:

- a 3D convolutional semantic encoder before ControlNet injection
- temporal attention over semantic latent sequences before flattening `B x T`
- a spatio-temporal condition encoder that outputs multi-scale semantic features

Why this matters:

- it directly addresses the current frame-wise conditioning weakness
- it is the cleanest response to the failure cases involving turns, junctions, and viewpoint changes
- it targets controllability rather than only visual quality

### 9.2 Priority 2: Reduce the conflict caused by repeating the first RGB latent across all frames

The current image-latent conditioning is useful, but it is too persistent.

Recommended options:

- decay the first-frame RGB latent influence over time
- inject it only into early layers rather than every frame at equal strength
- use the first `K` RGB frames instead of a single frame
- condition on ego-motion, depth, or optical flow so that the appearance prior can move with the scene

For driving scenes, ego-motion is especially important. It provides the missing link between a static first-frame appearance anchor and the future semantics.

### 9.3 Priority 3: Increase semantic resolution or use a hierarchical semantic representation

The current `8x` semantic compression is too aggressive for thin classes.

Recommended options:

- a `4x` semantic VAE
- a hierarchical semantic VAE with low-resolution global latent plus higher-resolution residual/boundary latent
- a lightweight high-resolution semantic adapter that bypasses the deepest bottleneck

This is likely the most direct way to improve:

- pole
- traffic light
- bicycle
- rider
- sidewalk-road separation

### 9.4 Priority 4: Add semantic-aware supervision on generated RGB

Stage 2 currently optimizes RGB latent denoising only.

Recommended additions:

- frozen-segmenter semantic consistency loss on generated RGB
- boundary-aware loss
- class-balanced auxiliary loss
- feature-level perceptual loss computed in a segmentation network

This would align training more directly with the reported controllability metric.

### 9.5 Priority 5: Use diagnostic evaluation splits, not only aggregate metrics

For thesis discussion, aggregate mIoU alone is not enough.

Recommended reporting splits:

- straight-road vs turning/junction clips
- vegetation-dominant vs urban-structured clips
- low-class-count vs high-class-count clips
- middle frames only vs full 25-frame average
- boundary-specific metrics in addition to mIoU
- sequence-wise breakdowns

This would turn the current qualitative observations into a stronger empirical argument.

### 9.6 Priority 6: Train for robustness to imperfect semantics

If the final goal is a controllable two-stage system, Stage 2 should not be trained only on ideal ground-truth semantics.

Recommended follow-ups:

- train with noisy semantic maps
- train with Stage 1 predictions
- add confidence or uncertainty maps from Stage 1 as extra conditioning

This would reduce the gap between the current upper-bound evaluation and actual end-to-end deployment.

### 9.7 Priority 7: Revisit multi-scale conditioning, but with a better design target

The reinjection idea should not be dropped, but it should be redesigned to support fine structure as well as coarse realism.

More promising variants would be:

- temporal semantic encoder + multi-scale injection
- high-resolution shallow adapter + coarse ControlNet residuals
- decoder-side semantic refinement for boundaries
- semantic-aware auxiliary loss during reinjection training

### 9.8 Priority 8: Consider larger transformer-based redesigns only after the control path is fixed

A DiT/STDiT-style video backbone may eventually help, especially for long-range temporal reasoning.

But if the current issues remain unchanged:

- frame-wise semantic control
- repeated first-frame appearance anchoring
- `8x` semantic compression
- no semantic-aware loss

then a larger backbone alone will not solve the main controllability bottlenecks.

So a transformer redesign is a reasonable later-stage direction, not the first next step.

---

## 10. Recommended experiment order

If the goal is to strengthen the thesis discussion and improve Stage 2 efficiently, the following order is the most defensible:

1. Report the current Stage 2 result explicitly as an upper bound under ground-truth semantic conditioning.
2. Add diagnostic breakdowns for motion-heavy, turn-heavy, and vegetation-dominant clips.
3. Replace frame-wise semantic entry with a native temporal semantic condition encoder.
4. Add semantic-aware auxiliary supervision on generated RGB.
5. Increase semantic resolution or add a hierarchical/high-resolution semantic branch.
6. Re-test multi-scale reinjection after the temporal and high-resolution control path is improved.
7. Only then evaluate larger spatio-temporal transformer backbones.

If only one next architectural update can be justified immediately, the strongest candidate is:

> an explicit temporal semantic conditioner plus higher-resolution semantic features

because it directly targets the Stage 2 failure mode that is currently most visible in both the metrics and the worst clips.

---

## 11. Final conclusion

The current Stage 2 model is not failing because it cannot render realistic video. In fact, realism is one of its strongest properties. The main issue is that the architecture is better at producing plausible RGB video than at preserving semantically precise, temporally consistent control.

The strongest evidence points to five interacting bottlenecks:

1. semantic control enters the network mostly frame by frame rather than through a dedicated temporal control encoder
2. the first RGB frame is repeated as a persistent appearance anchor across the full clip
3. semantic information is injected through encoder residual correction, while most of the decoder remains an RGB-video prior
4. the `8x` semantic compression and deeper bottlenecks erase fine classes and boundaries too early
5. the training objective rewards visual plausibility more directly than semantic recoverability

This is why the current best Stage 2 model reaches:

- high pixel accuracy and FW-IoU
- strong IoU on large classes
- realistic visual outputs

while still remaining limited in:

- rare classes
- thin structures
- boundary precision
- motion-heavy controllability

The reinjection experiment sharpens that conclusion further. It shows that the current architecture can still trade semantic control for realism and temporal smoothness. For a controllability-centered thesis, that is not a minor detail. It is one of the central findings.
