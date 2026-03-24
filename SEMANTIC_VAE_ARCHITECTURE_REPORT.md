# Semantic-Native VAE Architecture Report

## Purpose

This document describes the exact architecture implemented in `~/vae_semantic/semantic_vae_native/model.py` for the thesis diagram of the native semantic VAE. The focus is on:

- what the semantic stem does,
- which Stable Diffusion / Stable Video Diffusion VAE layers are bypassed or effectively replaced,
- how the custom semantic modules are connected to the pretrained VAE,
- the tensor shapes at each stage,
- which parts are trainable and which remain frozen.

This report uses the code in `model.py` as the primary source of truth. Where the inherited pretrained VAE structure is referenced, it follows the model name used in your code: `stabilityai/stable-video-diffusion-img2vid-xt`, loaded as `AutoencoderKLTemporalDecoder`.

## 1. High-Level Architecture

Your implemented model is:

```text
Semantic IDs [B, T, H, W]
  -> one-hot encoding [B*T, 19, H, W]
  -> SemanticStem (2D) [B*T, 128, H, W]
  -> pretrained VAE encoder core (frozen, conv_in bypassed)
  -> latent mean z [B*T, 4, H/8, W/8]
  -> pretrained VAE decoder core (frozen, decoded per frame)
  -> feature tap before decoder RGB projection [B*T, 128, H, W]
  -> reshape to clip [B, T, 128, H, W]
  -> SemanticHead (3D) [B, T, 19, H, W]
  -> semantic logits
```

Conceptually, the model is not an RGB VAE anymore. It is a semantic-to-latent-to-semantic network that reuses the internal encoder and decoder of a pretrained VAE, but replaces the semantic entry and semantic exit points.

## 2. Base Pretrained VAE Being Reused

Your code loads:

```python
AutoencoderKLTemporalDecoder.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    subfolder="vae"
)
```

From the official model config, the reused VAE has:

- class: `AutoencoderKLTemporalDecoder`
- `in_channels = 3`
- `out_channels = 3`
- `latent_channels = 4`
- `block_out_channels = [128, 256, 512, 512]`
- `layers_per_block = 2`
- 4 encoder down blocks

So the original pretrained VAE is designed for:

```text
RGB image [B, 3, H, W]
  -> encoder
  -> latent [B, 4, H/8, W/8]
  -> temporal decoder
  -> RGB image [B, 3, H, W]
```

Your architecture modifies only the input and output interfaces. The internal VAE core is kept frozen.

## 3. What the Semantic Stem Is

The semantic stem is the trainable front-end you added before the pretrained VAE encoder:

```python
self.stem = nn.Sequential(
    nn.Conv2d(num_classes, hidden_dim, kernel_size=3, padding=1),
    nn.GroupNorm(8, hidden_dim),
    nn.SiLU(),
    nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1),
)
```

With your default configuration:

- `num_classes = 19`
- `hidden_dim = 64`
- `output_dim = 128`

Therefore the semantic stem is:

```text
Conv2d(19 -> 64, kernel=3, stride=1, padding=1)
-> GroupNorm(8 groups, 64 channels)
-> SiLU
-> Conv2d(64 -> 128, kernel=3, stride=1, padding=1)
```

### Role of the Semantic Stem

The stem converts discrete semantic class information into a dense 128-channel feature map that matches the width expected by the pretrained encoder after its original RGB input projection.

In other words:

- original VAE expects `encoder.conv_in` to map `3 -> 128`,
- your stem directly produces `128` channels from semantic input,
- so the stem acts as the semantic replacement for the original RGB input projection.

### Parameter Count of the Stem

Exact trainable parameters:

- first conv: `64 * 19 * 3 * 3 + 64 = 11,008`
- GroupNorm: `64 + 64 = 128`
- second conv: `128 * 64 * 3 * 3 + 128 = 73,856`
- total stem parameters: `84,992`

## 4. What the Semantic Head Is

The semantic head is the trainable back-end you added after the pretrained VAE decoder features:

```python
self.head = nn.Sequential(
    nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1),
    nn.GroupNorm(8, hidden_dim),
    nn.SiLU(),
    nn.Conv3d(hidden_dim, num_classes, kernel_size=1),
)
```

With your defaults:

- `input_dim = 128`
- `hidden_dim = 64`
- `num_classes = 19`

So the semantic head is:

```text
Conv3d(128 -> 64, kernel=3x3x3, stride=1, padding=1)
-> GroupNorm(8 groups, 64 channels)
-> SiLU
-> Conv3d(64 -> 19, kernel=1x1x1)
```

### Role of the Semantic Head

The head replaces the final RGB prediction path with a semantic prediction path. It takes decoder features at width 128 and maps them directly to 19 semantic logits per frame.

### Parameter Count of the Head

Exact trainable parameters:

- first conv3d: `64 * 128 * 3 * 3 * 3 + 64 = 221,248`
- GroupNorm: `64 + 64 = 128`
- second conv3d: `19 * 64 + 19 = 1,235`
- total head parameters: `222,611`

## 5. Exact Layers Bypassed, Replaced, or Left Unused

This section is the most important one for drawing the architecture.

### 5.1 Encoder Side

The original encoder path is conceptually:

```text
RGB input
-> encoder.conv_in (3 -> 128)
-> encoder.down_blocks
-> encoder.mid_block
-> encoder.conv_norm_out
-> encoder.conv_act
-> encoder.conv_out (to latent parameters, typically 8 channels = mean + logvar)
```

In your implementation:

- `encoder.conv_in` is bypassed.
- the semantic stem is inserted in its place.
- everything after `encoder.conv_in` is reused unchanged and frozen.

So the active encoder path becomes:

```text
one-hot semantics [19 channels]
-> SemanticStem (19 -> 64 -> 128)
-> encoder.down_blocks
-> encoder.mid_block
-> encoder.conv_norm_out
-> encoder.conv_act
-> encoder.conv_out
-> split into mean and logvar
-> use mean as z
```

### Encoder Layer Replacement Summary

- logically replaced: `vae.encoder.conv_in`
- not physically deleted from the VAE object: `vae.encoder.conv_in` still exists in memory
- actually used in forward path: no

### 5.2 Decoder Side

The original decoder path is conceptually:

```text
latent z [4 channels]
-> decoder internal latent-to-feature projection
-> decoder.conv_in
-> decoder.mid_block
-> decoder.up_blocks
-> decoder.conv_norm_out
-> decoder.conv_act
-> decoder.conv_out (128 -> 3 RGB)
-> temporal RGB output processing
```

In your implementation:

- you still call the official `vae.decode(...)`,
- you attach a forward hook on `vae.decoder.conv_act`,
- you capture the 128-channel features immediately before `decoder.conv_out`,
- you ignore the RGB output sample,
- you apply `SemanticHead` to the captured 128-channel features.

So the semantic decoder path is:

```text
z
-> official frozen decoder
-> tap features at decoder.conv_act
-> SemanticHead (128 -> 64 -> 19)
-> semantic logits
```

### Decoder Layer Replacement Summary

- logically replaced: `vae.decoder.conv_out (128 -> 3)` as the semantic output projection
- effectively not used for prediction: the RGB decoder output is discarded
- important nuance: `decoder.conv_out` is still executed internally because `vae.decode()` runs the full decoder; you simply do not use its RGB output

### 5.3 Practical Thesis Diagram Recommendation

For the thesis figure, the cleanest truthful drawing is:

- show the semantic head branching from `decoder.conv_act`,
- mark `decoder.conv_out` and the RGB output branch as unused / bypassed for semantic prediction,
- note in the caption that the implementation captures decoder features with a hook before the RGB projection.

## 6. Exact Forward Connection of Your Stem to the VAE

The forward path in `SemanticVAENative.forward()` is:

### Step 1: Input semantic IDs

Input tensor:

```text
semantic_ids: [B, T, H, W]
```

Each pixel stores a class index in `[0, 18]`. Invalid labels such as `255` are clamped temporarily for one-hot generation and are expected to be handled later by the loss mask.

### Step 2: Flatten clip dimension

The temporal dimension is flattened into the batch:

```text
[B, T, H, W] -> [B*T, H, W]
```

This means the semantic stem and the encoder are applied frame-by-frame, not jointly over time.

### Step 3: One-hot encoding

```text
[B*T, H, W] -> one-hot -> [B*T, H, W, 19] -> permute -> [B*T, 19, H, W]
```

### Step 4: Semantic stem

```text
x_onehot [B*T, 19, H, W]
-> semantic_stem
-> h0 [B*T, 128, H, W]
```

This `h0` is the exact connection point into the pretrained encoder core. In architecture terms, `h0` is injected immediately after the place where the original VAE would have produced 128 channels with `encoder.conv_in`.

### Step 5: Frozen encoder core

Your code:

```python
for down_block in self.vae.encoder.down_blocks:
    h = down_block(h)

h = self.vae.encoder.mid_block(h)
h = self.vae.encoder.conv_norm_out(h)
h = self.vae.encoder.conv_act(h)
latent_params = self.vae.encoder.conv_out(h)
mean, logvar = torch.chunk(latent_params, 2, dim=1)
z = mean
```

So the encoder connection is manual and explicit. You are not calling `vae.encode()` because that would force the original RGB `conv_in`.

### Step 6: Latent representation

The encoder output is:

```text
latent_params: [B*T, 8, H/8, W/8]
```

This is split into:

- mean: `[B*T, 4, H/8, W/8]`
- logvar: `[B*T, 4, H/8, W/8]`

Your model uses:

```text
z = mean
```

So the current implementation is deterministic in the forward pass. It does not sample from the latent distribution inside `forward()`.

## 7. Decoder Connection in Exact Implementation Terms

The decoder helper is:

```python
def _decode_to_semantic_features(self, z_flat):
    captured_features = []

    def hook_fn(module, input, output):
        captured_features.append(output)

    hook = self.vae.decoder.conv_act.register_forward_hook(hook_fn)

    try:
        for i in range(BT):
            z_single = z_flat[i:i+1]
            _ = self.vae.decode(z_single, num_frames=1)

        features_stacked = torch.cat(captured_features, dim=0)
        features = features_stacked.squeeze(2)
    finally:
        hook.remove()
```

This means:

- decoding is done one frame at a time,
- each latent frame is decoded independently,
- `num_frames=1` is used for every decode call,
- the feature tensor is captured at `decoder.conv_act`,
- after capture, the singleton temporal dimension is removed with `squeeze(2)`.

The resulting decoder feature tensor is:

```text
f_dec_flat: [B*T, 128, H, W]
```

Then it is reshaped back to clip format:

```text
f_dec: [B, T, 128, H, W]
```

Finally:

```text
f_dec -> SemanticHead -> logits [B, T, 19, H, W]
```

## 8. Important Temporal Interpretation

This point matters for the thesis diagram and text.

### What is temporal in your implementation?

- `SemanticHead` is truly temporal because it uses `Conv3d` on `[B, C, T, H, W]`.
- the pretrained decoder is called with `num_frames=1` for each frame separately.

Therefore, in the code as written:

- the semantic stem is per-frame,
- the encoder is per-frame,
- the latent is per-frame,
- the decoder call is per-frame,
- the final semantic head is the stage that explicitly mixes information across time in the clip.

So the safest academic description is:

> The pretrained VAE decoder is reused as a frozen per-frame feature generator, while clip-level temporal aggregation for semantic prediction is performed by the trainable 3D semantic head.

That statement is more accurate than saying the full frozen decoder is providing cross-frame temporal smoothing in the current implementation.

## 9. Tensor Shapes Through the Whole Network

For a generic input size divisible by 8:

```text
Input semantic IDs                           [B, T, H, W]
Flatten clip                                [B*T, H, W]
One-hot                                     [B*T, 19, H, W]
SemanticStem conv1                          [B*T, 64, H, W]
SemanticStem conv2                          [B*T, 128, H, W]

Encoder down block 0                        [B*T, 128, H/2, W/2]   or same-width block before downsample internally
Encoder down block 1                        [B*T, 256, H/4, W/4]
Encoder down block 2                        [B*T, 512, H/8, W/8]
Encoder down block 3                        [B*T, 512, H/8, W/8]
Encoder mid block                           [B*T, 512, H/8, W/8]
Encoder conv_out                            [B*T, 8,   H/8, W/8]
Latent mean z                               [B*T, 4,   H/8, W/8]

Per-frame decoder feature tap at conv_act   [B*T, 128, H, W]
Reshape to clip                             [B, T, 128, H, W]
SemanticHead conv3d                         [B, T, 64,  H, W]
SemanticHead output                         [B, T, 19,  H, W]
```

For the training configuration in `config_native.yaml`:

- `H = 192`
- `W = 704`
- `T = 4`

So the nominal latent size is:

```text
[B*T, 4, 24, 88]
```

## 10. Trainable vs Frozen Parts

### Trainable

- `semantic_stem`
- `semantic_head`

Exact trainable parameter count:

- stem: `84,992`
- head: `222,611`
- total trainable: `307,603`

### Frozen

Entire pretrained VAE:

```python
self.vae.requires_grad_(False)
self.vae.eval()
```

So:

- no VAE weights are updated,
- gradients still pass through the frozen encoder and decoder activations,
- the stem receives gradients through the encoder,
- the head receives gradients directly from the loss.

## 11. What Was Removed Compared with a Standard VAE Pipeline

Strictly speaking, the layers are not deleted from the `self.vae` object. Instead, they are removed from the active semantic computation path.

### Removed from active forward path

- `vae.encoder.conv_in` is not used
- the RGB image output of `vae.decode(...).sample` is not used as the model output

### Replaced in functional role

- original encoder RGB input projection `3 -> 128`
  is replaced by
  semantic stem `19 -> 64 -> 128`

- original decoder RGB output projection `128 -> 3`
  is replaced by
  semantic head `128 -> 64 -> 19`

### Still present but not semantically central

- `vae.decoder.conv_out` still runs internally during `vae.decode()`
- any final RGB/video output branch after `conv_act` is ignored for semantic prediction

## 12. Clean Thesis Description of the Architecture

You can describe the model in thesis language like this:

> We convert each semantic frame into a one-hot tensor and pass it through a trainable semantic stem composed of two 2D convolutions, producing a 128-channel feature map aligned with the internal width of the pretrained Stable Video Diffusion VAE encoder. This stem replaces the original RGB input projection of the VAE. The resulting features are processed by the frozen encoder core to obtain a 4-channel latent representation at 1/8 spatial resolution. During decoding, each latent frame is passed independently through the frozen pretrained decoder. Instead of using the decoder's RGB output layer, we extract the 128-channel activations immediately before the final RGB projection and feed them to a trainable 3D semantic head, which aggregates information across the clip and predicts 19 semantic logits per frame.

## 13. Recommended Figure to Draw

For the academic architecture figure, draw these blocks:

```text
Semantic ID Clip [B,T,H,W]
-> One-Hot Encoding
-> Semantic Stem
   Conv2d 19->64, k3,s1,p1
   GroupNorm(8)
   SiLU
   Conv2d 64->128, k3,s1,p1
-> Frozen VAE Encoder Core
   down_blocks x4
   mid_block
   conv_norm_out
   conv_act
   conv_out -> 8 channels
   split -> mean/logvar
   choose mean
-> Latent z [4,H/8,W/8]
-> Frozen VAE Decoder Core
   decode each frame independently
   tap features at decoder.conv_act
-> Decoder Features [B,T,128,H,W]
-> Semantic Head
   Conv3d 128->64, k3,s1,p1
   GroupNorm(8)
   SiLU
   Conv3d 64->19, k1
-> Semantic Logits [B,T,19,H,W]
-> Argmax -> Predicted Semantic IDs
```

Also annotate the following two surgery points:

- `encoder.conv_in (3->128)` bypassed and replaced by semantic stem
- `decoder.conv_out (128->3)` bypassed for prediction and replaced by semantic head attached at `decoder.conv_act`

## 14. Source Files Used

Primary local sources:

- `~/vae_semantic/semantic_vae_native/model.py`
- `~/vae_semantic/semantic_vae_native/config_native.yaml`
- `~/vae_semantic/semantic_vae_native/test_model.py`
- `~/vae_semantic/semantic_vae_native/inspect_vae.py`

Supporting official references:

- Stable Video Diffusion VAE config:
  `https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/vae/config.json`
- Diffusers `AutoencoderKLTemporalDecoder` source:
  `https://github.com/huggingface/diffusers`

## 15. Final Bottom-Line Summary

Your semantic-native VAE is best understood as a pretrained video VAE whose RGB entry and RGB exit interfaces have been surgically replaced by semantic interfaces:

- input replacement: `semantic stem` instead of `encoder.conv_in`
- output replacement: `semantic head` instead of using the decoder RGB output
- reused frozen core: encoder body, latent bottleneck, and decoder body
- actual clip-level temporal fusion in your current code: mainly the final `Conv3d` semantic head

That is the correct architecture to draw for the thesis.
