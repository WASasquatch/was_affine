![Affine Logo](banner.png)

# AFFINE
## Adaptive Field Filtering by Intermittent Noise Enhancement

**AFFINE** is a custom ComfyUI node that performs sparse **affine transforms** in latent space to steer diffusion results during multi-phase sampling.  
It works by applying controlled scale and bias (`z' = scale * z + bias`) to masked regions of the latent tensor between sampling passes, such as High-Noise and Low-Noise phases.  
This technique biases the denoiser toward darker or brighter outputs *without collapsing noise structure* of parallel noise during sampling. Making it ideal for taming overly contrasty 
or bright outputs of Lightning LoRAs, and adding back detail where the model heavily collapsed noise.

### Example
Here we use some aggressive settings to give a neutral cinematic look without hard contrast.
[Affine WAN Example](https://streamable.com/wyc2le)

**Things to take note of:**
- The background rock quality is improved.
- The trees and brush look more naturally dispersed.
- Roads are more realistic
- The foreground detail is *essentially* better preserved.
- The overall contrast is reduced.


---

## 🧠 How it works (ELI5)

- **Two-part drawing**: The sampler draws your picture in two passes.
  - First pass = lots of noise (rough form).  
  - Second pass = little noise (clean-up, and details).
- **We make a soft mask**: Choose a pattern like White Noise or Perlin Noise. This mask says where we will gently push or pull.
- **Gently turn two knobs** inside the mask:  
  - `scale` = the “dimmer” knob (multiply).  
  - `bias` = the “brightness” knob (add).  
  We apply `z' = scale * z + bias` only where the mask is white.
- **Why only after the first pass?** Because small nudges early guide the clean-up phase to better detail and tone without breaking the noise structure.
- **External mask is a gate**: If you plug in a mask image, it multiplies with the pattern so the effect only happens where your mask is bright.
- **Simple options vs. pattern options**:  
  - `options` = common controls (how strong the mask is, blur edges, threshold, etc.).  
  - `noise_options` = special controls for your chosen pattern (e.g., Perlin frequency).  
  - If both set the same thing, `noise_options` wins. Think: house rules → room rules → final.

---

## ✨ Why AFFINE?

- **Improves Quality** – Can help improve detail of outputs with light loras, or taming overly contrasted/burned outputs with other speed-boosting loras.
- **Works in latent space** and avoids costly VAE decode/encode mid-sample and attempting raster adjustments or noise, which can lead to artifacts.  
- **Sparse control** – apply adjustments intermittently using masks (Perlin, Bayer, Checker, etc.).  
- **Stable video** through temporal modes keep masks consistent or "*noisy*" across frames.  
- **Subtle influence** goes a long way with small shifts (scale 0.95–0.98, bias −0.01), but playing with the mask strength can yield interesting results. 
  - Some models are more sensitive to this than others.
- **Not Noise Injection** – does not inject noise, but instead steers the denoiser toward darker or brighter outputs by applying controlled scale and bias to masked regions of the existing noise.
  - Latent Noise Injection is powerful, but can lead to collapsed noise structure and color burning.

---

## 🔧 Typical workflow
1. Run an **Advanced KSampler/Custom Sampling** for the high-noise sigmas; stop at your split step.  
2. Add **Latent Affine**. Feed options in one of two ways:  
   - Connect a single full options node (backward-compatible `Latent Affine Super Options`) to `options`.  
   - Or, connect `WASLatentAffineCommonOptions` to `options` and a noise/pattern-specific options node (e.g., `WASPerlinOptions`, `WASPoissonBlueOptions`, etc.) to `noise_options`.  
   - Merge precedence: defaults → `options` → `noise_options`.  
3. Resume with the second **Advanced KSampler/Custom Sampling** for the low-noise sigmas 
  - It may be best to retain the same scheduler/samplers and seed, but experimentation is always encouraged.  

**Note:** The affine transform is applied *after* the high-noise sampling, but *before* the low-noise sampling. This means that the affine transform is applied to the high-noise latent, but the low-noise latent is not affected.
**Note 2:** The effect on certain models may be more or less pronounced. Some models are more sensitive than others. Additionally, the scale may be inverted on certain models; for example, on Flux/Krea you'll want positive scale.

---

## ⚙️ Parameters

### Latent Affine Simple

A streamlined version of the node for quick toning and texturing. It auto-tunes mask parameters from latent size and chosen pattern. Ideal when you only need multiplicative scale and a procedural mask.

Inputs:
- `latent` – input latent
- `scale` – multiplicative gain where mask=1 (e.g., 0.95–0.98 to gently darken)
- `noise_pattern` – mask generator (Perlin, Poisson, Worley, Bayer, etc.)
- `seed` – randomness seed
- `temporal_mode` – `static` (stable across frames) or `per_frame` (varies each frame)
- `frame_seed_stride` – seed step per frame when `per_frame`

Returns:
- `latent` – adjusted latent
- `mask` – the mask used (for visualization/diagnostics)

Notes:
- Noise parameters are auto-chosen for rough/noisy masks; no extra options needed.
- For video, use `per_frame` for lively texture or `static` for stability.

### Latent Affine

<details>
<summary>Show parameters table</summary>


| Parameter | Description | Influence | Neutral Cinematic | Soft Vignette | Detail Recovery | High Contrast Stylized |
|---|---|---|---|---|---|---|
| `scale` | Multiplicative gain where mask=1. | <1 darkens; >1 brightens. Sensitive; start subtle. | 0.95 | 0.97 | 1.03 | 1.08 |
| `bias` | Additive offset where mask=1. | Shifts exposure; complements `scale`. | −0.02 | 0.00 | +0.01 | +0.03 |
| `pattern` | Mask generator. | Defines spatial selection. | `perlin` | `bayer` | `perlin` | `checker` |
| `temporal_mode` | Mask evolution over frames. | `static` = consistent, `per_frame` = varying. | `static` | `static` | `static` | `static` |
| `seed` | Procedural randomness seed. | Fix for repeatability; vary for alternates. | 0 | 0 | 0 | 0 |
| `external_mask` | IMAGE input (optional). | If `pattern='external_mask'` it is the mask. Otherwise, if connected, it gates the generated mask. | — | — | — | — |
| `options` | DICT: base/common/full options. | Use `WASLatentAffineCommonOptions` or full legacy node. | connected | connected | connected | connected |
| `noise_options` | DICT: overlay for pattern-specific params. | E.g., Worley/Perlin/Poisson settings. | optional | optional | optional | optional |

</details>

<details>
<summary>Show Noise Options tables</summary>


#### WASPerlinOptions

Smooth fractal noise (organic blobs to fine detail).

| Option | Description |
|---|---|
| `perlin_scale` | Base feature size in pixels. Larger = smoother patterns. |
| `perlin_octaves` | Number of noise octaves (multi-frequency detail). |
| `perlin_persistence` | Amplitude falloff per octave (lower = less high-frequency energy). |
| `perlin_lacunarity` | Frequency multiplier per octave (e.g., 2.0 doubles frequency each octave). |

#### WASWorleyEdgesOptions

Cellular noise emphasizing edges (crack-like structures).

| Option | Description |
|---|---|
| `worley_points_per_kpx` | Feature point density per 1000 pixels. Higher = finer cells. |
| `worley_metric` | Distance metric, `L2` or `L1`. |
| `worley_edge_sharpness` | Exponent to emphasize edges; higher = crisper edges. |

#### WASPoissonBlueOptions

Blue-noise Poisson-disk distance field (evenly spaced spots/voids).

| Option | Description |
|---|---|
| `poisson_radius_px` | Target Poisson-disk minimum spacing in pixels. |
| `poisson_softness` | Distance field softness for smoother masks. |

#### WASRingNoiseOptions

Narrow annulus of high frequencies (ring in frequency domain).

| Option | Description |
|---|---|
| `ring_center_frac` | Ring center as fraction of Nyquist radius. |
| `ring_bandwidth_frac` | Ring Gaussian bandwidth (thickness). |

#### WASHighpassWhiteOptions

High-pass filtered white noise (emphasize fine details/edges).

| Option | Description |
|---|---|
| `highpass_cutoff_frac` | Butterworth HPF cutoff as fraction of Nyquist. |
| `highpass_order` | Butterworth order (steeper when higher). |

#### WASCrossHatchOptions

Oriented gratings and cross-hatch patterns.

| Option | Description |
|---|---|
| `hatch_freq_cyc_px` | Line frequency in cycles/pixel. Higher = denser lines. |
| `hatch_angle1_deg` | First hatch angle (degrees). |
| `hatch_angle2_deg` | Second cross angle (degrees). |
| `hatch_square` | Use square wave (adds harmonics). |
| `hatch_phase_jitter` | Random phase jitter [0..1]. |
| `hatch_supersample` | Supersampling factor (anti-aliasing). |

#### WASTileLinesOptions

Oriented lines randomized per tile (directional micro-structure).

| Option | Description |
|---|---|
| `tile_line_tile_size` | Tile edge length in pixels. |
| `tile_line_freq_cyc_px` | Line frequency in cycles/pixel per tile. |
| `tile_line_jitter` | Phase/orientation jitter per tile [0..1]. |

#### WASDotScreenOptions

Halftone-style dots with jitter (print-like texture).

| Option | Description |
|---|---|
| `dot_cell_size` | Halftone cell size (px). |
| `dot_jitter_px` | Dot center jitter (px). |
| `dot_fill_ratio` | Approximate fill area per cell (0..1). |

#### WASVelvetOptions

Sparse high-frequency impulses (speckled highlights).

| Option | Description |
|---|---|
| `velvet_taps_per_kpx` | Impulse density: taps per 1000 pixels. |

#### WASGreenNoiseOptions

Band-pass mid-frequency emphasis (between low and high frequencies).

| Option | Description |
|---|---|
| `green_center_frac` | Band-pass center as fraction of Nyquist (mid-frequencies). |
| `green_bandwidth_frac` | Relative bandwidth around the center frequency. |

#### WASBlackNoiseOptions

Sparse narrowband spectrum in rFFT domain (structured frequency bands).

| Option | Description |
|---|---|
| `black_bins_per_kpx` | Active frequency bins per 1000 pixels (rFFT domain). |

#### WASCheckerOptions

Checkerboard tiles (binary grid pattern).

| Option | Description |
|---|---|
| `checker_size` | Checkerboard cell size (px). |

#### WASBayerOptions

Ordered dithering matrix (Bayer) tiling.

| Option | Description |
|---|---|
| `bayer_size` | Bayer matrix base size (px). |

</details>

**Merging:** defaults → `options` → `noise_options`. If `external_mask` is provided, it gates the generated mask.

### Options Nodes (modular)

You can configure `WASLatentAffine` with modular DICT nodes:

- `WASLatentAffineCommonOptions` → connect to `options` (mask strength, threshold, invert, blur, clamp, seed stride, etc.).
- Noise/pattern-specific options (connect to `noise_options`):
  - `WASPerlinOptions`, `WASWorleyEdgesOptions`, `WASPoissonBlueOptions`, `WASRingNoiseOptions`, `WASGreenNoiseOptions`, `WASBlackNoiseOptions`, `WASHighpassWhiteOptions`, `WASCrossHatchOptions`, `WASVelvetOptions`, `WASTileLinesOptions`, `WASDotScreenOptions`, `WASCheckerOptions`, `WASBayerOptions`.
- Backward-compatible: the legacy `Latent Affine Super Options` node still works as a single full DICT into `options`.

<details>
<summary>Show Common Options table</summary>


| Option | Description | Influence | Neutral Cinematic | Soft Vignette | Detail Recovery | High Contrast Stylized |
|---|---|---|---|---|---|---|
| `mask_strength` | Scales mask intensity. | Higher → stronger effect. | 1.0 | 0.8 | 0.7 | 1.2 |
| `threshold` | Binarize if >0. | Higher → sparser white areas. | 0.6 | 0.3 | 0.4 | 0.7 |
| `invert_mask` | Invert after threshold/blur. | Swap affected regions. | False | True | False | False |
| `blur_ksize` | Gaussian kernel size (odd). | Larger → softer edges. | 5 | 3 | 5 | 0 |
| `blur_sigma` | Gaussian sigma. | Blur strength (with ksize>1). | 1.0 | 0.5 | 1.0 | 0.0 |
| `clamp` | Enable output clamping. | Prevents extreme values. | False | False | False | False |
| `clamp_min` | Lower clamp bound. | Used if `clamp=True`. | — | — | — | — |
| `clamp_max` | Upper clamp bound. | Used if `clamp=True`. | — | — | — | — |
| `frame_seed_stride` | Seed step per frame. | Used when `temporal_mode='per_frame'`. | 9973 | 9973 | 9973 | 9973 |

</details>

<details>
<summary>Show Notes on Noise Options</summary>

- Noise-specific parameters live in their respective nodes (e.g., Perlin frequency, Poisson radius, Worley jitter). Use node tooltips for ranges and guidance.
- Connect only the relevant noise node for your chosen `pattern`. Unused keys are ignored.

</details>

<details>
<summary>Pattern → Options Node mapping</summary>

- `perlin` → `WASPerlinOptions`
- `worley_edges` → `WASWorleyEdgesOptions`
- `poisson_blue_mask` → `WASPoissonBlueOptions`
- `ring_noise` → `WASRingNoiseOptions`
- `highpass_white` → `WASHighpassWhiteOptions`
- `cross_hatch` → `WASCrossHatchOptions`
- `tile_oriented_lines` → `WASTileLinesOptions`
- `dot_screen_jitter` → `WASDotScreenOptions`
- `velvet_noise` → `WASVelvetOptions`
- `green_noise` → `WASGreenNoiseOptions`
- `black_noise` → `WASBlackNoiseOptions`
- `checker` → `WASCheckerOptions`
- `bayer` → `WASBayerOptions`
- `white_noise`, `pink_noise`, `brown_noise`/`red_noise`, `blue_noise`, `violet_noise`/`purple_noise` → no dedicated options node; use only `WASLatentAffineCommonOptions`.

</details>

---

## 🏞️ Gating with an external mask
You can use an external mask to gate the generated mask. This is useful for applying the affine transform only in certain regions of the image.

- Connect a grayscale mask image to `external_mask`.
- Set `pattern = perlin` (or any non-external pattern).
- The generated mask is multiplied by the external mask, so affine only applies in masked regions.

---

## 📂 Installation

1. Clone the repository to your `ComfyUI/custom_nodes` directory.
2. Restart ComfyUI via Manager or console.  
3. Bake cookies in celebration.

---

## 🧪 Tips
- Start subtle! Latent space is sensitive: `scale` at `0.95` is already technically strong on some models.  
- For video, prefer `perlin` and maybe add subtle blur. `white_noise` will flicker and require a deflicker node of some sort, though this could be a desired effect like old film grain.  
- Combine with **lower CFG (1–2)** for best results on Lightning LoRAs.  

---

## 📜 License
[MIT](LICENSE) – free to use, modify, and share with attribution.
