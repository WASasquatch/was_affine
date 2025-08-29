![Affine Logo](banner.png)

# AFFINE
## Adaptive Field Filtering by Intermittent Noise Enhancement

**AFFINE** is a custom ComfyUI node that performs sparse **affine transforms** in latent space to steer diffusion results during multi-phase sampling.  
It works by applying controlled scale and bias (`z' = scale * z + bias`) to masked regions of the latent tensor between sampling passes, such as High-Noise and Low-Noise phases.  
This technique biases the denoiser toward darker or brighter outputs *without collapsing noise structure* of parallel noise during sampling. Making it ideal for taming overly contrasty, 
or bright outputs of lightning loras, and adding back detail where the model heavily collapsed noise.

### Example
Here we use some agressive settings to give a neutral cinematic look without hard contrast.
[![affine example](affine_example.gif)](https://streamable.com/wyc2le)

**Things to take note of:**
- The background rock quality is improved.
- The trees and bursh look more naturally dispersed.
- Roads more realistic
- The foreground detail is *essentially* better preserved.
- The overall contrast is reduced.


---

## ✨ Why AFFINE?

- **Improves Quality** – Can help improve detail of outputs with light loras, or taming overly contrasted/burned outputs with other speed-boosting loras.
- **Works in latent space** and avoids costly VAE decode/encode mid-sample and attempting raster adjustments or noise, which can lead to artifacts.  
- **Sparse control** – apply adjustments intermittently using masks (Perlin, Bayer, Checker, etc.).  
- **Stable video** through temporal modes keep masks consistent or "*noisy*" across frames.  
- **Subtle influence** goes a long way with small shifts (scale 0.95–0.98, bias −0.01), but playing with the mask strength can yield interesting results. 
  - Some models are more sensitive to this than others.
- **Not Noise Injection** – does not inject noise, but instead steers the denoiser toward darker or brighter outputs by applying controlled scale and bias to masked regions of the existing noise.
  - Latent Noise Injection is powerful, but can lead to collapsed noise structure and and color burning.

---

## 🔧 Typical workflow
1. Run an **Advanced KSampler/Custom Sampling** for the high-noise sigmas; stop at your split step.  
2. Add **Latent Affine**. Optionally add **Latent Affine Options** and connect it to `options`.  
3. Resume with the second **Advanced KSampler/Custom Sampling** for the low-noise sigmas 
  - It may be best to retain the same scheduler/samplers and seed, but experimentation is always encouraged.  

---

## ⚙️ Parameters

### Latent Affine

| Parameter | Description | Influence | Neutral Cinematic | Soft Vignette | Detail Recovery | High Contrast Stylized |
|---|---|---|---|---|---|---|
| `scale` | Multiplicative gain where mask=1. | <1 darkens; >1 brightens. Sensitive; start subtle. | 0.95 | 0.97 | 1.03 | 1.08 |
| `bias` | Additive offset where mask=1. | Shifts exposure; complements `scale`. | −0.02 | 0.00 | +0.01 | +0.03 |
| `pattern` | Mask generator. | Defines spatial selection. | `perlin` | `bayer` | `perlin` | `checker` |
| `temporal_mode` | Mask evolution over frames. | `static` = consistent, `per_frame` = varying. | `static` | `static` | `static` | `static` |
| `seed` | Procedural randomness seed. | Fix for repeatability; vary for alternates. | 0 | 0 | 0 | 0 |
| `external_mask` | IMAGE input (optional). | If `pattern='external_mask'` it is the mask. Otherwise, if connected, it gates the generated mask. | — | — | — | — |
| `options` | DICT from Options node. | Enables fine control below. | connected | connected | connected | connected |

### Latent Affine Options

| Option | Description | Influence | Neutral Cinematic | Soft Vignette | Detail Recovery | High Contrast Stylized |
|---|---|---|---|---|---|---|
| `mask_strength` | Scales mask intensity. | Higher → stronger effect. | 1.0 | 0.8 | 0.7 | 1.2 |
| `threshold` | Binarize if >0. | Higher → sparser white areas. | 0.6 | 0.3 | 0.4 | 0.7 |
| `invert_mask` | Invert after threshold/blur. | Swap affected regions. | False | True | False | False |
| `perlin_scale` | Perlin frequency scale. | Larger → finer features. | 64 | — | 64 | — |
| `perlin_octaves` | Number of Perlin octaves. | More → richer multi-scale detail. | 3 | — | 3 | — |
| `perlin_persistence` | Amplitude multiplier. | Higher → more high-frequency contrast. | 0.5 | — | 0.6 | — |
| `perlin_lacunarity` | Frequency multiplier. | Higher → faster detail increase. | 2.0 | — | 2.2 | — |
| `checker_size` | Checker cell size (px). | Larger → bigger tiles. | — | 8 | — | 12 |
| `bayer_size` | Bayer base size. | Larger → larger dither blocks. | — | 8 | — | — |
| `blur_ksize` | Gaussian kernel size (odd). | Larger → softer edges. | 5 | 3 | 5 | 0 |
| `blur_sigma` | Gaussian sigma. | Blur strength (with ksize>1). | 1.0 | 0.5 | 1.0 | 0.0 |
| `clamp` | Enable output clamping. | Prevents extreme values. | False | False | False | False |
| `clamp_min` | Lower clamp bound. | Used if `clamp=True`. | — | — | — | — |
| `clamp_max` | Upper clamp bound. | Used if `clamp=True`. | — | — | — | — |
| `frame_seed_stride` | Seed step per frame. | Used when `temporal_mode='per_frame'`. | 9973 | 9973 | 9973 | 9973 |

---

## 🏞️ Gating with an external mask
You can use an external mask to gate the generated mask. This is useful for applying the affine transform only in certain regions of the image.

- Connect a grayscale mask image to `external_mask`.
- Set `pattern = perlin` (or any non-external pattern).
- The generated mask is multiplied by the external mask, so affine only applies in masked regions.

---

## 📂 Installation

1. Clone the reposity to your `ComfyUI/custom_nodes` directory.
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
