import torch
import torch.nn.functional as F

from .utils import gaussian_kernel1d, gaussian_blur, perlin_noise, bayer_matrix


class WASLatentAffineOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Scales mask intensity before applying scale/bias. Examples: 0.5 = weaker effect, 1.0 = normal, 2.0 = strong. Influence: higher values increase the contribution of the mask to scaling and bias."}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Binarize mask if > 0. Converts mask to 0/1 using this threshold. Examples: 0.5 creates hard separation. Influence: higher threshold produces larger black areas; lower produces larger white areas."}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Invert the mask after threshold/blur. Examples: True flips dark/bright regions. Influence: swaps where scale/bias are applied."}),
                "perlin_scale": ("FLOAT", {"default": 64.0, "min": 4.0, "max": 1024.0, "step": 1.0, "tooltip": "Controls Perlin noise frequency. Examples: 32 = coarse blobs, 128 = fine details. Influence: larger scale gives smaller features (higher frequency)."}),
                "perlin_octaves": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Number of Perlin octaves. Examples: 1 = simple, 4 = richer. Influence: more octaves add multi-scale detail."}),
                "perlin_persistence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier between octaves. Examples: 0.3 = smoother, 0.8 = higher contrast. Influence: higher increases contribution of finer octaves."}),
                "perlin_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Frequency multiplier between octaves. Examples: 2.0 (default), 3.0 = more fine detail. Influence: higher values add finer features faster."}),
                "checker_size": ("INT", {"default": 8, "min": 2, "max": 256, "tooltip": "Checkerboard cell size in pixels. Examples: 8 = small squares, 64 = large squares. Influence: sets tiling size for 'checker' pattern."}),
                "bayer_size": ("INT", {"default": 8, "min": 2, "max": 64, "tooltip": "Bayer matrix base size. Examples: 4, 8, 16. Influence: controls dithering pattern scale for 'bayer'."}),
                "blur_ksize": ("INT", {"default": 0, "min": 0, "max": 51, "tooltip": "Gaussian blur kernel size (odd). Examples: 0/1 = no blur, 9 = soft edges. Influence: larger values smooth mask transitions."}),
                "blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Gaussian blur sigma. Examples: 0 = no blur, 1.5 = moderate. Influence: controls blur strength; used only if > 0 and ksize > 1."}),
                "clamp": ("BOOLEAN", {"default": False, "tooltip": "Clamp output latent values to [min,max]. Examples: True to avoid extreme values. Influence: prevents overflows after scaling/bias."}),
                "clamp_min": ("FLOAT", {"default": -10.0, "min": -100.0, "max": 0.0, "step": 0.1, "tooltip": "Lower clamp bound if clamping is enabled. Examples: -5.0."}),
                "clamp_max": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Upper clamp bound if clamping is enabled. Examples: 5.0."}),
                "frame_seed_stride": ("INT", {"default": 1, "min": 1, "max": 100000, "tooltip": "Seed increment per frame when 'per_frame' temporal mode is used. Examples: 1, 9973 (prime)."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "latent/wan"

    def build(
        self,
        mask_strength,
        threshold,
        invert_mask,
        perlin_scale,
        perlin_octaves,
        perlin_persistence,
        perlin_lacunarity,
        checker_size,
        bayer_size,
        blur_ksize,
        blur_sigma,
        clamp,
        clamp_min,
        clamp_max,
        frame_seed_stride,
    ):
        return ({
            "mask_strength": float(mask_strength),
            "threshold": float(threshold),
            "invert_mask": bool(invert_mask),
            "perlin_scale": float(perlin_scale),
            "perlin_octaves": int(perlin_octaves),
            "perlin_persistence": float(perlin_persistence),
            "perlin_lacunarity": float(perlin_lacunarity),
            "checker_size": int(checker_size),
            "bayer_size": int(bayer_size),
            "blur_ksize": int(blur_ksize),
            "blur_sigma": float(blur_sigma),
            "clamp": bool(clamp),
            "clamp_min": float(clamp_min),
            "clamp_max": float(clamp_max),
            "frame_seed_stride": int(frame_seed_stride),
        },)


class WASLatentAffine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Input latent input to apply Affine to."}),
                "scale": ("FLOAT", {"default": 0.96, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Multiplicative factor applied. Examples: 1.0 = no change, <1 darkens, >1 amplifies features. Influence: controls gain in masked regions."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "tooltip": "Additive offset applied. Examples: 0.1 brightens, -0.1 darkens. Influence: shifts values in masked regions."}),
                "pattern": (["white_noise","perlin","checker","bayer","external_mask"], {"tooltip": "Mask source. white_noise: random; perlin: smooth noise; checker/bayer: tiled patterns; external_mask: use provided IMAGE directly. Note: if an external mask is connected and pattern != external_mask, the external mask will gate the generated mask (restrict effect to masked areas)."}),
                "temporal_mode": (["static","per_frame"], {"tooltip": "Static: one mask for all frames. Per-frame: vary mask over time.."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Random seed for procedural masks."}),
            },
            "optional": {
                "external_mask": ("IMAGE", {"tooltip": "External mask image [N,H,W,C]. If pattern='external_mask', this image (after threshold/invert/blur) is used as the mask. Otherwise, if connected, it gates the generated mask so the affine/noise only applies where this mask is 1."}),
                "options": ("DICT", {"tooltip": "Optional options DICT from 'Latent Affine Options'. Examples: leave empty to use defaults. Influence: fine-tunes mask generation, blur, clamping, and temporal stride."}),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "mask")
    FUNCTION = "apply"
    CATEGORY = "latent/wan"

    def _mask_2d(self, h, w, pattern, params, device, dtype, seed):
        if pattern == "white_noise":
            rng = torch.Generator(device=device); rng.manual_seed(seed)
            m = torch.rand((1, 1, h, w), generator=rng, device=device, dtype=dtype)
        elif pattern == "perlin":
            p = perlin_noise(h, w, params["perlin_scale"], params["perlin_octaves"],
                             params["perlin_persistence"], params["perlin_lacunarity"],
                             seed, device, dtype)
            m = p.view(1, 1, h, w)
        elif pattern == "checker":
            cs = max(2, int(params["checker_size"]))
            yy = torch.arange(h, device=device).view(h, 1)
            xx = torch.arange(w, device=device).view(1, w)
            m = (((yy // cs) + (xx // cs)) % 2).to(dtype).view(1, 1, h, w)
        elif pattern == "bayer":
            bs = max(2, int(params["bayer_size"]))
            b = bayer_matrix(bs, device, dtype).unsqueeze(0).unsqueeze(0)
            rh = h // bs + (1 if h % bs else 0)
            rw = w // bs + (1 if w % bs else 0)
            m = b.repeat(1, 1, rh, rw)[..., :h, :w]
        else:
            raise ValueError("pattern not handled here")
        return m

    def _apply_threshold_blur(self, m, params):
        if params["threshold"] > 0.0:
            m = (m >= params["threshold"]).to(m.dtype)
        if params["invert_mask"]:
            m = 1.0 - m
        if params["blur_ksize"] > 1 and params["blur_sigma"] > 0.0:
            m = gaussian_blur(m, params["blur_ksize"], params["blur_sigma"]).clamp(0.0, 1.0)
        m = (m * params["mask_strength"]).clamp(0.0, 2.0)
        return m

    def _image_to_mask_2d(self, img, h, w, device, dtype, params):
        if img is None:
            raise ValueError("external_mask IMAGE not provided")
        if img.ndim != 4:
            raise ValueError("IMAGE must be [N,H,W,C]")
        x = img.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)  # [N,1,H,W]
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = x.to(device=device, dtype=dtype)
        x = self._apply_threshold_blur(x, params)
        return x

    def _make_mask_4d(self, n, h, w, pattern, params, device, dtype, seed):
        if pattern == "external_mask":
            raise RuntimeError("external_mask handled elsewhere")
        m = self._mask_2d(h, w, pattern, params, device, dtype, seed)
        m = self._apply_threshold_blur(m, params)
        return m.repeat(n, 1, 1, 1)

    def _make_mask_5d(self, n, f, h, w, pattern, params, device, dtype, base_seed, temporal_mode, frame_stride):
        if pattern == "external_mask":
            raise RuntimeError("external_mask handled elsewhere")
        if temporal_mode == "static":
            m2d = self._mask_2d(h, w, pattern, params, device, dtype, base_seed)
            m2d = self._apply_threshold_blur(m2d, params)  # [1,1,H,W]
            m = m2d.unsqueeze(2).repeat(n, 1, f, 1, 1)     # [N,1,F,H,W]
        else:
            ms = []
            for i in range(f):
                seed_i = base_seed + i * frame_stride
                m2d = self._mask_2d(h, w, pattern, params, device, dtype, seed_i)
                m2d = self._apply_threshold_blur(m2d, params)  # [1,1,H,W]
                ms.append(m2d)
            m = torch.stack(ms, dim=2).repeat(n, 1, 1, 1, 1)  # [N,1,F,H,W]
        return m

    def apply(
        self,
        latent,
        scale,
        bias,
        pattern,
        temporal_mode,
        seed,
        external_mask=None,
        options=None,
    ):
        x = latent["samples"]
        device, dtype = x.device, x.dtype

        opts = {
            "perlin_scale": 64.0,
            "perlin_octaves": 3,
            "perlin_persistence": 0.5,
            "perlin_lacunarity": 2.0,
            "checker_size": 8,
            "bayer_size": 8,
            "blur_ksize": 0,
            "blur_sigma": 0.0,
            "threshold": 0.0,
            "invert_mask": False,
            "mask_strength": 1.0,
            "clamp": False,
            "clamp_min": -10.0,
            "clamp_max": 10.0,
            "frame_seed_stride": 9973,
        }
        if isinstance(options, dict):
            opts.update(options)

        params = {
            "perlin_scale": float(opts["perlin_scale"]),
            "perlin_octaves": int(opts["perlin_octaves"]),
            "perlin_persistence": float(opts["perlin_persistence"]),
            "perlin_lacunarity": float(opts["perlin_lacunarity"]),
            "checker_size": int(opts["checker_size"]),
            "bayer_size": int(opts["bayer_size"]),
            "blur_ksize": int(opts["blur_ksize"]),
            "blur_sigma": float(opts["blur_sigma"]),
            "threshold": float(opts["threshold"]),
            "invert_mask": bool(opts["invert_mask"]),
            "mask_strength": float(opts["mask_strength"]),
        }

        s = torch.as_tensor(scale, dtype=dtype, device=device).view(1, 1, 1, 1)
        b = torch.as_tensor(bias, dtype=dtype, device=device).view(1, 1, 1, 1)

        if x.ndim == 4:
            n, c, h, w = x.shape
            if pattern == "external_mask":
                m = self._image_to_mask_2d(external_mask, h, w, device, dtype, params)
                if m.shape[0] == 1 and n > 1:
                    m = m.repeat(n, 1, 1, 1)
                elif m.shape[0] != n:
                    m = m[:1].repeat(n, 1, 1, 1)
            else:
                m = self._make_mask_4d(n, h, w, pattern, params, device, dtype, seed)
                if external_mask is not None:
                    ext = self._image_to_mask_2d(external_mask, h, w, device, dtype, params)
                    if ext.shape[0] == 1 and n > 1:
                        ext = ext.repeat(n, 1, 1, 1)
                    elif ext.shape[0] != n:
                        ext = ext[:1].repeat(n, 1, 1, 1)
                    m = m * ext
            s_map = (1.0 - m) + m * s
            y = x * s_map + b * m
            if opts["clamp"]:
                y = y.clamp(opts["clamp_min"], opts["clamp_max"])
            out = {"samples": y}
            for k, v in latent.items():
                if k != "samples":
                    out[k] = v
            mask_img = m.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(dtype)
            return (out, mask_img)

        elif x.ndim == 5:
            n, c, f, h, w = x.shape
            if pattern == "external_mask":
                if external_mask is None or external_mask.ndim != 4:
                    raise ValueError("external_mask IMAGE must be [M,H,W,C]")
                em = external_mask
                M = em.shape[0]
                if M not in (1, f):
                    em = em[:1]
                    M = 1
                m_list = []
                for i in range(f):
                    idx = 0 if M == 1 else i
                    m2d = em[idx:idx+1].mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)  # [1,1,H,W]
                    m2d = F.interpolate(m2d, size=(h, w), mode="bilinear", align_corners=False).to(device=device, dtype=dtype)
                    m2d = self._apply_threshold_blur(m2d, params)
                    m_list.append(m2d)
                m = torch.stack(m_list, dim=2).repeat(n, 1, 1, 1, 1)  # [N,1,F,H,W]
            else:
                m = self._make_mask_5d(n, f, h, w, pattern, params, device, dtype, seed, temporal_mode, int(opts["frame_seed_stride"]))
                if external_mask is not None:
                    em = external_mask
                    if em.ndim != 4:
                        raise ValueError("external_mask IMAGE must be [M,H,W,C]")
                    M = em.shape[0]
                    if M not in (1, f):
                        em = em[:1]
                        M = 1
                    m_list = []
                    for i in range(f):
                        idx = 0 if M == 1 else i
                        m2d = em[idx:idx+1].mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
                        m2d = F.interpolate(m2d, size=(h, w), mode="bilinear", align_corners=False).to(device=device, dtype=dtype)
                        m2d = self._apply_threshold_blur(m2d, params)
                        m_list.append(m2d)
                    ext = torch.stack(m_list, dim=2).repeat(n, 1, 1, 1, 1)  # [N,1,F,H,W]
                    m = m * ext
            s_map = (1.0 - m) + m * s.view(1, 1, 1, 1, 1)
            y = x * s_map + b.view(1, 1, 1, 1, 1) * m
            if opts["clamp"]:
                y = y.clamp(opts["clamp_min"], opts["clamp_max"])
            out = {"samples": y}
            for k, v in latent.items():
                if k != "samples":
                    out[k] = v
            mask_img = m.permute(0, 2, 3, 4, 1).contiguous().view(n * f, h, w, 1).clamp(0.0, 1.0).to(dtype)
            return (out, mask_img)

        else:
            raise ValueError("latent['samples'] must be 4D [N,C,H,W] or 5D [N,C,F,H,W]")


NODE_CLASS_MAPPINGS = {
    "LatentAffineOptions": WASLatentAffineOptions,
    "LatentMaskedAffine": WASLatentAffine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentAffineOptions": "Latent Affine Options",
    "LatentMaskedAffine": "Latent Affine",
}
