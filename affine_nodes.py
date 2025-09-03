import torch
import torch.nn.functional as F
from comfy.samplers import KSAMPLER
from comfy.sample import sample as comfy_sample

from .utils import (
    gaussian_blur,
    unsharp_mask,
    sobel_magnitude_nchw,
    laplacian_magnitude_nchw,
    local_variance_nchw,
    perlin_noise,
    bayer_matrix,
    pink_noise_2d,
    brown_noise_2d,
    blue_noise_2d,
    violet_noise_2d,
    velvet_noise,
    green_noise_2d,
    black_noise_2d,
    cross_hatch_2d,
    highpass_white_2d,
    ring_noise_2d,
    poisson_blue_mask_2d,
    worley_edges_2d,
    tile_oriented_lines_2d,
    dot_screen_jitter_2d,
    affine_step_schedule,
)


PATTERN_CHOICES = [
    "white_noise",
    "pink_noise",
    "brown_noise",
    "red_noise",
    "blue_noise",
    "violet_noise",
    "purple_noise",
    "green_noise",
    "black_noise",
    "cross_hatch",
    "highpass_white",
    "ring_noise",
    "poisson_blue_mask",
    "worley_edges",
    "tile_oriented_lines",
    "dot_screen_jitter",
    "velvet_noise",
    "perlin",
    "checker",
    "bayer",
    "solid",
    "detail_region",
    "smooth_region",
    "edges_sobel",
    "edges_laplacian",
]

PATTERN_CHOICES_LIST = PATTERN_CHOICES + ["external_mask"]


class WASLatentAffine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Input latent input to apply Affine to."}),
                "scale": ("FLOAT", {"default": 0.96, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Multiplicative factor applied. Examples: 1.0 = no change, <1 darkens, >1 amplifies features. Influence: controls gain in masked regions."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "tooltip": "Additive offset applied. Examples: 0.1 brightens, -0.1 darkens. Influence: shifts values in masked regions."}),
                "pattern": (PATTERN_CHOICES_LIST, {"tooltip": "Mask source. Procedural: white/pink/brown(red)/blue/violet(purple)/green/black (spectrally-shaped); cross_hatch (oriented gratings); highpass_white (Butterworth-shaped); ring_noise (narrow high-freq annulus); poisson_blue_mask (Poisson-disk distance field); worley_edges (cell boundaries); tile_oriented_lines (per-tile gratings); dot_screen_jitter (halftone dots with jitter); velvet (sparse impulses); perlin (smooth noise); checker/bayer (tiled); solid (constant alpha). Content-aware (from latent): detail_region (high texture/variance), smooth_region (low detail), edges_sobel, edges_laplacian. external_mask: use provided IMAGE directly. If an external mask is connected and pattern != external_mask, it gates the generated mask so the affine/noise only applies where this mask is 1."}),
                "temporal_mode": (["static","per_frame"], {"tooltip": "Static: one mask for all frames. Per-frame: vary mask over time.."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Random seed for procedural masks."}),
            },
            "optional": {
                "external_mask": ("IMAGE", {"tooltip": "External mask image [N,H,W,C]. If pattern='external_mask', this image (after threshold/invert/blur) is used as the mask. Otherwise, if connected, it gates the generated mask so the affine/noise only applies where this mask is 1."}),
                "noise_options": ("DICT", {"tooltip": "DICT for pattern-specific params (e.g., poisson_*, worley_*, tile_line_*, dot_*). Applied AFTER 'options'."}),
                "options": ("DICT", {"tooltip": "Base options DICT. Use for Common Options or the Full Options node. 'noise_options' will override overlapping keys from here."}),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "mask")
    FUNCTION = "apply"
    CATEGORY = "latent/adjust"

    @staticmethod
    def _select_device(pref: str, index: int, fallback_device: torch.device) -> torch.device:
        try:
            if pref == "cpu":
                return torch.device("cpu")
            if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
                if torch.cuda.is_available():
                    idx = max(0, int(index))
                    idx = min(idx, torch.cuda.device_count() - 1)
                    return torch.device(f"cuda:{idx}")
        except Exception:
            pass
        # Fallback to latent's device or CPU
        if isinstance(fallback_device, torch.device):
            return fallback_device

    def _mask_2d(self, h, w, pattern, params, device, dtype, seed):
        if pattern == "white_noise":
            rng = torch.Generator(device=device); rng.manual_seed(seed)
            m = torch.rand((1, 1, h, w), generator=rng, device=device, dtype=dtype)
        elif pattern == "pink_noise":
            m = pink_noise_2d(h, w, seed, device, dtype).view(1, 1, h, w)
        elif pattern in ("brown_noise", "red_noise"):
            m = brown_noise_2d(h, w, seed, device, dtype).view(1, 1, h, w)
        elif pattern == "blue_noise":
            m = blue_noise_2d(h, w, seed, device, dtype).view(1, 1, h, w)
        elif pattern in ("violet_noise", "purple_noise"):
            m = violet_noise_2d(h, w, seed, device, dtype).view(1, 1, h, w)
        elif pattern == "green_noise":
            cf = float(params.get("green_center_frac", 0.35))
            bw = float(params.get("green_bandwidth_frac", 0.15))
            m = green_noise_2d(h, w, seed, device, dtype, center_frac=cf, bandwidth_frac=bw).view(1, 1, h, w)
        elif pattern == "black_noise":
            density = max(1, int(params.get("black_bins_per_kpx", 512)))
            # Scale by image size; use rFFT bins approximately ~ h*(w/2+1)
            total_bins = h * (w // 2 + 1)
            k = max(1, int(round(density * (h * w) / 1000.0)))
            k = min(k, max(1, total_bins - 1))
            m = black_noise_2d(h, w, seed, device, dtype, bins=k).view(1, 1, h, w)
        elif pattern == "cross_hatch":
            f = float(params.get("hatch_freq_cyc_px", 0.45))
            a1 = float(params.get("hatch_angle1_deg", 0))
            a2 = float(params.get("hatch_angle2_deg", 90))
            sq = bool(params.get("hatch_square", False))
            pj = float(params.get("hatch_phase_jitter", 0.0))
            ss = int(params.get("hatch_supersample", 1))
            m2 = cross_hatch_2d(h, w, f, (a1, a2), sq, pj, ss, seed, device, dtype)
            m = m2.view(1, 1, h, w)
        elif pattern == "highpass_white":
            cf = float(params.get("highpass_cutoff_frac", 0.7))
            od = int(params.get("highpass_order", 2))
            m2 = highpass_white_2d(h, w, cf, od, seed, device, dtype)
            m = m2.view(1, 1, h, w)
        elif pattern == "ring_noise":
            cfr = float(params.get("ring_center_frac", 0.9))
            bw = float(params.get("ring_bandwidth_frac", 0.05))
            m2 = ring_noise_2d(h, w, cfr, bw, seed, device, dtype)
            m = m2.view(1, 1, h, w)
        elif pattern == "poisson_blue_mask":
            rpx = float(params.get("poisson_radius_px", 8.0))
            soft = float(params.get("poisson_softness", 6.0))
            m2 = poisson_blue_mask_2d(h, w, rpx, soft, seed, device, dtype)
            m = m2.view(1, 1, h, w)
        elif pattern == "worley_edges":
            ppk = float(params.get("worley_points_per_kpx", 2.0))
            met = str(params.get("worley_metric", "L2"))
            es = float(params.get("worley_edge_sharpness", 1.0))
            m2 = worley_edges_2d(h, w, ppk, met, es, seed, device, dtype)
            m = m2.view(1, 1, h, w)
        elif pattern == "tile_oriented_lines":
            ts = int(params.get("tile_line_tile_size", 32))
            ff = float(params.get("tile_line_freq_cyc_px", 0.4))
            jt = float(params.get("tile_line_jitter", 0.25))
            m2 = tile_oriented_lines_2d(h, w, ts, ff, jt, seed, device, dtype)
            m = m2.view(1, 1, h, w)
        elif pattern == "dot_screen_jitter":
            cs = int(params.get("dot_cell_size", 12))
            jp = float(params.get("dot_jitter_px", 1.5))
            fr = float(params.get("dot_fill_ratio", 0.3))
            m2 = dot_screen_jitter_2d(h, w, cs, jp, fr, seed, device, dtype)
            m = m2.view(1, 1, h, w)
        elif pattern == "velvet_noise":
            density = max(1, int(params.get("velvet_taps_per_kpx", 10)))
            taps = max(1, int(round(density * (h * w) / 1000.0)))
            taps = min(taps, h * w)
            m = velvet_noise(h, w, taps, seed, device, dtype).view(1, 1, h, w)
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
        elif pattern == "solid":
            a = float(params.get("solid_alpha", 1.0))
            a = max(0.0, min(1.0, a))
            m = torch.full((1, 1, h, w), fill_value=a, device=device, dtype=dtype)
        else:
            raise ValueError("pattern not handled here")
        # Optional latent noise sharpening before threshold/blur
        try:
            if bool(params.get("sharpen_enable", False)):
                sig = float(params.get("sharpen_sigma", 0.0))
                amt = float(params.get("sharpen_amount", 0.0))
                thr = float(params.get("sharpen_threshold", 0.0))
                if sig > 0.0 and amt != 0.0:
                    m = unsharp_mask(m, sigma=sig, amount=amt, threshold=thr)
        except Exception:
            # Fail-safe: ignore sharpening errors
            pass
        return m

    def _content_mask_from_latent(self, x4d: torch.Tensor, pattern: str, params) -> torch.Tensor:
        n, c, h, w = x4d.shape
        if pattern == "edges_sobel":
            m = sobel_magnitude_nchw(x4d)
        elif pattern == "edges_laplacian":
            m = laplacian_magnitude_nchw(x4d)
        elif pattern == "detail_region":
            sob = sobel_magnitude_nchw(x4d)   # [N,1,H,W]
            kw = int(params.get("content_window", 7))
            kw = max(3, kw | 1)  # enforce odd >= 3
            var = local_variance_nchw(x4d, ksize=kw)  # [N,1,H,W]
            m = (sob + var) * 0.5
        elif pattern == "smooth_region":
            sob = sobel_magnitude_nchw(x4d)
            kw = int(params.get("content_window", 7))
            kw = max(3, kw | 1)
            var = local_variance_nchw(x4d, ksize=kw)
            m = 1.0 - ((sob + var) * 0.5)
            m = m.clamp(0.0, 1.0)
        else:
            raise ValueError("_content_mask_from_latent: unsupported pattern")

        # Optional sharpening before threshold/blur
        try:
            if bool(params.get("sharpen_enable", False)):
                sig = float(params.get("sharpen_sigma", 0.0))
                amt = float(params.get("sharpen_amount", 0.0))
                thr = float(params.get("sharpen_threshold", 0.0))
                if sig > 0.0 and amt != 0.0:
                    m = unsharp_mask(m, sigma=sig, amount=amt, threshold=thr)
        except Exception:
            pass
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
        noise_options=None,
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
            "velvet_taps_per_kpx": 10,
            "green_center_frac": 0.35,
            "green_bandwidth_frac": 0.15,
            "black_bins_per_kpx": 512,
            "hatch_freq_cyc_px": 0.45,
            "hatch_angle1_deg": 0,
            "hatch_angle2_deg": 90,
            "hatch_square": False,
            "hatch_phase_jitter": 0.0,
            "hatch_supersample": 1,
            "highpass_cutoff_frac": 0.7,
            "highpass_order": 2,
            "ring_center_frac": 0.9,
            "ring_bandwidth_frac": 0.05,
            "poisson_radius_px": 8.0,
            "poisson_softness": 6.0,
            "worley_points_per_kpx": 2.0,
            "worley_metric": "L2",
            "worley_edge_sharpness": 1.0,
            "tile_line_tile_size": 32,
            "tile_line_freq_cyc_px": 0.4,
            "tile_line_jitter": 0.25,
            "dot_cell_size": 12,
            "dot_jitter_px": 1.5,
            "dot_fill_ratio": 0.3,
            "content_window": 7,
            "solid_alpha": 1.0,
            "blur_ksize": 0,
            "blur_sigma": 0.0,
            "threshold": 0.0,
            "invert_mask": False,
            "mask_strength": 1.0,
            "sharpen_enable": False,
            "sharpen_sigma": 0.8,
            "sharpen_amount": 0.3,
            "sharpen_threshold": 0.0,
            "clamp": False,
            "clamp_min": -10.0,
            "clamp_max": 10.0,
            "frame_seed_stride": 9973,
            "compute_device": "cuda" if torch.cuda.is_available() else "cpu",
            "device_index": 0,
        }
        
        if isinstance(options, dict):
            opts.update(options)
        if isinstance(noise_options, dict):
            opts.update(noise_options)

        params = {
            "perlin_scale": float(opts["perlin_scale"]),
            "perlin_octaves": int(opts["perlin_octaves"]),
            "perlin_persistence": float(opts["perlin_persistence"]),
            "perlin_lacunarity": float(opts["perlin_lacunarity"]),
            "checker_size": int(opts["checker_size"]),
            "bayer_size": int(opts["bayer_size"]),
            # Convert density to absolute taps for current size on demand in _mask_2d
            "velvet_taps_per_kpx": int(opts["velvet_taps_per_kpx"]),
            "green_center_frac": float(opts["green_center_frac"]),
            "green_bandwidth_frac": float(opts["green_bandwidth_frac"]),
            "black_bins_per_kpx": int(opts["black_bins_per_kpx"]),
            "hatch_freq_cyc_px": float(opts["hatch_freq_cyc_px"]),
            "hatch_angle1_deg": float(opts["hatch_angle1_deg"]),
            "hatch_angle2_deg": float(opts["hatch_angle2_deg"]),
            "hatch_square": bool(opts["hatch_square"]),
            "hatch_phase_jitter": float(opts["hatch_phase_jitter"]),
            "hatch_supersample": int(opts["hatch_supersample"]),
            "highpass_cutoff_frac": float(opts["highpass_cutoff_frac"]),
            "highpass_order": int(opts["highpass_order"]),
            "ring_center_frac": float(opts["ring_center_frac"]),
            "ring_bandwidth_frac": float(opts["ring_bandwidth_frac"]),
            "poisson_radius_px": float(opts["poisson_radius_px"]),
            "poisson_softness": float(opts["poisson_softness"]),
            "worley_points_per_kpx": float(opts["worley_points_per_kpx"]),
            "worley_metric": str(opts["worley_metric"]),
            "worley_edge_sharpness": float(opts["worley_edge_sharpness"]),
            "tile_line_tile_size": int(opts["tile_line_tile_size"]),
            "tile_line_freq_cyc_px": float(opts["tile_line_freq_cyc_px"]),
            "tile_line_jitter": float(opts["tile_line_jitter"]),
            "dot_cell_size": int(opts["dot_cell_size"]),
            "dot_jitter_px": float(opts["dot_jitter_px"]),
            "dot_fill_ratio": float(opts["dot_fill_ratio"]),
            "content_window": int(opts["content_window"]),
            "solid_alpha": float(opts["solid_alpha"]),
            "blur_ksize": int(opts["blur_ksize"]),
            "blur_sigma": float(opts["blur_sigma"]),
            "threshold": float(opts["threshold"]),
            "invert_mask": bool(opts["invert_mask"]),
            "mask_strength": float(opts["mask_strength"]),
            "sharpen_enable": bool(opts["sharpen_enable"]),
            "sharpen_sigma": float(opts["sharpen_sigma"]),
            "sharpen_amount": float(opts["sharpen_amount"]),
            "sharpen_threshold": float(opts["sharpen_threshold"]),
        }

        gen_device = self._select_device(str(opts.get("compute_device", "auto")), int(opts.get("device_index", 0)), device)

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
                if pattern in ("detail_region", "smooth_region", "edges_sobel", "edges_laplacian"):
                    # Compute directly from latent on its device
                    m = self._content_mask_from_latent(x, pattern, params)
                    m = self._apply_threshold_blur(m, params)
                else:
                    m = self._make_mask_4d(n, h, w, pattern, params, gen_device, dtype, seed)
                # Move mask back to latent device for arithmetic
                if m.device != device:
                    m = m.to(device=device)
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
            mask_img = m.squeeze(1).clamp(0.0, 1.0).to(dtype)  # [N,H,W]
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
                if pattern in ("detail_region", "smooth_region", "edges_sobel", "edges_laplacian"):
                    # Compute per-frame from latent on its device using a batch reshape
                    xf = x.permute(0, 2, 1, 3, 4).contiguous().view(n * f, c, h, w)
                    mf = self._content_mask_from_latent(xf, pattern, params)    # [N*F,1,H,W]
                    mf = self._apply_threshold_blur(mf, params)
                    m = mf.view(n, f, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()  # [N,1,F,H,W]
                else:
                    m = self._make_mask_5d(n, f, h, w, pattern, params, gen_device, dtype, seed, temporal_mode, int(opts["frame_seed_stride"]))
                # Move mask back to latent device
                if m.device != device:
                    m = m.to(device=device)
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
            mask_img = m.squeeze(1).contiguous().view(n * f, h, w).clamp(0.0, 1.0).to(dtype)  # [N*F,H,W]
            return (out, mask_img)

        else:
            raise ValueError("latent['samples'] must be 4D [N,C,H,W] or 5D [N,C,F,H,W]")


class WASLatentAffineSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Input latent to apply Affine to."}),
                "scale": ("FLOAT", {"default": 0.96, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Multiplicative factor. <1 darkens; >1 brightens."}),
                "noise_pattern": (PATTERN_CHOICES, {"tooltip": "Pattern used to generate the mask. Auto-tuned for rough/noisy results."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Random seed for mask generation."}),
                "temporal_mode": (["static", "per_frame"], {"default": "static", "tooltip": "static: same mask for all frames; per_frame: re-generate per frame for lively/noisy motion."}),
                "frame_seed_stride": ("INT", {"default": 9973, "min": 1, "max": 100000, "tooltip": "Seed increment per frame when temporal_mode is per_frame."}),
            }
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "mask")
    FUNCTION = "apply"
    CATEGORY = "latent/adjust"

    def _auto_params(self, h, w, pattern):
        smin = min(h, w)
        area = h * w
        params = {}

        if pattern == "perlin":
            params.update({
                "perlin_scale": max(12.0, smin / 16.0),
                "perlin_octaves": 4,
                "perlin_persistence": 0.55,
                "perlin_lacunarity": 2.2,
            })
        elif pattern == "poisson_blue_mask":
            rpx = max(5.0, smin / 30.0)
            params.update({
                "poisson_radius_px": float(rpx),
                "poisson_softness": float(max(2.0, rpx * 0.8)),
            })
        elif pattern == "worley_edges":
            params.update({
                "worley_points_per_kpx": 3.0,
                "worley_metric": "L2",
                "worley_edge_sharpness": 1.2,
            })
        elif pattern == "black_noise":
            params.update({
                "black_bins_per_kpx": 768,
            })
        elif pattern == "green_noise":
            params.update({
                "green_center_frac": 0.38,
                "green_bandwidth_frac": 0.18,
            })
        elif pattern == "cross_hatch":
            params.update({
                "hatch_freq_cyc_px": 0.6,
                "hatch_angle1_deg": 0,
                "hatch_angle2_deg": 90,
                "hatch_square": False,
                "hatch_phase_jitter": 0.1,
                "hatch_supersample": 1,
            })
        elif pattern == "highpass_white":
            params.update({
                "highpass_cutoff_frac": 0.75,
                "highpass_order": 2,
            })
        elif pattern == "ring_noise":
            params.update({
                "ring_center_frac": 0.85,
                "ring_bandwidth_frac": 0.06,
            })
        elif pattern == "tile_oriented_lines":
            ts = int(max(16, smin // 40))
            params.update({
                "tile_line_tile_size": ts,
                "tile_line_freq_cyc_px": 0.55,
                "tile_line_jitter": 0.3,
            })
        elif pattern == "dot_screen_jitter":
            cs = int(max(6, smin // 56))
            params.update({
                "dot_cell_size": cs,
                "dot_jitter_px": 1.6,
                "dot_fill_ratio": 0.32,
            })
        elif pattern == "velvet_noise":
            params.update({
                "velvet_taps_per_kpx": 14,
            })
        elif pattern == "checker":
            params.update({
                "checker_size": int(max(4, smin // 36)),
            })
        elif pattern == "bayer":
            params.update({
                "bayer_size": 2,
            })
        return params

    def apply(self, latent, scale, noise_pattern, seed, temporal_mode, frame_seed_stride):
        x = latent["samples"]
        device, dtype = x.device, x.dtype

        base_params = {
            "threshold": 0.0,
            "invert_mask": False,
            "blur_ksize": 0,
            "blur_sigma": 0.0,
            "mask_strength": 1.0,
            "sharpen_enable": False,
            "sharpen_sigma": 0.8,
            "sharpen_amount": 0.3,
            "sharpen_threshold": 0.0,
        }

        aff = WASLatentAffine()

        s = torch.as_tensor(scale, dtype=dtype, device=device)

        if x.ndim == 4:
            n, c, h, w = x.shape
            params = {**base_params, **self._auto_params(h, w, noise_pattern)}
            m = aff._make_mask_4d(n, h, w, noise_pattern, params, device, dtype, seed)
            s_map = (1.0 - m) + m * s.view(1, 1, 1, 1)
            y = x * s_map
            out = {"samples": y}
            for k, v in latent.items():
                if k != "samples":
                    out[k] = v
            mask_img = m.squeeze(1).clamp(0.0, 1.0).to(dtype)
            return (out, mask_img)

        elif x.ndim == 5:
            n, c, f, h, w = x.shape
            params = {**base_params, **self._auto_params(h, w, noise_pattern)}
            mode = str(temporal_mode)
            stride = int(frame_seed_stride)
            m = aff._make_mask_5d(n, f, h, w, noise_pattern, params, device, dtype, seed, mode, stride)
            s_map = (1.0 - m) + m * s.view(1, 1, 1, 1, 1)
            y = x * s_map
            out = {"samples": y}
            for k, v in latent.items():
                if k != "samples":
                    out[k] = v
            mask_img = m.squeeze(1).contiguous().view(n * f, h, w).clamp(0.0, 1.0).to(dtype)
            return (out, mask_img)

        else:
            raise ValueError("latent['samples'] must be 4D [N,C,H,W] or 5D [N,C,F,H,W]")


class WASAffineKSamplerAdvanced:
    CATEGORY = "sampling/ksampler"
    RETURN_TYPES = ("LATENT", "MASK")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        import comfy
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model to sample with."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "Input latent to continue sampling from."}),
                "add_noise": ("BOOLEAN", {"default": True, "tooltip": "Add initial noise at the first step (common for text-to-image)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Random seed for the sampler."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1, "tooltip": "Number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Fraction of noise to remove (lower = stronger preserve)."}),
                "affine_interval": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Interval in steps to apply affine (1 = every step; 2 = every 2 steps, etc.). Does not change total steps."}),
                "max_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Upper bound on multiplicative affine strength applied at schedule peak."}),
                "max_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "tooltip": "Upper bound on additive bias applied at schedule peak."}),
                "pattern": (PATTERN_CHOICES_LIST, {"default": "white_noise", "tooltip": "Mask/noise pattern used when applying affine between sampling steps."}),
                "affine_seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Seed for affine mask generation (separate from sampler seed)."}),
                "affine_seed_increment": ("BOOLEAN", {"default": False, "tooltip": "If enabled, increment affine seed for each group application (temporal masks)."}),
                "affine_schedule": ("DICT", {"tooltip": "Use WASAffineScheduleOptions (interpreted over total steps)."}),
            },
            "optional": {
                "external_mask": ("IMAGE", {"tooltip": "Optional external mask image; when provided and pattern != external_mask, it gates where affine applies."}),
                "options": ("DICT", {"tooltip": "Base options DICT for affine (e.g., common or full options)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides that layer onto 'options'."}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "First step index (inclusive) to run within the sampler."}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "step": 1, "tooltip": "End step index (exclusive) within the sampler."}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False, "tooltip": "If true, force last step to full denoise behavior."}),
                "merge_inactive_steps": ("BOOLEAN", {"default": True, "tooltip": "Greedily merge steps outside the active schedule window into larger batches."}),
            },
        }

    @classmethod
    def sample(
        cls,
        model,
        positive,
        negative,
        latent_image,
        add_noise,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        affine_interval,
        max_scale,
        max_bias,
        pattern,
        affine_seed,
        affine_seed_increment,
        affine_schedule,
        external_mask=None,
        options=None,
        noise_options=None,
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise=False,
        merge_inactive_steps=True,
    ):
        import inspect as _inspect
        import torch as _torch

        from nodes import KSamplerAdvanced as _KSamplerAdvanced
        ks = _KSamplerAdvanced()
        ks_sample = getattr(ks, "sample", None)
        if not callable(ks_sample):
            raise RuntimeError("KSamplerAdvanced.sample not callable")
        sig = _inspect.signature(ks_sample)
        accepted = {p.name for p in sig.parameters.values()}

        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            raise ValueError("latent_image must be a LATENT dict with 'samples'")
        cur = latent_image["samples"]
        if not isinstance(cur, _torch.Tensor) or cur.dim() != 4:
            raise ValueError("LATENT['samples'] must be 4D tensor [B,C,H,W]")

        sched = affine_schedule or {}
        start = float(sched.get("start", 0.2))
        end = float(sched.get("end", 0.8))
        bias = float(sched.get("bias", 0.5))
        exponent = float(sched.get("exponent", 1.0))
        start_offset = float(sched.get("start_offset", 0.0))
        end_offset = float(sched.get("end_offset", 0.0))

        dd = affine_step_schedule(int(steps), {
            "start": start,
            "end": end,
            "bias": bias,
            "exponent": exponent,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "curve": sched.get("curve", "easeInOutSine"),
            "back_k": sched.get("back_k", 1.70158),
            "bezier": sched.get("bezier"),
        })

        aff = WASLatentAffine()

        s0 = max(0, int(start_at_step))
        s1 = min(10000 if int(end_at_step) <= 0 else int(end_at_step), int(steps))
        s1 = max(s0, s1)

        try:
            import comfy.utils as _cu
            _total_steps = max(0, s1 - s0)
            _pbar = _cu.ProgressBar(_total_steps)
        except Exception:
            _pbar = None

        total_steps_done = 0
        interval = max(1, int(affine_interval))
        affine_applications = 0
        i = s0
        mask_img = None  # will hold the last applied mask if/when affine is applied
        eps = 1e-8
        while i < s1:
            boundary_idx = min((((i // interval) + 1) * interval) - 1, s1 - 1)
            t_boundary = float(dd[min(boundary_idx, len(dd) - 1)]) if len(dd) > 0 else 0.0

            if i < boundary_idx:
                if merge_inactive_steps and (t_boundary <= eps):
                    start_at = i
                    batch_end_idx = boundary_idx
                    # Greedily extend across future inactive boundaries
                    while True:
                        next_start = batch_end_idx + 1
                        if next_start >= s1:
                            break
                        next_boundary_idx = min((((next_start // interval) + 1) * interval) - 1, s1 - 1)
                        t_next = float(dd[min(next_boundary_idx, len(dd) - 1)]) if len(dd) > 0 else 0.0
                        if t_next <= eps:
                            batch_end_idx = next_boundary_idx
                            continue
                        break
                    end_at = min(batch_end_idx + 1, s1)
                    first_step = (total_steps_done == 0)
                    add_noise_enum = "enable" if (first_step and add_noise) else "disable"
                    available = {
                        "model": model,
                        "positive": positive,
                        "negative": negative,
                        "latent_image": {"samples": cur},
                        "seed": int(seed),
                        "noise_seed": int(seed),
                        "steps": int(steps),
                        "cfg": cfg,
                        "sampler_name": sampler_name,
                        "scheduler": scheduler,
                        "denoise": float(denoise),
                        "add_noise": add_noise_enum,
                        "start_at_step": start_at,
                        "end_at_step": end_at,
                        "return_with_leftover_noise": "enable",
                    }
                    call_kwargs = {k: v for k, v in available.items() if k in accepted and v is not None}
                    if isinstance(options, dict) and options.get("debug"):
                        print(f"[WASAffineKSamplerAdvanced] Greedy batch start={start_at} end={end_at}")
                    out = ks_sample(**call_kwargs)
                    res = None
                    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], dict):
                        res = out[0]
                    elif isinstance(out, dict):
                        res = out
                    if res is None:
                        if isinstance(out, _torch.Tensor):
                            cur = out
                        else:
                            raise RuntimeError("Unexpected output from KSamplerAdvanced.sample (inactive greedy batch)")
                    else:
                        if "samples" in res:
                            cur = res["samples"]
                    done_now = end_at - start_at
                    if _pbar is not None:
                        try:
                            for _ in range(done_now):
                                total_steps_done += 1
                                if hasattr(_pbar, "set_message"):
                                    _pbar.set_message(f"step {min(total_steps_done, int(steps))}/{int(steps)}")
                                if hasattr(_pbar, "update_absolute"):
                                    _pbar.update_absolute(total_steps_done)
                                else:
                                    _pbar.update(1)
                        except Exception:
                            pass
                    else:
                        total_steps_done += done_now
                    i = end_at
                    continue  # skip boundary handling

                # Otherwise, batch up to just before boundary
                start_at = i
                end_at = boundary_idx
                first_step = (total_steps_done == 0)
                add_noise_enum = "enable" if (first_step and add_noise) else "disable"
                available = {
                    "model": model,
                    "positive": positive,
                    "negative": negative,
                    "latent_image": {"samples": cur},
                    "seed": int(seed),
                    "noise_seed": int(seed),
                    "steps": int(steps),
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": float(denoise),
                    "add_noise": add_noise_enum,
                    "start_at_step": start_at,
                    "end_at_step": end_at,
                    "return_with_leftover_noise": "enable",
                }
                call_kwargs = {k: v for k, v in available.items() if k in accepted and v is not None}
                if isinstance(options, dict) and options.get("debug"):
                    print(f"[WASAffineKSamplerAdvanced] batch start={start_at} end={end_at} add_noise={add_noise_enum}")
                out = ks_sample(**call_kwargs)
                res = None
                if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], dict):
                    res = out[0]
                elif isinstance(out, dict):
                    res = out
                if res is None:
                    if isinstance(out, _torch.Tensor):
                        cur = out
                    else:
                        raise RuntimeError("Unexpected output from KSamplerAdvanced.sample (batch)")
                else:
                    if "samples" in res:
                        cur = res["samples"]
                done_now = end_at - start_at
                if _pbar is not None:
                    try:
                        for _ in range(done_now):
                            total_steps_done += 1
                            if hasattr(_pbar, "set_message"):
                                _pbar.set_message(f"step {min(total_steps_done, int(steps))}/{int(steps)}")
                            if hasattr(_pbar, "update_absolute"):
                                _pbar.update_absolute(total_steps_done)
                            else:
                                _pbar.update(1)
                    except Exception:
                        pass
                else:
                    total_steps_done += done_now
                i = end_at  # move to boundary step index

            if i <= boundary_idx and i < s1:
                if (t_boundary <= eps) and ((i + 1) % interval) == 0:
                    first_step = (total_steps_done == 0)
                    add_noise_enum = "enable" if (first_step and add_noise) else "disable"
                    start_at = i
                    end_at = i + 1
                    available = {
                        "model": model,
                        "positive": positive,
                        "negative": negative,
                        "latent_image": {"samples": cur},
                        "seed": int(seed),
                        "noise_seed": int(seed),
                        "steps": int(steps),
                        "cfg": cfg,
                        "sampler_name": sampler_name,
                        "scheduler": scheduler,
                        "denoise": float(denoise),
                        "add_noise": add_noise_enum,
                        "start_at_step": start_at,
                        "end_at_step": end_at,
                        "return_with_leftover_noise": "enable" if end_at < s1 else "disable",
                    }
                    call_kwargs = {k: v for k, v in available.items() if k in accepted and v is not None}
                    if isinstance(options, dict) and options.get("debug"):
                        print(f"[WASAffineKSamplerAdvanced] boundary(batch no-affine) start={start_at} end={end_at}")
                    out = ks_sample(**call_kwargs)
                    res = None
                    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], dict):
                        res = out[0]
                    elif isinstance(out, dict):
                        res = out
                    if res is None:
                        if isinstance(out, _torch.Tensor):
                            cur = out
                        else:
                            raise RuntimeError("Unexpected output from KSamplerAdvanced.sample (boundary no-affine)")
                    else:
                        if "samples" in res:
                            cur = res["samples"]
                    total_steps_done += 1
                    if _pbar is not None:
                        try:
                            if hasattr(_pbar, "set_message"):
                                _pbar.set_message(f"step {min(end_at, int(steps))}/{int(steps)}")
                            if hasattr(_pbar, "update_absolute"):
                                _pbar.update_absolute(total_steps_done)
                            else:
                                _pbar.update(1)
                        except Exception:
                            pass
                    i = end_at
                    continue

                # Apply affine when schedule active at boundary, then single-step call
                _applied_affine = False
                if ((i + 1) % interval) == 0 and (t_boundary > eps):
                    s_val = 1.0 + (float(max_scale) - 1.0) * t_boundary
                    b_val = float(max_bias) * t_boundary
                    lat = {"samples": cur}
                    seed_i = int(affine_seed) + (affine_applications if affine_seed_increment else 0)
                    
                    print(f"[WASAffineKSamplerAdvanced] Applying Affine at step {i} scale={s_val} bias={b_val} seed={seed_i} pattern={pattern} noise={noise_options}")
                    lat2, _mask = aff.apply(
                        lat,
                        s_val,
                        b_val,
                        pattern,
                        "static",
                        seed_i,
                        external_mask=external_mask,
                        noise_options=noise_options,
                        options=options,
                    )
                    cur = lat2["samples"]
                    affine_applications += 1
                    _applied_affine = True
                    mask_img = _mask  # capture the last mask used

                first_step = (total_steps_done == 0)
                add_noise_enum = "enable" if (first_step and add_noise) else "disable"
                start_at = i
                end_at = i + 1
                available = {
                    "model": model,
                    "positive": positive,
                    "negative": negative,
                    "latent_image": {"samples": cur},
                    "seed": int(seed),
                    "noise_seed": int(seed),
                    "steps": int(steps),
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": float(denoise),
                    "add_noise": add_noise_enum,
                    "start_at_step": start_at,
                    "end_at_step": end_at,
                    "return_with_leftover_noise": "enable" if end_at < s1 else "disable",
                }
                call_kwargs = {k: v for k, v in available.items() if k in accepted and v is not None}
                if isinstance(options, dict) and options.get("debug"):
                    print(f"[WASAffineKSamplerAdvanced] step={i} add_noise={add_noise_enum} start={start_at} end={end_at}")
                out = ks_sample(**call_kwargs)
                res = None
                if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], dict):
                    res = out[0]
                elif isinstance(out, dict):
                    res = out
                if res is None:
                    if isinstance(out, _torch.Tensor):
                        cur = out
                    else:
                        raise RuntimeError("Unexpected output from KSamplerAdvanced.sample (boundary)")
                else:
                    if "samples" in res:
                        cur = res["samples"]
                total_steps_done += 1
                if _pbar is not None:
                    try:
                        if hasattr(_pbar, "set_message"):
                            _pbar.set_message(f"step {min(end_at, int(steps))}/{int(steps)}")
                        if hasattr(_pbar, "update_absolute"):
                            _pbar.update_absolute(total_steps_done)
                        else:
                            _pbar.update(1)
                    except Exception:
                        pass
                i = end_at

        # If no mask was applied (schedule inactive), return a zero mask matching the batch/spatial dims
        if mask_img is None:
            try:
                b, c, h, w = cur.shape
                mask_img = _torch.zeros((b, h, w), dtype=cur.dtype, device=cur.device)
            except Exception:
                import torch as _torch  # fallback if _torch not in scope due to errors
                if isinstance(cur, _torch.Tensor) and cur.dim() == 4:
                    b, c, h, w = cur.shape
                    mask_img = _torch.zeros((b, h, w), dtype=cur.dtype, device=cur.device)
                else:
                    mask_img = None
        return ({"samples": cur}, mask_img)


class WASAffineKSampler:
    CATEGORY = "sampling/ksampler"
    RETURN_TYPES = ("LATENT", "MASK")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        import comfy
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model to sample with."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "Input latent to continue sampling from."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Random seed for the sampler."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1, "tooltip": "Number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Fraction of noise to remove (lower = stronger preserve)."}),
                "affine_interval": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Interval in steps to apply affine (1 = every step). Does not change total steps."}),
                "max_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Upper bound on multiplicative affine strength applied at schedule peak."}),
                "max_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "tooltip": "Upper bound on additive bias applied at schedule peak."}),
                "pattern": (PATTERN_CHOICES_LIST, {"default": "white_noise", "tooltip": "Mask/noise pattern used when applying affine between sampling steps."}),
                "affine_seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Seed for affine mask generation (separate from sampler seed)."}),
                "affine_seed_increment": ("BOOLEAN", {"default": False, "tooltip": "If enabled, increment affine seed for each group application (temporal masks)."}),
                "affine_schedule": ("DICT", {"tooltip": "Use WASAffineScheduleOptions (interpreted over total steps)."}),
            },
            "optional": {
                "external_mask": ("IMAGE", {"tooltip": "Optional external mask image; when provided and pattern != external_mask, it gates where affine applies."}),
                "options": ("DICT", {"tooltip": "Base options DICT for affine (e.g., common or full options)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides that layer onto 'options'."}),
                "merge_inactive_steps": ("BOOLEAN", {"default": True, "tooltip": "Greedily merge steps outside the active schedule window into larger batches."}),
            },
        }

    @classmethod
    def sample(
        cls,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        affine_interval,
        max_scale,
        max_bias,
        pattern,
        affine_seed,
        affine_seed_increment,
        affine_schedule,
        external_mask=None,
        options=None,
        noise_options=None,
        merge_inactive_steps=True,
    ):
        return WASAffineKSamplerAdvanced.sample(
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            affine_interval=affine_interval,
            add_noise=True,  # add_noise on first block
            max_scale=max_scale,
            max_bias=max_bias,
            pattern=pattern,
            affine_seed=affine_seed,
            affine_seed_increment=affine_seed_increment,
            affine_schedule=affine_schedule,
            external_mask=external_mask,
            options=options,
            noise_options=noise_options,
            start_at_step=0,
            end_at_step=10000,
            return_with_leftover_noise=False,
            merge_inactive_steps=merge_inactive_steps,
        )


AFFINE_NODE_CLASS_MAPPINGS = {
    "WASLatentAffine": WASLatentAffine,
    "WASLatentAffineSimple": WASLatentAffineSimple,
    "WASAffineKSamplerAdvanced": WASAffineKSamplerAdvanced,
    "WASAffineKSampler": WASAffineKSampler,
}

AFFINE_NODE_DISPLAY_NAME_MAPPINGS = {
    "WASLatentAffine": "Latent Affine",
    "WASLatentAffineSimple": "Latent Affine Simple",
    "WASAffineKSamplerAdvanced": "KSampler Affine Advanced",
    "WASAffineKSampler": "KSampler Affine",
}
