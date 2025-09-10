import torch
import torch.nn.functional as F
from comfy.samplers import KSAMPLER
from comfy.sample import sample as comfy_sample
import comfy.sample as comfy_sample_mod
import comfy

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
    get_cfg_for_step,
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

    def _content_mask_from_latent(self, x4d: torch.Tensor, pattern: str, params) -> torch.Tensor:
        n, c, h, w = x4d.shape
        if pattern == "edges_sobel":
            m = sobel_magnitude_nchw(x4d)
        elif pattern == "edges_laplacian":
            m = laplacian_magnitude_nchw(x4d)
        elif pattern == "detail_region":
            sob = sobel_magnitude_nchw(x4d)
            kw = int(params.get("content_window", 7))
            kw = max(3, kw | 1)
            var = local_variance_nchw(x4d, ksize=kw)
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
        x = img.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
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
            m2d = self._apply_threshold_blur(m2d, params)
            m = m2d.unsqueeze(2).repeat(n, 1, f, 1, 1)
        else:
            ms = []
            for i in range(f):
                seed_i = base_seed + i * frame_stride
                m2d = self._mask_2d(h, w, pattern, params, device, dtype, seed_i)
                m2d = self._apply_threshold_blur(m2d, params)
                ms.append(m2d)
            m = torch.stack(ms, dim=2).repeat(n, 1, 1, 1, 1)
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
                    m = self._content_mask_from_latent(x, pattern, params)
                    m = self._apply_threshold_blur(m, params)
                else:
                    m = self._make_mask_4d(n, h, w, pattern, params, gen_device, dtype, seed)
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
            mask_img = m.squeeze(1).clamp(0.0, 1.0).to(dtype)
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
                    m2d = em[idx:idx+1].mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
                    m2d = F.interpolate(m2d, size=(h, w), mode="bilinear", align_corners=False).to(device=device, dtype=dtype)
                    m2d = self._apply_threshold_blur(m2d, params)
                    m_list.append(m2d)
                m = torch.stack(m_list, dim=2).repeat(n, 1, 1, 1, 1)
            else:
                if pattern in ("detail_region", "smooth_region", "edges_sobel", "edges_laplacian"):
                    xf = x.permute(0, 2, 1, 3, 4).contiguous().view(n * f, c, h, w)
                    mf = self._content_mask_from_latent(xf, pattern, params)
                    mf = self._apply_threshold_blur(mf, params)
                    m = mf.view(n, f, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
                else:
                    m = self._make_mask_5d(n, f, h, w, pattern, params, gen_device, dtype, seed, temporal_mode, int(opts["frame_seed_stride"]))
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
                    ext = torch.stack(m_list, dim=2).repeat(n, 1, 1, 1, 1)
                    m = m * ext
            s_map = (1.0 - m) + m * s.view(1, 1, 1, 1, 1)
            y = x * s_map + b.view(1, 1, 1, 1, 1) * m
            if opts["clamp"]:
                y = y.clamp(opts["clamp_min"], opts["clamp_max"])
            out = {"samples": y}
            for k, v in latent.items():
                if k != "samples":
                    out[k] = v
            mask_img = m.squeeze(1).contiguous().view(n * f, h, w).clamp(0.0, 1.0).to(dtype)
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
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. Can be a single float value or a list of float values for per-step CFG. If list is shorter than total steps, the last value will be repeated for remaining steps."}),
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
                "temporal_mode": (["static", "per_frame"], {"default": "static", "tooltip": "Temporal behavior of the affine mask for video latents. 'static': one mask reused across all frames at each application. 'per_frame': re-generate mask per frame (livelier/noisier motion)."}),
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
        temporal_mode="static",
        segment_mode="late_only",
        late_only_percent=0.2,
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
        if not isinstance(cur, _torch.Tensor) or cur.dim() not in (4, 5):
            raise ValueError("LATENT['samples'] must be 4D [B,C,H,W] or 5D [B,C,F,H,W]")

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
        mask_img = None
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
                    current_cfg = get_cfg_for_step(cfg, start_at, steps)
                    available = {
                        "model": model,
                        "positive": positive,
                        "negative": negative,
                        "latent_image": {"samples": cur},
                        "seed": int(seed),
                        "noise_seed": int(seed),
                        "steps": int(steps),
                        "cfg": current_cfg,
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
                    continue

                start_at = i
                end_at = boundary_idx
                first_step = (total_steps_done == 0)
                add_noise_enum = "enable" if (first_step and add_noise) else "disable"
                current_cfg = get_cfg_for_step(cfg, start_at, steps)
                available = {
                    "model": model,
                    "positive": positive,
                    "negative": negative,
                    "latent_image": {"samples": cur},
                    "seed": int(seed),
                    "noise_seed": int(seed),
                    "steps": int(steps),
                    "cfg": current_cfg,
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
                i = end_at

            if i <= boundary_idx and i < s1:
                if (t_boundary <= eps) and ((i + 1) % interval) == 0:
                    first_step = (total_steps_done == 0)
                    add_noise_enum = "enable" if (first_step and add_noise) else "disable"
                    start_at = i
                    end_at = i + 1
                    current_cfg = get_cfg_for_step(cfg, i, steps)
                    available = {
                        "model": model,
                        "positive": positive,
                        "negative": negative,
                        "latent_image": {"samples": cur},
                        "seed": int(seed),
                        "noise_seed": int(seed),
                        "steps": int(steps),
                        "cfg": current_cfg,
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
                        temporal_mode,
                        seed_i,
                        external_mask=external_mask,
                        noise_options=noise_options,
                        options=options,
                    )
                    cur = lat2["samples"]
                    affine_applications += 1
                    _applied_affine = True
                    mask_img = _mask

                first_step = (total_steps_done == 0)
                add_noise_enum = "enable" if (first_step and add_noise) else "disable"
                start_at = i
                end_at = i + 1
                current_cfg = get_cfg_for_step(cfg, i, steps)
                available = {
                    "model": model,
                    "positive": positive,
                    "negative": negative,
                    "latent_image": {"samples": cur},
                    "seed": int(seed),
                    "noise_seed": int(seed),
                    "steps": int(steps),
                    "cfg": current_cfg,
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

        if mask_img is None:
            try:
                if cur.dim() == 5:
                    b, c, f, h, w = cur.shape
                    mask_img = _torch.zeros((b * f, h, w), dtype=cur.dtype, device=cur.device)
                else:
                    b, c, h, w = cur.shape
                    mask_img = _torch.zeros((b, h, w), dtype=cur.dtype, device=cur.device)
            except Exception:
                import torch as _torch
                if isinstance(cur, _torch.Tensor):
                    if cur.dim() == 5:
                        b, c, f, h, w = cur.shape
                        mask_img = _torch.zeros((b * f, h, w), dtype=cur.dtype, device=cur.device)
                    elif cur.dim() == 4:
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
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. Can be a single float value or a list of float values for per-step CFG. If list is shorter than total steps, the last value will be repeated for remaining steps."}),
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


class WASAffineCustomAdvanced:
    CATEGORY = "sampling/custom_sampling"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        import comfy
        return {
            "required": {
                "noise": ("NOISE", {"tooltip": "Noise generator used at the first step to start from noise when needed."}),
                "guider": ("GUIDER", {"tooltip": "CFG guider to use for denoising."}),
                "sampler": ("SAMPLER", {"tooltip": "Base sampler (e.g., euler, dpmpp_2m)."}),
                "sigmas": ("SIGMAS", {"tooltip": "Sigma schedule defining the trajectory; length = steps+1."}),
                "latent_image": ("LATENT", {"tooltip": "Initial latent input to denoise along the provided sigma schedule."}),
                "affine_interval": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Apply affine every N steps (1 = every step)."}),
                "max_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Scale multiplier at schedule peak: 1 + (max_scale-1)*t."}),
                "max_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "tooltip": "Bias added at schedule peak: max_bias*t."}),
                "pattern": (PATTERN_CHOICES_LIST, {"default": "white_noise", "tooltip": "Mask/noise pattern used by Affine."}),
                "affine_seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Seed for affine mask generation (separate from sampler seed)."}),
                "affine_seed_increment": ("BOOLEAN", {"default": False, "tooltip": "If enabled, increment affine seed after each application (temporal masks)."}),
                "affine_schedule": ("DICT", {"tooltip": "Use WASAffineScheduleOptions; interpreted over total steps (start/end/bias/exponent/curve/etc.)."}),
            },
            "optional": {
                "external_mask": ("IMAGE", {"tooltip": "Optional external mask image to gate affine application."}),
                "options": ("DICT", {"tooltip": "Base options for Affine (common/full options)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides layered onto 'options'."}),
                "temporal_mode": (["static", "per_frame"], {"default": "static", "tooltip": "Temporal behavior of the affine mask when applicable."}),
            },
        }

    @classmethod
    def sample(
        cls,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
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
        temporal_mode="static",
        cfg_list=None,
        **kwargs,
    ):
        import torch as _torch
        import comfy.utils as _cu

        if isinstance(latent_image, dict):
            lat_dict = dict(latent_image)
        else:
            lat_dict = {"samples": latent_image}
        x = lat_dict["samples"]
        #device, dtype = x.device, x.dtype

        try:
            mp = getattr(guider, 'model_patcher', None)
            if mp is not None:
                x_fixed = comfy_sample_mod.fix_empty_latent_channels(mp, x)
                if x_fixed is not x:
                    lat_dict = lat_dict.copy()
                    lat_dict["samples"] = x_fixed
                    x = x_fixed
        except Exception:
            pass

        steps = max(int(sigmas.shape[-1]) - 1, 0)
        try:
            print(f"[WASAffineCustomAdvanced] steps={steps} sigmas_len={int(sigmas.shape[-1])}")
        except Exception:
            pass
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
        }) if steps > 0 else []

        aff = WASLatentAffine()
        interval = max(int(affine_interval), 1)
        eps = 1e-8
        _aff_seed = int(affine_seed)
        _applications = 0
        try:
            _pbar = _cu.ProgressBar(steps)
        except Exception:
            _pbar = None
        debug = isinstance(options, dict)

        noise_mask = lat_dict.get("noise_mask", None)
        empty_noise = _torch.zeros_like(x, device=x.device)
        try:
            gen_noise = noise.generate_noise(lat_dict)
            if isinstance(gen_noise, _torch.Tensor):
                gen_noise = gen_noise.to(device=x.device, dtype=x.dtype)
        except Exception:
            gen_noise = empty_noise
        base_seed = getattr(noise, 'seed', 0)

        boundaries = [0]
        for i in range(steps):
            if ((i + 1) % interval) == 0:
                boundaries.append(i + 1)
        if boundaries[-1] != steps:
            boundaries.append(steps)
        

        cur = x
        _global_noop_affine = (abs(float(max_scale) - 1.0) < 1e-8) and (abs(float(max_bias)) < 1e-8)
        try:
            print(f"[WASAffineCustomAdvanced] max_scale={max_scale}, max_bias={max_bias}, _global_noop_affine={_global_noop_affine}")
            print(f"[WASAffineCustomAdvanced] interval={interval}, pattern={pattern}")
        except Exception:
            pass
        try:
            import latent_preview as _lp
            _x0_output = {}
            _callback = _lp.prepare_callback(getattr(guider, 'model_patcher', None), steps, _x0_output)
        except Exception:
            _callback = None
        try:
            import comfy.utils as _cu2
            _disable_pbar = not _cu2.PROGRESS_BAR_ENABLED
        except Exception:
            _disable_pbar = True
        done_steps = 0
        for seg_idx in range(len(boundaries) - 1):
            s0 = boundaries[seg_idx]
            s1 = boundaries[seg_idx + 1]
            if s1 <= s0:
                continue
            _sig = sigmas[s0:s1 + 1].contiguous()
            
            if seg_idx == 0:
                step_noise = gen_noise
            else:
                step_noise = empty_noise
            seed_use = base_seed
            
            current_guider = guider
            if cfg_list is not None:
                current_cfg = get_cfg_for_step(cfg_list, s0, steps)
                try:
                    original_cfg = getattr(guider, 'cfg', None)
                    if original_cfg is None or abs(float(current_cfg) - float(original_cfg)) > 1e-6:
                        import comfy.samplers
                        model_patcher = getattr(guider, 'model_patcher', None)
                        if model_patcher is not None:
                            current_guider = comfy.samplers.CFGGuider(model_patcher)
                            if hasattr(guider, 'conds'):
                                current_guider.conds = guider.conds
                            else:
                                pos_cond = getattr(guider, 'positive', None)
                                neg_cond = getattr(guider, 'negative', None)
                                if pos_cond is not None and neg_cond is not None:
                                    current_guider.set_conds(pos_cond, neg_cond)
                            current_guider.set_cfg(current_cfg)
                except Exception as e:
                    current_guider = guider
                    try:
                        print(f"[WASAffineCustomAdvanced] Failed to create dynamic CFGGuider: {e}")
                    except Exception:
                        pass
            
            if noise_mask is not None:
                cur = current_guider.sample(step_noise, cur, sampler, _sig, denoise_mask=noise_mask, callback=_callback, disable_pbar=_disable_pbar, seed=seed_use)
            else:
                cur = current_guider.sample(step_noise, cur, sampler, _sig, callback=_callback, disable_pbar=_disable_pbar, seed=seed_use)
            
            i_end = s1 - 1
            if i_end >= 0 and not _global_noop_affine:
                t_end = float(dd[i_end]) if (i_end < len(dd)) else 1.0
                if ((i_end + 1) % interval) == 0 and (t_end > eps):
                    s_val = 1.0 + (float(max_scale) - 1.0) * t_end
                    b_val = float(max_bias) * t_end
                    lat = {"samples": cur}
                    seed_i = _aff_seed + (_applications if bool(affine_seed_increment) else 0)
                    _skip = False
                    try:
                        print(f"[WASAffineCustomAdvanced] Step {i_end+1}: t_end={t_end:.3f}, s_val={s_val:.3f}, b_val={b_val:.3f}")
                    except Exception:
                        pass
                    try:
                        if abs(s_val - 1.0) < 1e-8 and abs(b_val) < 1e-8:
                            _skip = True
                        elif (pattern == "external_mask") and (external_mask is not None):
                            import torch as _t
                            if isinstance(external_mask, _t.Tensor):
                                _skip = (_t.max(external_mask).item() <= 0.0)
                    except Exception:
                        pass
                    if not _skip:
                        try:
                            print(f"[WASAffineCustomAdvanced] Applying affine: pattern={pattern}, seed={seed_i}")
                        except Exception:
                            pass
                        lat2, _mask = aff.apply(
                            lat,
                            s_val,
                            b_val,
                            pattern,
                            temporal_mode,
                            int(seed_i),
                            external_mask=external_mask,
                            noise_options=noise_options,
                            options=options,
                        )
                        cur = lat2["samples"]
                        try:
                            mp = getattr(guider, 'model_patcher', None)
                            if mp is not None:
                                cur = comfy_sample_mod.fix_empty_latent_channels(mp, cur)
                        except Exception:
                            pass
                        _applications += 1
                        try:
                            print(f"[WASAffineCustomAdvanced] Applied affine #{_applications}")
                        except Exception:
                            pass
                    else:
                        try:
                            print(f"[WASAffineCustomAdvanced] Skipped affine: s_val={s_val:.3f}, b_val={b_val:.3f}")
                        except Exception:
                            pass
                else:
                    pass

            done_steps += (s1 - s0)
            if _pbar is not None:
                try:
                    if hasattr(_pbar, "set_message"):
                        _pbar.set_message(f"step {min(done_steps, steps)}/{steps}")
                    if hasattr(_pbar, "update_absolute"):
                        _pbar.update_absolute(min(done_steps, steps))
                    else:
                        _pbar.update(s1 - s0)
                except Exception:
                    pass

        out = lat_dict.copy()
        out["samples"] = cur

        out_denoised = out
        try:
            if isinstance(locals().get('_x0_output', None), dict) and ('x0' in _x0_output):
                mp = getattr(guider, 'model_patcher', None)
                if mp is not None and hasattr(mp.model, 'process_latent_out'):
                    out_denoised = lat_dict.copy()
                    out_denoised["samples"] = mp.model.process_latent_out(_x0_output['x0'].cpu())
            else:
                mp = getattr(guider, 'model_patcher', None)
                if mp is not None and hasattr(mp.model, 'process_latent_out'):
                    out_denoised = lat_dict.copy()
                    out_denoised["samples"] = mp.model.process_latent_out(cur.cpu())
        except Exception:
            pass

        return (out, out_denoised)

class WASAffinePatternNoise:
    CATEGORY = "sampling/custom_sampling/noise"
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"

    _EXCLUDE = set(["solid", "detail_region", "smooth_region", "edges_sobel", "edges_laplacian", "external_mask"])

    @classmethod
    def INPUT_TYPES(cls):
        patterns = [p for p in PATTERN_CHOICES if p not in cls._EXCLUDE]
        return {
            "required": {
                "pattern": (patterns, {"default": "white_noise", "tooltip": "Pattern used to synthesize structured noise (excludes solid and content-derived patterns)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Seed for reproducible noise generation."}),
                "affine_scale": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Pattern amplitude multiplier applied to the base ComfyUI noise. Controls how much the pattern affects the final noise."}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "If enabled, center and scale the generated pattern to zero-mean unit-std before amplitude scaling."}),
                "affine_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "tooltip": "Additive bias applied where the pattern mask is active. Positive values brighten, negative values darken."}),
            },
            "optional": {
                "options": ("DICT", {"tooltip": "Base options DICT for pattern parameters (same keys as Affine patterns)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides layered onto 'options'."}),
            },
        }

    @classmethod
    def get_noise(cls, pattern, seed, affine_scale, normalize, affine_bias, options=None, noise_options=None):
        import types as _types
        import torch as _torch
        import comfy.sample

        class _WASPatternNoise:
            def __init__(self, pattern, seed, affine_scale, normalize, affine_bias, opts):
                self.pattern = pattern
                self.seed = int(seed)
                self.affine_scale = float(affine_scale)
                self.normalize = bool(normalize)
                self.affine_bias = float(affine_bias)
                self.opts = opts or {}

            def generate_noise(self, input_latent):
                lat = input_latent
                if not isinstance(lat, dict) or "samples" not in lat:
                    raise ValueError("generate_noise expects a LATENT dict with 'samples'")
                x = lat["samples"]
                if not isinstance(x, _torch.Tensor) or x.dim() < 4:
                    raise ValueError("LATENT['samples'] must be a 4D/5D tensor")
                
                batch_inds = lat["batch_index"] if "batch_index" in lat else None
                base_noise = comfy.sample.prepare_noise(x, self.seed, batch_inds)
                
                if x.dim() == 5:
                    b, c, f, h, w = x.shape
                    b_eff = b * f
                    device, dtype = x.device, x.dtype
                    
                    if self.affine_scale == 0.0 and self.affine_bias == 0.0:
                        return base_noise.clamp_(-3.0, 3.0)
                    
                    aff = WASLatentAffine()
                    params = {}
                    if isinstance(self.opts, dict):
                        params.update(self.opts)
                    
                    m2d = aff._mask_2d(h, w, self.pattern, params, device, dtype, self.seed)
                    m = m2d.repeat(b_eff, c, 1, 1)
                    
                    if self.normalize:
                        mu = m.mean(dim=(1,2,3), keepdim=True)
                        std = m.std(dim=(1,2,3), keepdim=True).clamp_min(1e-6)
                        m = (m - mu) / std
                    
                    pattern_mask = m * self.affine_scale
                    bias_factor = pattern_mask * self.affine_bias
                    bias_factor = bias_factor.view(b, c, f, h, w)
                    result = base_noise + bias_factor
                    
                    return result.clamp_(-3.0, 3.0)
                    
                else:
                    b, c, h, w = x.shape
                    device, dtype = x.device, x.dtype
                    
                    if self.affine_scale == 0.0 and self.affine_bias == 0.0:
                        return base_noise.clamp_(-3.0, 3.0)
                    
                    aff = WASLatentAffine()
                    params = {}
                    if isinstance(self.opts, dict):
                        params.update(self.opts)
                    
                    m2d = aff._mask_2d(h, w, self.pattern, params, device, dtype, self.seed)
                    m = m2d.repeat(b, c, 1, 1)
                    
                    if self.normalize:
                        mu = m.mean(dim=(1,2,3), keepdim=True)
                        std = m.std(dim=(1,2,3), keepdim=True).clamp_min(1e-6)
                        m = (m - mu) / std
                    
                    pattern_mask = m * self.affine_scale
                    bias_factor = pattern_mask * self.affine_bias
                    result = base_noise + bias_factor
                    
                    return result.clamp_(-3.0, 3.0)

        opts = {}
        if isinstance(options, dict):
            opts.update(options)
        if isinstance(noise_options, dict):
            opts.update(noise_options)
        n = _WASPatternNoise(pattern, seed, affine_scale, normalize, affine_bias, opts)
        return (n,)

AFFINE_NODE_CLASS_MAPPINGS = {
    "WASLatentAffine": WASLatentAffine,
    "WASLatentAffineSimple": WASLatentAffineSimple,
    "WASAffineKSamplerAdvanced": WASAffineKSamplerAdvanced,
    "WASAffineKSampler": WASAffineKSampler,
    "WASAffineCustomAdvanced": WASAffineCustomAdvanced,
    "WASAffinePatternNoise": WASAffinePatternNoise,
}

AFFINE_NODE_DISPLAY_NAME_MAPPINGS = {
    "WASLatentAffine": "Latent Affine",
    "WASLatentAffineSimple": "Latent Affine Simple",
    "WASAffineKSamplerAdvanced": "KSampler Affine Advanced",
    "WASAffineKSampler": "KSampler Affine",
    "WASAffineCustomAdvanced": "Custom Sampler Affine Advanced",
    "WASAffinePatternNoise": "Affine Pattern Noise",
}

# -----------------------
# Ultimate SD Upscaler Affine Ports
# -----------------------

def _usdu_available() -> bool:
    """
    Dirty check for Ultimate SD Upscaler.
    """
    try:
        try:
            from folder_paths import folder_names_and_paths
            from nodes import NODES_CLASS_MAPPINGS as _NODES
            import os
            
            usdu_nodes = ["UltimateSDUpscale", "UltimateSDUpscaleNoUpscale", "UltimateSDUpscaleCustom"]
            if any(node in _NODES for node in usdu_nodes):
                return True
            
            if "custom_nodes" in folder_names_and_paths:
                custom_nodes_data = folder_names_and_paths["custom_nodes"]
                if isinstance(custom_nodes_data, (list, tuple)) and len(custom_nodes_data) > 0:
                    custom_nodes_dirs = custom_nodes_data[0]
                    if not isinstance(custom_nodes_dirs, list):
                        custom_nodes_dirs = [custom_nodes_dirs]
                    usdu_folder_names = [
                        "ComfyUI_UltimateSDUpscale", 
                        "ComfyUI-UltimateSDUpscale", 
                        "comfyui-ultimatesdupscale",
                        "UltimateSDUpscale"
                    ]
                    
                    for custom_nodes_dir in custom_nodes_dirs:
                        if os.path.exists(custom_nodes_dir):
                            try:
                                for item in os.listdir(custom_nodes_dir):
                                    if item in usdu_folder_names:
                                        item_path = os.path.join(custom_nodes_dir, item)
                                        if os.path.isdir(item_path):
                                            return True
                            except (OSError, PermissionError):
                                continue
        except Exception:
            pass
        return True
    except Exception:
        return False


def _build_sigmas_from_model(model, scheduler, steps: int, denoise: float):
    import torch as _torch
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return _torch.FloatTensor([])
        total_steps = int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
    sigmas = sigmas[-(steps + 1):]
    return sigmas


_SEAM_FIX_CHOICES = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]


class WASUltimateCustomAdvancedAffineNoUpscale:
    CATEGORY = "image/upscaling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to process (no internal upscale)."}),
                "model": ("MODEL", {"tooltip": "Diffusion model to sample with."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
                "vae": ("VAE", {"tooltip": "VAE used to encode/decode latent tiles."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Seed for base sampler (noise)."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. Can be a single float value or a list of float values for per-step CFG. If list is shorter than total steps, the last value will be repeated for remaining steps."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Base sampler algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler for sigma schedule."}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise fraction (<=1)."}),
                # Tiling
                "mode_type": (["Linear", "Chess", "None"], {"default": "None", "tooltip": "Tile traversal mode (placeholder for parity)."}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Tile width in pixels."}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Tile height in pixels."}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "Mask blur (px) for tile feathering."}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8, "tooltip": "Tile padding/overlap size (px)."}),
                # Seam fix
                "seam_fix_mode": (_SEAM_FIX_CHOICES, {"default": "None", "tooltip": "Seam fix strategy (initial pass uses overlap+feather)."}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise factor applied in seam-fix steps (placeholder)."}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 8192, "step": 8, "tooltip": "Seam fix width (px)."}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "Seam fix mask blur (px)."}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 8192, "step": 8, "tooltip": "Seam fix padding (px)."}),
                # Misc
                "force_uniform_tiles": ("BOOLEAN", {"default": True, "tooltip": "Force uniform tiles across the grid."}),
                "tiled_decode": ("BOOLEAN", {"default": False, "tooltip": "Decode in tiles (placeholder)."}),
                # Custom sampling inputs
                "noise": ("NOISE", {"tooltip": "World-aligned noise generator for step 0; zeros for subsequent steps."}),
                # Affine controls
                "affine_interval": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Apply affine every N steps (1 = every step)."}),
                "max_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Scale at schedule peak: 1 + (max_scale-1)*t."}),
                "max_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "tooltip": "Bias at schedule peak: max_bias*t."}),
                "pattern": (PATTERN_CHOICES_LIST, {"default": "white_noise", "tooltip": "Affine mask pattern."}),
                "affine_seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": "Seed for affine mask generation."}),
                "affine_seed_increment": ("BOOLEAN", {"default": False, "tooltip": "Increment affine seed after each application (temporal)."}),
                "affine_schedule": ("DICT", {"tooltip": "WASAffineScheduleOptions dict."}),
            },
            "optional": {
                "external_mask": ("IMAGE", {"tooltip": "Optional external mask to gate affine."}),
                "options": ("DICT", {"tooltip": "Base options for Affine (common/full options)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides layered onto 'options'."}),
                "temporal_mode": (["static", "per_frame"], {"default": "static", "tooltip": "Temporal behavior of the affine mask."}),
            },
        }

    @classmethod
    def upscale(
        cls,
        image,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        noise,
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
        temporal_mode="static",
    ):
        try:
            latent = vae.encode(image)
        except Exception as e:
            raise RuntimeError(f"VAE.encode failed: {e}")

        try:
            g = comfy.samplers.CFGGuider(model)
            g.set_conds(positive, negative)
            cfg_value = get_cfg_for_step(cfg, 0, steps)
            g.set_cfg(cfg_value)
            sampler_obj = comfy.samplers.sampler_object(sampler_name)
            sigmas = _build_sigmas_from_model(model, scheduler, int(steps), float(denoise))
        except Exception as e:
            raise RuntimeError(f"Failed to prepare sampler: {e}")

        x_full = latent["samples"] if isinstance(latent, dict) else latent
        try:
            mp = getattr(g, 'model_patcher', None)
            target_device = getattr(mp, 'device', x_full.device) if mp is not None else x_full.device
        except Exception:
            target_device = x_full.device
        if x_full.device != target_device:
            x_full = x_full.to(target_device)
            if isinstance(latent, dict):
                latent = dict(latent)
                latent["samples"] = x_full
        
        if len(x_full.shape) == 5:
            b, cL, fL, hL, wL = x_full.shape
            is_video = True
        elif len(x_full.shape) == 4:
            b, cL, hL, wL = x_full.shape
            is_video = False
        else:
            raise ValueError(f"Unsupported latent shape: {x_full.shape}. Expected 4D or 5D tensor.")
        
        try:
            if len(image.shape) == 5 and is_video:
                _, _, Hpx, Wpx, _ = image.shape
            elif len(image.shape) == 4:
                _, Hpx, Wpx, _ = image.shape
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
            scale_h = max(1.0, float(Hpx) / float(hL))
            scale_w = max(1.0, float(Wpx) / float(wL))
        except Exception:
            scale_h = scale_w = 8.0

        def to_lat(v_px: int, scale: float):
            return max(1, int(round(float(v_px) / scale)))

        twL = to_lat(int(tile_width), scale_w)
        thL = to_lat(int(tile_height), scale_h)
        padL = to_lat(int(tile_padding), (scale_h + scale_w) * 0.5)
        blurL = to_lat(int(mask_blur), (scale_h + scale_w) * 0.5)

        try:
            full_noise = noise.generate_noise(latent)
            if isinstance(full_noise, torch.Tensor):
                full_noise = full_noise.to(device=x_full.device, dtype=x_full.dtype)
            else:
                full_noise = torch.zeros_like(x_full)
        except Exception:
            full_noise = torch.zeros_like(x_full)

        out_acc = torch.zeros_like(x_full)
        w_acc = torch.zeros((b, 1, hL, wL), device=x_full.device, dtype=x_full.dtype)

        def feather_mask(h: int, w: int, yi0: int, yi1: int, xi0: int, xi1: int):
            HH = yi1 - yi0
            WW = xi1 - xi0
            yy = torch.arange(HH, device=x_full.device, dtype=x_full.dtype).view(HH, 1)
            xx = torch.arange(WW, device=x_full.device, dtype=x_full.dtype).view(1, WW)
            top = yy.float()
            left = xx.float()
            bottom = (HH - 1) - yy.float()
            right = (WW - 1) - xx.float()
            d = torch.min(torch.min(top, bottom), torch.min(left, right))
            ramp = torch.clamp(d / float(max(1, blurL)), 0.0, 1.0)
            return ramp

        stride_y = max(1, thL - 2 * padL)
        stride_x = max(1, twL - 2 * padL)
        for by in range(0, hL, stride_y):
            for bx in range(0, wL, stride_x):
                y0 = max(0, by - padL)
                x0 = max(0, bx - padL)
                y1 = min(hL, by + thL + padL)
                x1 = min(wL, bx + twL + padL)
                if (y1 - y0) <= 1 or (x1 - x0) <= 1:
                    continue

                tile_lat = {"samples": x_full[:, :, y0:y1, x0:x1]}

                class _TileNoise:
                    def __init__(self, base_noise, y0, y1, x0, x1):
                        self.base = base_noise
                        self.seed = 0
                        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1

                    def generate_noise(self, input_latent):
                        return self.base[:, :, self.y0:self.y1, self.x0:self.x1]

                tnoise = _TileNoise(full_noise, y0, y1, x0, x1)

                out_lat_tile, _ = WASAffineCustomAdvanced.sample(
                    tnoise,
                    g,
                    sampler_obj,
                    sigmas,
                    tile_lat,
                    affine_interval,
                    max_scale,
                    max_bias,
                    pattern,
                    affine_seed,
                    affine_seed_increment,
                    affine_schedule,
                    external_mask=external_mask,
                    options=options,
                    noise_options=noise_options,
                    temporal_mode=temporal_mode,
                    cfg_list=cfg,
                )
                tile_out = out_lat_tile["samples"] if isinstance(out_lat_tile, dict) else out_lat_tile
                try:
                    if (tile_out.device != x_full.device) or (tile_out.dtype != x_full.dtype):
                        print(f"[WAS-UltimateCustom] Casting tile_out from {tile_out.device},{tile_out.dtype} -> {x_full.device},{x_full.dtype}")
                except Exception:
                    pass
                tile_out = tile_out.to(device=x_full.device, dtype=x_full.dtype)
                if not tile_out.is_contiguous():
                    tile_out = tile_out.contiguous()

                w2d = feather_mask(hL, wL, y0, y1, x0, x1).unsqueeze(0).unsqueeze(0)
                try:
                    if (w2d.device != x_full.device) or (w2d.dtype != x_full.dtype):
                        print(f"[WAS-UltimateCustom] Casting w2d from {w2d.device},{w2d.dtype} -> {x_full.device},{x_full.dtype}")
                except Exception:
                    pass
                w2d = w2d.to(device=x_full.device, dtype=x_full.dtype)

                if (out_acc.device != x_full.device) or (w_acc.device != x_full.device):
                    out_acc = out_acc.to(device=x_full.device, dtype=x_full.dtype)
                    w_acc = w_acc.to(device=x_full.device, dtype=x_full.dtype)
                if (tile_out.device != x_full.device) or (w2d.device != x_full.device):
                    raise RuntimeError(f"Device mismatch before accumulation: tile_out={tile_out.device}, w2d={w2d.device}, x_full={x_full.device}")
                if (tile_out.dtype != x_full.dtype) or (w2d.dtype != x_full.dtype):
                    try:
                        print(f"[WAS-UltimateCustom] Dtype mismatch before accumulation: tile_out={tile_out.dtype}, w2d={w2d.dtype}, x_full={x_full.dtype}")
                    except Exception:
                        pass
                    tile_out = tile_out.to(dtype=x_full.dtype)
                    w2d = w2d.to(dtype=x_full.dtype)

                try:
                    print(f"[WAS-UltimateCustom] Accumulate devs: out_acc={out_acc.device}, tile_out={tile_out.device}, w2d={w2d.device}")
                except Exception:
                    pass
                out_acc[:, :, y0:y1, x0:x1] += tile_out * w2d
                w_acc[:, :, y0:y1, x0:x1] += w2d

        w_safe = torch.where(w_acc > 0, w_acc, torch.ones_like(w_acc))
        merged = out_acc / w_safe
        merged = torch.where(w_acc > 0, merged, x_full)

        try:
            out_img = vae.decode(merged)
        except Exception as e:
            raise RuntimeError(f"VAE.decode failed: {e}")
        return (out_img,)


class WASUltimateCustomAdvancedAffineCustom(WASUltimateCustomAdvancedAffineNoUpscale):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["custom_sampler"] = ("SAMPLER", {"tooltip": "Optional custom SAMPLER; if provided, used instead of sampler_name."})
        base["optional"]["custom_sigmas"] = ("SIGMAS", {"tooltip": "Optional custom SIGMAS; if provided, overrides scheduler/steps/denoise."})
        return base

    @classmethod
    def upscale(
        cls,
        image,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        noise,
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
        temporal_mode="static",
        custom_sampler=None,
        custom_sigmas=None,
    ):
        try:
            latent = vae.encode(image)
        except Exception as e:
            raise RuntimeError(f"VAE.encode failed: {e}")

        try:
            g = comfy.samplers.CFGGuider(model)
            g.set_conds(positive, negative)
            cfg_value = get_cfg_for_step(cfg, 0, steps)
            g.set_cfg(cfg_value)
            sampler_obj = custom_sampler if custom_sampler is not None else comfy.samplers.sampler_object(sampler_name)
            sigmas = custom_sigmas if custom_sigmas is not None else _build_sigmas_from_model(model, scheduler, int(steps), float(denoise))
        except Exception as e:
            raise RuntimeError(f"Failed to prepare sampler: {e}")

        x_full = latent["samples"] if isinstance(latent, dict) else latent
        
        if len(x_full.shape) == 5:
            b, cL, fL, hL, wL = x_full.shape
            is_video = True
        elif len(x_full.shape) == 4:
            b, cL, hL, wL = x_full.shape
            is_video = False
        else:
            raise ValueError(f"Unsupported latent shape: {x_full.shape}. Expected 4D or 5D tensor.")
        
        try:
            if len(image.shape) == 5 and is_video:
                _, _, Hpx, Wpx, _ = image.shape
            elif len(image.shape) == 4:
                _, Hpx, Wpx, _ = image.shape
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
            scale_h = max(1.0, float(Hpx) / float(hL))
            scale_w = max(1.0, float(Wpx) / float(wL))
        except Exception:
            scale_h = scale_w = 8.0

        def to_lat(v_px: int, scale: float):
            return max(1, int(round(float(v_px) / scale)))

        twL = to_lat(int(tile_width), scale_w)
        thL = to_lat(int(tile_height), scale_h)
        padL = to_lat(int(tile_padding), (scale_h + scale_w) * 0.5)
        blurL = to_lat(int(mask_blur), (scale_h + scale_w) * 0.5)

        try:
            full_noise = noise.generate_noise(latent)
            if isinstance(full_noise, torch.Tensor):
                full_noise = full_noise.to(device=x_full.device, dtype=x_full.dtype)
            else:
                full_noise = torch.zeros_like(x_full)
        except Exception:
            full_noise = torch.zeros_like(x_full)

        out_acc = torch.zeros_like(x_full)
        w_acc = torch.zeros((b, 1, hL, wL), device=x_full.device, dtype=x_full.dtype)

        def feather_mask(h: int, w: int, yi0: int, yi1: int, xi0: int, xi1: int):
            HH = yi1 - yi0
            WW = xi1 - xi0
            yy = torch.arange(HH, device=x_full.device, dtype=x_full.dtype).view(HH, 1)
            xx = torch.arange(WW, device=x_full.device, dtype=x_full.dtype).view(1, WW)
            top = yy.float()
            left = xx.float()
            bottom = (HH - 1) - yy.float()
            right = (WW - 1) - xx.float()
            d = torch.min(torch.min(top, bottom), torch.min(left, right))
            ramp = torch.clamp(d / float(max(1, blurL)), 0.0, 1.0)
            return ramp  # [HH,WW]

        stride_y = max(1, thL - 2 * padL)
        stride_x = max(1, twL - 2 * padL)
        for by in range(0, hL, stride_y):
            for bx in range(0, wL, stride_x):
                y0 = max(0, by - padL)
                x0 = max(0, bx - padL)
                y1 = min(hL, by + thL + padL)
                x1 = min(wL, bx + twL + padL)
                if (y1 - y0) <= 1 or (x1 - x0) <= 1:
                    continue

                tile_lat = {"samples": x_full[:, :, y0:y1, x0:x1]}

                class _TileNoise:
                    def __init__(self, base_noise, y0, y1, x0, x1):
                        self.base = base_noise
                        self.seed = 0
                        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1

                    def generate_noise(self, input_latent):
                        return self.base[:, :, self.y0:self.y1, self.x0:self.x1]

                tnoise = _TileNoise(full_noise, y0, y1, x0, x1)

                out_lat_tile, _ = WASAffineCustomAdvanced.sample(
                    tnoise,
                    g,
                    sampler_obj,
                    sigmas,
                    tile_lat,
                    affine_interval,
                    max_scale,
                    max_bias,
                    pattern,
                    affine_seed,
                    affine_seed_increment,
                    affine_schedule,
                    external_mask=external_mask,
                    options=options,
                    noise_options=noise_options,
                    temporal_mode=temporal_mode,
                    cfg_list=cfg,
                )
                tile_out = out_lat_tile["samples"] if isinstance(out_lat_tile, dict) else out_lat_tile
                
                tile_out = tile_out.to(device=x_full.device, dtype=x_full.dtype)
                if not tile_out.is_contiguous():
                    tile_out = tile_out.contiguous()

                w2d = feather_mask(hL, wL, y0, y1, x0, x1).unsqueeze(0).unsqueeze(0)
                w2d = w2d.to(device=x_full.device, dtype=x_full.dtype)
                
                if out_acc.device != x_full.device:
                    out_acc = out_acc.to(device=x_full.device, dtype=x_full.dtype)
                if w_acc.device != x_full.device:
                    w_acc = w_acc.to(device=x_full.device, dtype=x_full.dtype)
                
                out_acc[:, :, y0:y1, x0:x1] += tile_out * w2d
                w_acc[:, :, y0:y1, x0:x1] += w2d

        w_safe = torch.where(w_acc > 0, w_acc, torch.ones_like(w_acc))
        merged = out_acc / w_safe
        merged = torch.where(w_acc > 0, merged, x_full)

        try:
            out_img = vae.decode(merged)
        except Exception as e:
            raise RuntimeError(f"VAE.decode failed: {e}")
        return (out_img,)


class WASUltimateCustomAdvancedAffine(WASUltimateCustomAdvancedAffineNoUpscale):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"]["upscale_model"] = ("UPSCALE_MODEL", {"tooltip": "Upscale model to use for pre-upscaling the image before tiling."})
        base["required"]["upscale_factor"] = ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "tooltip": "Target upscale factor. Image will be rescaled to this size regardless of model's native scale."})
        return base

    @classmethod
    def upscale(
        cls,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_model,
        upscale_factor,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        noise,
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
        temporal_mode="static",
    ):
        print(f"[WAS Affine] Upscaling image to {upscale_factor}x...")
        
        import torch.nn.functional as F
        
        is_video = image.dim() == 5
        
        if is_video:
            b, f, h, w, c = image.shape
            print(f"[WAS Affine] Processing video with {f} frames at {h}x{w}")
            new_h = int(h * upscale_factor)
            new_w = int(w * upscale_factor)
            
            try:
                from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
                upscaler = ImageUpscaleWithModel()
                
                upscaled_frames = []
                for frame_idx in range(f):
                    frame = image[:, frame_idx]
                    model_upscaled_frame = upscaler.upscale(upscale_model, frame)[0]
                    
                    if model_upscaled_frame.shape[1] != new_h or model_upscaled_frame.shape[2] != new_w:
                        frame_tensor = model_upscaled_frame.permute(0, 3, 1, 2)
                        frame_tensor = F.interpolate(
                            frame_tensor,
                            size=(new_h, new_w),
                            mode='bilinear',
                            align_corners=False
                        )
                        model_upscaled_frame = frame_tensor.permute(0, 2, 3, 1)
                    
                    upscaled_frames.append(model_upscaled_frame)
                
                upscaled_image = torch.stack(upscaled_frames, dim=1)
                
            except ImportError:
                print(f"[WAS Affine] ImageUpscaleWithModel not available, using bilinear rescaling for video")
                img_reshaped = image.view(b * f, h, w, c)
                img_tensor = img_reshaped.permute(0, 3, 1, 2)
                
                upscaled_tensor = F.interpolate(
                    img_tensor,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                upscaled_reshaped = upscaled_tensor.permute(0, 2, 3, 1)
                upscaled_image = upscaled_reshaped.view(b, f, new_h, new_w, c)
            
            print(f"[WAS Affine] Video upscaled: {h}x{w} -> {new_h}x{new_w} ({upscale_factor}x, {f} frames)")
            
        else:
            img_tensor = image.permute(0, 3, 1, 2)
            b, c, h, w = img_tensor.shape
            new_h = int(h * upscale_factor)
            new_w = int(w * upscale_factor)
            
            try:
                from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
                upscaler = ImageUpscaleWithModel()
                model_upscaled = upscaler.upscale(upscale_model, image)[0]
                model_tensor = model_upscaled.permute(0, 3, 1, 2)
                mb, mc, mh, mw = model_tensor.shape
                
                if mh != new_h or mw != new_w:
                    print(f"[WAS Affine] Model upscaled to {mh}x{mw}, rescaling to target {new_h}x{new_w}")
                    upscaled_tensor = F.interpolate(
                        model_tensor,
                        size=(new_h, new_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    upscaled_image = upscaled_tensor.permute(0, 2, 3, 1)
                else:
                    upscaled_image = model_upscaled
                    
            except ImportError:
                print(f"[WAS Affine] ImageUpscaleWithModel not available, using bilinear rescaling to {upscale_factor}x")
                upscaled_tensor = F.interpolate(
                    img_tensor, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                upscaled_image = upscaled_tensor.permute(0, 2, 3, 1)
            
            print(f"[WAS Affine] Image upscaled: {h}x{w} -> {new_h}x{new_w} ({upscale_factor}x)")
        
        return super().upscale(
            upscaled_image,
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            mode_type,
            tile_width,
            tile_height,
            mask_blur,
            tile_padding,
            seam_fix_mode,
            seam_fix_denoise,
            seam_fix_mask_blur,
            seam_fix_width,
            seam_fix_padding,
            force_uniform_tiles,
            tiled_decode,
            noise,
            affine_interval,
            max_scale,
            max_bias,
            pattern,
            affine_seed,
            affine_seed_increment,
            affine_schedule,
            external_mask=external_mask,
            options=options,
            noise_options=noise_options,
            temporal_mode=temporal_mode,
        )


if _usdu_available():
    print("[was-affine] Ultimate Custom Advanced Affine Upscalers loaded.")
    AFFINE_NODE_CLASS_MAPPINGS.update({
        "WASUltimateCustomAdvancedAffineNoUpscale": WASUltimateCustomAdvancedAffineNoUpscale,
        "WASUltimateCustomAdvancedAffineCustom": WASUltimateCustomAdvancedAffineCustom,
        "WASUltimateCustomAdvancedAffine": WASUltimateCustomAdvancedAffine,
    })
    AFFINE_NODE_DISPLAY_NAME_MAPPINGS.update({
        "WASUltimateCustomAdvancedAffineNoUpscale": "Ultimate Affine KSampler (No Upscale) - USDU",
        "WASUltimateCustomAdvancedAffineCustom": "Ultimate Affine KSampler (Custom) - USDU",
        "WASUltimateCustomAdvancedAffine": "Ultimate Affine KSampler - USDU",
    })
else:
    print("[was-affine] Ultimate Custom Advanced Affine Upscalers not registered: USDU not found")