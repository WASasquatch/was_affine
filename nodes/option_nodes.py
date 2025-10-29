import torch

from ..modules.utils import affine_step_schedule

class WASLatentAffineOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.0001, "tooltip": "Scales mask intensity before applying scale/bias. Examples: 0.5 = weaker effect, 1.0 = normal, 2.0 = strong. Influence: higher values increase the contribution of the mask to scaling and bias."}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Binarize mask if > 0. Converts mask to 0/1 using this threshold. Examples: 0.5 creates hard separation. Influence: higher threshold produces larger black areas; lower produces larger white areas."}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Invert the mask after threshold/blur. Examples: True flips dark/bright regions. Influence: swaps where scale/bias are applied."}),
                "perlin_scale": ("FLOAT", {"default": 64.0, "min": 4.0, "max": 1024.0, "step": 1.0, "tooltip": "Controls Perlin noise frequency. Examples: 32 = coarse blobs, 128 = fine details. Influence: larger scale gives smaller features (higher frequency)."}),
                "perlin_octaves": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Number of Perlin octaves. Examples: 1 = simple, 4 = richer. Influence: more octaves add multi-scale detail."}),
                "perlin_persistence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier between octaves. Examples: 0.3 = smoother, 0.8 = higher contrast. Influence: higher increases contribution of finer octaves."}),
                "perlin_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Frequency multiplier between octaves. Examples: 2.0 (default), 3.0 = more fine detail. Influence: higher values add finer features faster."}),
                "checker_size": ("INT", {"default": 8, "min": 2, "max": 256, "tooltip": "Checkerboard cell size in pixels. Examples: 8 = small squares, 64 = large squares. Influence: sets tiling size for 'checker' pattern."}),
                "bayer_size": ("INT", {"default": 8, "min": 2, "max": 64, "tooltip": "Bayer matrix base size. Examples: 4, 8, 16. Influence: controls dithering pattern scale for 'bayer'."}),
                "velvet_taps_per_kpx": ("INT", {"default": 10, "min": 1, "max": 10000, "tooltip": "Velvet noise density in taps per kilo-pixel (per 1000 pixels). Higher = more impulses (denser)."}),
                "green_center_frac": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Green noise band center as a fraction of Nyquist radius (0..1)."}),
                "green_bandwidth_frac": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Green noise Gaussian band width as a fraction of Nyquist radius (0..1)."}),
                "black_bins_per_kpx": ("INT", {"default": 512, "min": 1, "max": 500000, "tooltip": "Black noise density as active frequency bins per 1000 pixels (rFFT domain). Higher = more narrowbands."}),
                "hatch_freq_cyc_px": ("FLOAT", {"default": 0.45, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": "Cross-hatch frequency in cycles/pixel. >0.5 allows intentional aliasing."}),
                "hatch_angle1_deg": ("INT", {"default": 0, "min": 0, "max": 179, "tooltip": "First hatch angle (degrees)."}),
                "hatch_angle2_deg": ("INT", {"default": 90, "min": 0, "max": 179, "tooltip": "Second hatch angle (degrees)."}),
                "hatch_square": ("BOOLEAN", {"default": False, "tooltip": "Use square-wave (adds harmonics)."}),
                "hatch_phase_jitter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Random phase jitter multiplier per seed/frame."}),
                "hatch_supersample": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Supersampling factor for anti-aliased subpixel hatch."}),
                "highpass_cutoff_frac": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Butterworth HPF cutoff as fraction of Nyquist radius."}),
                "highpass_order": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Butterworth HPF order."}),
                "ring_center_frac": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ring center as fraction of Nyquist radius."}),
                "ring_bandwidth_frac": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 1.0, "step": 0.005, "tooltip": "Ring Gaussian bandwidth as fraction of Nyquist radius."}),
                "poisson_radius_px": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 256.0, "step": 0.5, "tooltip": "Poisson-disk minimum spacing in pixels."}),
                "poisson_softness": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 256.0, "step": 0.1, "tooltip": "Distance mapping softness. Larger = smoother mask."}),
                "worley_points_per_kpx": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 200.0, "step": 0.1, "tooltip": "Feature points per 1000 pixels."}),
                "worley_metric": (["L2", "L1"], {"default": "L2", "tooltip": "Distance metric for Worley."}),
                "worley_edge_sharpness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Exponent to sharpen edges."}),
                "tile_line_tile_size": ("INT", {"default": 32, "min": 4, "max": 512, "tooltip": "Tile size in pixels."}),
                "tile_line_freq_cyc_px": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": "Line frequency in cycles/pixel per tile."}),
                "tile_line_jitter": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Phase jitter per tile (0..1)."}),
                "dot_cell_size": ("INT", {"default": 12, "min": 2, "max": 256, "tooltip": "Halftone cell size in pixels."}),
                "dot_jitter_px": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Dot center jitter in pixels."}),
                "dot_fill_ratio": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 0.95, "step": 0.01, "tooltip": "Approximate fill area per cell (0..1)."}),
                "content_window": ("INT", {"default": 7, "min": 3, "max": 63, "tooltip": "Odd kernel size for content-aware local variance (detail/smooth patterns). Typical: 5-11."}),
                "solid_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Solid mask constant alpha (0..1). 0 disables; 1 full mask."}),
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
    CATEGORY = "latent/adjust"

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
        velvet_taps_per_kpx,
        green_center_frac,
        green_bandwidth_frac,
        black_bins_per_kpx,
        hatch_freq_cyc_px,
        hatch_angle1_deg,
        hatch_angle2_deg,
        hatch_square,
        hatch_phase_jitter,
        hatch_supersample,
        highpass_cutoff_frac,
        highpass_order,
        ring_center_frac,
        ring_bandwidth_frac,
        poisson_radius_px,
        poisson_softness,
        worley_points_per_kpx,
        worley_metric,
        worley_edge_sharpness,
        tile_line_tile_size,
        tile_line_freq_cyc_px,
        tile_line_jitter,
        dot_cell_size,
        dot_jitter_px,
        dot_fill_ratio,
        content_window,
        solid_alpha,
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
            "velvet_taps_per_kpx": int(velvet_taps_per_kpx),
            "green_center_frac": float(green_center_frac),
            "green_bandwidth_frac": float(green_bandwidth_frac),
            "black_bins_per_kpx": int(black_bins_per_kpx),
            "hatch_freq_cyc_px": float(hatch_freq_cyc_px),
            "hatch_angle1_deg": float(hatch_angle1_deg),
            "hatch_angle2_deg": float(hatch_angle2_deg),
            "hatch_square": bool(hatch_square),
            "hatch_phase_jitter": float(hatch_phase_jitter),
            "hatch_supersample": int(hatch_supersample),
            "highpass_cutoff_frac": float(highpass_cutoff_frac),
            "highpass_order": int(highpass_order),
            "ring_center_frac": float(ring_center_frac),
            "ring_bandwidth_frac": float(ring_bandwidth_frac),
            "poisson_radius_px": float(poisson_radius_px),
            "poisson_softness": float(poisson_softness),
            "worley_points_per_kpx": float(worley_points_per_kpx),
            "worley_metric": str(worley_metric),
            "worley_edge_sharpness": float(worley_edge_sharpness),
            "tile_line_tile_size": int(tile_line_tile_size),
            "tile_line_freq_cyc_px": float(tile_line_freq_cyc_px),
            "tile_line_jitter": float(tile_line_jitter),
            "dot_cell_size": int(dot_cell_size),
            "dot_jitter_px": float(dot_jitter_px),
            "dot_fill_ratio": float(dot_fill_ratio),
            "content_window": int(content_window),
            "solid_alpha": float(solid_alpha),
            "blur_ksize": int(blur_ksize),
            "blur_sigma": float(blur_sigma),
            "clamp": bool(clamp),
            "clamp_min": float(clamp_min),
            "clamp_max": float(clamp_max),
            "frame_seed_stride": int(frame_seed_stride),
        },)


class WASLatentAffineCommonOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.0001, "tooltip": "Scales mask intensity before applying scale/bias."}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Binarize mask if > 0."}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Invert after threshold/blur."}),
                "sharpen_enable": ("BOOLEAN", {"default": False, "tooltip": "Sharpen the generated noise/mask before threshold/blur using unsharp masking. Off by default."}),
                "sharpen_sigma": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip": "Sharpen radius (Gaussian sigma). 0 disables. Typical: 0.5-1.5."}),
                "sharpen_amount": ("FLOAT", {"default": 0.3, "min": -5.0, "max": 5.0, "step": 0.01, "tooltip": "Sharpen gain. 0 = none; 0.3 subtle; 1.0 strong. Negative softens (inverse). Extended range [-5,5] for stronger effects."}),
                "sharpen_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Only enhance details where |orig-blur| exceeds this threshold. Helps avoid amplifying flat noise."}),
                "blur_ksize": ("INT", {"default": 0, "min": 0, "max": 51, "tooltip": "Gaussian blur kernel size (odd)."}),
                "blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Gaussian blur sigma."}),
                "clamp": ("BOOLEAN", {"default": False, "tooltip": "Clamp output latent values to [min,max]."}),
                "clamp_min": ("FLOAT", {"default": -10.0, "min": -100.0, "max": 0.0, "step": 0.1, "tooltip": "Lower clamp bound when clamping is enabled."}),
                "clamp_max": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Upper clamp bound when clamping is enabled."}),
                "frame_seed_stride": ("INT", {"default": 9973, "min": 1, "max": 100000, "tooltip": "Seed increment per frame when temporal mode is per_frame."}),
                "compute_device": (["auto", "cuda", "cpu"], {"default": "cuda" if torch.cuda.is_available() else "cpu", "tooltip": "Where to generate procedural masks/noise. 'auto' prefers CUDA when available; otherwise CPU."}),
                "device_index": ("INT", {"default": 0, "min": 0, "max": 7, "tooltip": "CUDA device index when using GPU."}),
                "solid_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Solid mask constant alpha (0..1). 0 disables; 1 full mask."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, mask_strength, threshold, invert_mask, sharpen_enable, sharpen_sigma, sharpen_amount, sharpen_threshold, blur_ksize, blur_sigma, clamp, clamp_min, clamp_max, frame_seed_stride, compute_device, device_index, solid_alpha):
        return ({
            "mask_strength": float(mask_strength),
            "threshold": float(threshold),
            "invert_mask": bool(invert_mask),
            "sharpen_enable": bool(sharpen_enable),
            "sharpen_sigma": float(sharpen_sigma),
            "sharpen_amount": float(sharpen_amount),
            "sharpen_threshold": float(sharpen_threshold),
            "blur_ksize": int(blur_ksize),
            "blur_sigma": float(blur_sigma),
            "clamp": bool(clamp),
            "clamp_min": float(clamp_min),
            "clamp_max": float(clamp_max),
            "frame_seed_stride": int(frame_seed_stride),
            "compute_device": str(compute_device),
            "device_index": int(device_index),
            "solid_alpha": float(solid_alpha),
        },)


class WASDetailRegionOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "content_window": ("INT", {"default": 7, "min": 3, "max": 63, "tooltip": "Odd kernel size for local variance used in detail regions."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, content_window):
        return ({
            "content_window": int(content_window),
        },)


class WASSmoothRegionOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "content_window": ("INT", {"default": 7, "min": 3, "max": 63, "tooltip": "Odd kernel size for local variance used in smooth regions."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, content_window):
        return ({
            "content_window": int(content_window),
        },)


class WASCrossHatchOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hatch_freq_cyc_px": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": "Line frequency in cycles per pixel. Higher = denser lines."}),
                "hatch_angle1_deg": ("INT", {"default": 0, "min": 0, "max": 179, "tooltip": "Primary hatch angle in degrees."}),
                "hatch_angle2_deg": ("INT", {"default": 90, "min": 0, "max": 179, "tooltip": "Secondary cross angle in degrees (for cross-hatch)."}),
                "hatch_square": ("BOOLEAN", {"default": False, "tooltip": "If true, uses square wave instead of sine for sharper lines."}),
                "hatch_phase_jitter": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Random phase jitter [0-1] to break uniformity."}),
                "hatch_supersample": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Supersampling factor to reduce aliasing (slower when >1)."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, hatch_freq_cyc_px, hatch_angle1_deg, hatch_angle2_deg, hatch_square, hatch_phase_jitter, hatch_supersample):
        return ({
            "hatch_freq_cyc_px": float(hatch_freq_cyc_px),
            "hatch_angle1_deg": float(hatch_angle1_deg),
            "hatch_angle2_deg": float(hatch_angle2_deg),
            "hatch_square": bool(hatch_square),
            "hatch_phase_jitter": float(hatch_phase_jitter),
            "hatch_supersample": int(hatch_supersample),
        },)


class WASHighpassWhiteOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "highpass_cutoff_frac": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Normalized cutoff fraction (0-1) of Nyquist; higher keeps more high frequencies."}),
            "highpass_order": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Butterworth filter order. Higher = steeper roll-off."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, highpass_cutoff_frac, highpass_order):
        return ({
            "highpass_cutoff_frac": float(highpass_cutoff_frac),
            "highpass_order": int(highpass_order),
        },)


class WASRingNoiseOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ring_center_frac": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ring center as a fraction of Nyquist radius (0-1)."}),
            "ring_bandwidth_frac": ("FLOAT", {"default": 0.06, "min": 0.005, "max": 1.0, "step": 0.005, "tooltip": "Fractional bandwidth (thickness) of the ring."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, ring_center_frac, ring_bandwidth_frac):
        return ({
            "ring_center_frac": float(ring_center_frac),
            "ring_bandwidth_frac": float(ring_bandwidth_frac),
        },)


class WASPoissonBlueOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "poisson_radius_px": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 256.0, "step": 0.5, "tooltip": "Target Poisson-disk minimum distance in pixels (controls dot spacing)."}),
            "poisson_softness": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 256.0, "step": 0.1, "tooltip": "Softening of distance field to produce smoother mask."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, poisson_radius_px, poisson_softness):
        return ({
            "poisson_radius_px": float(poisson_radius_px),
            "poisson_softness": float(poisson_softness),
        },)


class WASWorleyEdgesOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "worley_points_per_kpx": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 200.0, "step": 0.1, "tooltip": "Cell density: points per kilo-pixel (1000 px). Higher = finer cells."}),
            "worley_metric": (["L2", "L1"], {"default": "L2", "tooltip": "Distance metric used by Worley noise (L2=Euclidean, L1=Manhattan)."}),
            "worley_edge_sharpness": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Exponent for emphasizing edges; higher = crisper edges."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, worley_points_per_kpx, worley_metric, worley_edge_sharpness):
        return ({
            "worley_points_per_kpx": float(worley_points_per_kpx),
            "worley_metric": str(worley_metric),
            "worley_edge_sharpness": float(worley_edge_sharpness),
        },)


class WASTileLinesOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "tile_line_tile_size": ("INT", {"default": 24, "min": 4, "max": 512, "tooltip": "Tile edge length in pixels. Each tile has an oriented line pattern."}),
            "tile_line_freq_cyc_px": ("FLOAT", {"default": 0.55, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": "Line frequency in cycles per pixel inside each tile."}),
            "tile_line_jitter": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Randomness of tile orientation/phase [0-1]."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, tile_line_tile_size, tile_line_freq_cyc_px, tile_line_jitter):
        return ({
            "tile_line_tile_size": int(tile_line_tile_size),
            "tile_line_freq_cyc_px": float(tile_line_freq_cyc_px),
            "tile_line_jitter": float(tile_line_jitter),
        },)


class WASDotScreenOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dot_cell_size": ("INT", {"default": 10, "min": 2, "max": 256, "tooltip": "Halftone cell size in pixels."}),
            "dot_jitter_px": ("FLOAT", {"default": 1.6, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Random jitter amplitude in pixels for dot centers."}),
            "dot_fill_ratio": ("FLOAT", {"default": 0.32, "min": 0.01, "max": 0.95, "step": 0.01, "tooltip": "Target area coverage per cell (0-1)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, dot_cell_size, dot_jitter_px, dot_fill_ratio):
        return ({
            "dot_cell_size": int(dot_cell_size),
            "dot_jitter_px": float(dot_jitter_px),
            "dot_fill_ratio": float(dot_fill_ratio),
        },)


class WASGreenNoiseOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "green_center_frac": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Band-pass center as fraction of Nyquist (mid-frequencies)."}),
            "green_bandwidth_frac": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Relative bandwidth around the center frequency."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, green_center_frac, green_bandwidth_frac):
        return ({
            "green_center_frac": float(green_center_frac),
            "green_bandwidth_frac": float(green_bandwidth_frac),
        },)


class WASBlackNoiseOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "black_bins_per_kpx": ("INT", {"default": 768, "min": 1, "max": 500000, "tooltip": "Histogram bins per kilo-pixel for black noise construction (higher = smoother spectrum)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, black_bins_per_kpx):
        return ({
            "black_bins_per_kpx": int(black_bins_per_kpx),
        },)


class WASPerlinOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "perlin_scale": ("FLOAT", {"default": 48.0, "min": 4.0, "max": 1024.0, "step": 1.0, "tooltip": "Base feature size in pixels. Larger = smoother patterns."}),
            "perlin_octaves": ("INT", {"default": 4, "min": 1, "max": 8, "tooltip": "Number of noise octaves to sum (multi-frequency detail)."}),
            "perlin_persistence": ("FLOAT", {"default": 0.55, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Amplitude falloff per octave (lower = less high-frequency energy)."}),
            "perlin_lacunarity": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Frequency multiplier per octave (2.0 doubles frequency each octave)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, perlin_scale, perlin_octaves, perlin_persistence, perlin_lacunarity):
        return ({
            "perlin_scale": float(perlin_scale),
            "perlin_octaves": int(perlin_octaves),
            "perlin_persistence": float(perlin_persistence),
            "perlin_lacunarity": float(perlin_lacunarity),
        },)


class WASVelvetOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "velvet_taps_per_kpx": ("INT", {"default": 14, "min": 1, "max": 10000, "tooltip": "Impulse density: taps per kilo-pixel (sparse high-frequency impulses)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, velvet_taps_per_kpx):
        return ({
            "velvet_taps_per_kpx": int(velvet_taps_per_kpx),
        },)


class WASCheckerOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checker_size": ("INT", {"default": 6, "min": 2, "max": 256, "tooltip": "Size of each checker square in pixels."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, checker_size):
        return ({
            "checker_size": int(checker_size),
        },)


class WASBayerOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bayer_size": ("INT", {"default": 2, "min": 2, "max": 64, "tooltip": "Bayer matrix size. Use small values (e.g., 2) to act as fine noise instead of obvious dithering."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, bayer_size):
        return ({
            "bayer_size": int(bayer_size),
        },)


class WASSolidMaskOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "solid_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Solid mask constant alpha (0..1). 0 = no effect; 1 = full mask."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("noise_options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, solid_alpha):
        return ({
            "solid_alpha": float(solid_alpha),
        },)


class WASAffineScheduleOptions:
    CATEGORY = "sampling/custom_sampling/options"
    RETURN_TYPES = ("DICT", "IMAGE")
    RETURN_NAMES = ("affine_schedule", "plot")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "curve": ([
                    "linear",
                    "easeInSine","easeOutSine","easeInOutSine",
                    "easeInQuad","easeOutQuad","easeInOutQuad",
                    "easeInCubic","easeOutCubic","easeInOutCubic",
                    "easeInQuart","easeOutQuart","easeInOutQuart",
                    "easeInQuint","easeOutQuint","easeInOutQuint",
                    "easeInExpo","easeOutExpo","easeInOutExpo",
                    "easeInCirc","easeOutCirc","easeInOutCirc",
                    "easeInBack","easeOutBack","easeInOutBack",
                    "cubicBezier"
                ], {"default": "easeInOutSine", "tooltip": "Easing curve that shapes the schedule between start and end."}),
            },
            "optional": {
                "plot_steps": ("INT", {"default": 100, "min": 10, "max": 2000, "step": 10, "tooltip": "Number of points to sample across 0..steps for plotting the schedule."}),
                "plot_width": ("INT", {"default": 640, "min": 160, "max": 4096, "step": 16, "tooltip": "Output plot image width in pixels."}),
                "plot_height": ("INT", {"default": 320, "min": 120, "max": 2048, "step": 8, "tooltip": "Output plot image height in pixels."}),
                "show_grid": ("BOOLEAN", {"default": True, "tooltip": "Overlay a light grid on the plot if matplotlib is available."}),
                "show_points": ("BOOLEAN", {"default": False, "tooltip": "Draw point markers on the curve if matplotlib is available."}),
            },
        }

    @classmethod
    def build(
        cls,
        start,
        end,
        bias,
        exponent,
        start_offset,
        end_offset,
        curve,
        plot_steps=100,
        plot_width=640,
        plot_height=320,
        show_grid=True,
        show_points=False,
    ):
        sched = {
            "start": float(start),
            "end": float(end),
            "bias": float(bias),
            "exponent": float(exponent),
            "start_offset": float(start_offset),
            "end_offset": float(end_offset),
            "curve": str(curve),
        }
        if curve == "cubicBezier":
            sched["bezier"] = (0.25, 0.1, 0.25, 1.0)
        elif curve in ("easeInBack", "easeOutBack", "easeInOutBack"):
            sched["back_k"] = 1.70158

        try:
            steps = max(2, int(plot_steps))
        except Exception:
            steps = 100
        try:
            w = max(16, int(plot_width))
            h = max(16, int(plot_height))
        except Exception:
            w, h = 640, 320

        # Reuse internal schedule builder
        dd = affine_step_schedule(steps, {
            "start": sched["start"],
            "end": sched["end"],
            "bias": sched["bias"],
            "exponent": sched["exponent"],
            "start_offset": sched["start_offset"],
            "end_offset": sched["end_offset"],
            "curve": sched.get("curve", "easeInOutSine"),
            "back_k": sched.get("back_k", 1.70158),
            "bezier": sched.get("bezier"),
        })

        plot_img = None
        # Try matplotlib first
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

            fig = Figure(figsize=(w / 100.0, h / 100.0), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            xs = list(range(steps))
            ax.plot(xs, dd, "-o" if show_points else "-", linewidth=2.0, markersize=3.0)
            ax.set_xlim(0, steps - 1)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Step")
            ax.set_ylabel("Schedule value")
            ax.set_title("Affine Schedule")
            if show_grid:
                ax.grid(True, linestyle=":", alpha=0.4)
            fig.tight_layout()
            canvas.draw()
            import numpy as _np
            buf = _np.asarray(canvas.buffer_rgba())  # H, W, 4
            rgb = buf[..., :3].astype(_np.float32) / 255.0
            plot_img = torch.from_numpy(rgb).unsqueeze(0)  # [1,H,W,3]
        except Exception:
            img = torch.zeros((h, w, 3), dtype=torch.float32)
            for x in range(w):
                t = x / max(1, w - 1)
                idx = min(int(t * (steps - 1)), steps - 1)
                v = float(dd[idx])
                y = h - 1 - int(v * (h - 1))
                # Draw a small vertical tick for visibility
                y0 = max(0, y - 1)
                y1 = min(h - 1, y + 1)
                img[y0:y1 + 1, x, :] = 1.0
            plot_img = img.unsqueeze(0)  # [1,H,W,3]

        return (sched, plot_img)


NODE_CLASS_MAPPINGS = {
    "WASLatentAffineOptions": WASLatentAffineOptions,
    "WASLatentAffineCommonOptions": WASLatentAffineCommonOptions,
    "WASDetailRegionOptions": WASDetailRegionOptions,
    "WASSmoothRegionOptions": WASSmoothRegionOptions,
    "WASCrossHatchOptions": WASCrossHatchOptions,
    "WASHighpassWhiteOptions": WASHighpassWhiteOptions,
    "WASRingNoiseOptions": WASRingNoiseOptions,
    "WASPoissonBlueOptions": WASPoissonBlueOptions,
    "WASWorleyEdgesOptions": WASWorleyEdgesOptions,
    "WASTileLinesOptions": WASTileLinesOptions,
    "WASDotScreenOptions": WASDotScreenOptions,
    "WASGreenNoiseOptions": WASGreenNoiseOptions,
    "WASBlackNoiseOptions": WASBlackNoiseOptions,
    "WASPerlinOptions": WASPerlinOptions,
    "WASVelvetOptions": WASVelvetOptions,
    "WASCheckerOptions": WASCheckerOptions,
    "WASBayerOptions": WASBayerOptions,
    "WASAffineScheduleOptions": WASAffineScheduleOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASLatentAffineOptions": "Latent Affine Super Options",
    "WASLatentAffine": "Latent Affine",
    "WASLatentAffineSimple": "Latent Affine Simple",
    "WASAffineKSamplerAdvanced": "KSampler Affine Advanced",
    "WASAffineKSampler": "KSampler Affine",
    "WASLatentAffineCommonOptions": "Latent Affine Common Options",
    "WASDetailRegionOptions": "Affine Detail Region Options",
    "WASSmoothRegionOptions": "Affine Smooth Region Options",
    "WASCrossHatchOptions": "Affine Cross-Hatch Noise Options",
    "WASHighpassWhiteOptions": "Affine High-pass White Noise Options",
    "WASRingNoiseOptions": "Affine Ring Noise Noise Options",
    "WASPoissonBlueOptions": "Affine Poisson Blue Noise Options",
    "WASWorleyEdgesOptions": "Affine Worley Edges Noise Options",
    "WASTileLinesOptions": "Affine Tile Lines Noise Options",
    "WASDotScreenOptions": "Affine Dot Screen Noise Options",
    "WASGreenNoiseOptions": "Affine Green Noise Options",
    "WASBlackNoiseOptions": "Affine Black Noise Options",
    "WASPerlinOptions": "Affine Perlin Noise Options",
    "WASVelvetOptions": "Affine Velvet Noise Options",
    "WASCheckerOptions": "Affine Checker Noise Options",
    "WASBayerOptions": "Affine Bayer Noise Options",
    "WASAffineScheduleOptions": "Affine Schedule Options",
}
