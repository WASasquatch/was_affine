import torch
import torch.nn.functional as F

from .utils import (
    gaussian_blur,
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
)


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
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Scales mask intensity before applying scale/bias."}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Binarize mask if > 0."}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Invert after threshold/blur."}),
                "blur_ksize": ("INT", {"default": 0, "min": 0, "max": 51, "tooltip": "Gaussian blur kernel size (odd)."}),
                "blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Gaussian blur sigma."}),
                "clamp": ("BOOLEAN", {"default": False, "tooltip": "Clamp output latent values to [min,max]."}),
                "clamp_min": ("FLOAT", {"default": -10.0, "min": -100.0, "max": 0.0, "step": 0.1}),
                "clamp_max": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "frame_seed_stride": ("INT", {"default": 9973, "min": 1, "max": 100000, "tooltip": "Seed increment per frame when temporal mode is per_frame."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, mask_strength, threshold, invert_mask, blur_ksize, blur_sigma, clamp, clamp_min, clamp_max, frame_seed_stride):
        return ({
            "mask_strength": float(mask_strength),
            "threshold": float(threshold),
            "invert_mask": bool(invert_mask),
            "blur_ksize": int(blur_ksize),
            "blur_sigma": float(blur_sigma),
            "clamp": bool(clamp),
            "clamp_min": float(clamp_min),
            "clamp_max": float(clamp_max),
            "frame_seed_stride": int(frame_seed_stride),
        },)


class WASCrossHatchOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hatch_freq_cyc_px": ("FLOAT", {"default": 0.45, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": "Line frequency in cycles per pixel. Higher = denser lines."}),
                "hatch_angle1_deg": ("INT", {"default": 0, "min": 0, "max": 179, "tooltip": "Primary hatch angle in degrees."}),
                "hatch_angle2_deg": ("INT", {"default": 90, "min": 0, "max": 179, "tooltip": "Secondary cross angle in degrees (for cross-hatch)."}),
                "hatch_square": ("BOOLEAN", {"default": False, "tooltip": "If true, uses square wave instead of sine for sharper lines."}),
                "hatch_phase_jitter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Random phase jitter [0-1] to break uniformity."}),
                "hatch_supersample": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Supersampling factor to reduce aliasing (slower when >1)."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "highpass_cutoff_frac": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Normalized cutoff fraction (0-1) of Nyquist; higher keeps more high frequencies."}),
            "highpass_order": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Butterworth filter order. Higher = steeper roll-off."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "ring_center_frac": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ring center as a fraction of Nyquist radius (0-1)."}),
            "ring_bandwidth_frac": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 1.0, "step": 0.005, "tooltip": "Fractional bandwidth (thickness) of the ring."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "poisson_radius_px": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 256.0, "step": 0.5, "tooltip": "Target Poisson-disk minimum distance in pixels (controls dot spacing)."}),
            "poisson_softness": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 256.0, "step": 0.1, "tooltip": "Softening of distance field to produce smoother mask."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "worley_points_per_kpx": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 200.0, "step": 0.1, "tooltip": "Cell density: points per kilo-pixel (1000 px). Higher = finer cells."}),
            "worley_metric": (["L2", "L1"], {"default": "L2", "tooltip": "Distance metric used by Worley noise (L2=Euclidean, L1=Manhattan)."}),
            "worley_edge_sharpness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Exponent for emphasizing edges; higher = crisper edges."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "tile_line_tile_size": ("INT", {"default": 32, "min": 4, "max": 512, "tooltip": "Tile edge length in pixels. Each tile has an oriented line pattern."}),
            "tile_line_freq_cyc_px": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": "Line frequency in cycles per pixel inside each tile."}),
            "tile_line_jitter": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Randomness of tile orientation/phase [0-1]."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "dot_cell_size": ("INT", {"default": 12, "min": 2, "max": 256, "tooltip": "Halftone cell size in pixels."}),
            "dot_jitter_px": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Random jitter amplitude in pixels for dot centers."}),
            "dot_fill_ratio": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 0.95, "step": 0.01, "tooltip": "Target area coverage per cell (0-1)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "green_center_frac": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Band-pass center as fraction of Nyquist (mid-frequencies)."}),
            "green_bandwidth_frac": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Relative bandwidth around the center frequency."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "black_bins_per_kpx": ("INT", {"default": 512, "min": 1, "max": 500000, "tooltip": "Histogram bins per kilo-pixel for black noise construction (higher = smoother spectrum)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "perlin_scale": ("FLOAT", {"default": 64.0, "min": 4.0, "max": 1024.0, "step": 1.0, "tooltip": "Base feature size in pixels. Larger = smoother patterns."}),
            "perlin_octaves": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Number of noise octaves to sum (multi-frequency detail)."}),
            "perlin_persistence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Amplitude falloff per octave (lower = less high-frequency energy)."}),
            "perlin_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Frequency multiplier per octave (2.0 doubles frequency each octave)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "velvet_taps_per_kpx": ("INT", {"default": 10, "min": 1, "max": 10000, "tooltip": "Impulse density: taps per kilo-pixel (sparse high-frequency impulses)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "checker_size": ("INT", {"default": 8, "min": 2, "max": 256, "tooltip": "Size of each checker square in pixels."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
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
            "bayer_size": ("INT", {"default": 8, "min": 2, "max": 64, "tooltip": "Bayer matrix size (power-of-two typical)."}),
        }}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "latent/adjust"

    def build(self, bayer_size):
        return ({
            "bayer_size": int(bayer_size),
        },)


class WASLatentAffine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Input latent input to apply Affine to."}),
                "scale": ("FLOAT", {"default": 0.96, "min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Multiplicative factor applied. Examples: 1.0 = no change, <1 darkens, >1 amplifies features. Influence: controls gain in masked regions."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001, "tooltip": "Additive offset applied. Examples: 0.1 brightens, -0.1 darkens. Influence: shifts values in masked regions."}),
                "pattern": ([
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
                    "external_mask"
                ], {"tooltip": "Mask source. White/pink/brown(red)/blue/violet(purple)/green/black are spectrally-shaped noises; cross_hatch is oriented gratings; highpass_white is Butterworth-shaped; ring_noise is narrow high-freq annulus; poisson_blue_mask is Poisson-disk distance field; worley_edges highlights cellular edges; tile_oriented_lines randomizes gratings per tile; dot_screen_jitter is halftone dots with jitter; velvet is sparse impulses; perlin: smooth noise; checker/bayer: tiled patterns; external_mask: use provided IMAGE directly. If an external mask is connected and pattern != external_mask, it gates the generated mask so the affine/noise only applies where this mask is 1."}),
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
        # Merge precedence: defaults -> options (base) -> noise_options (overlay)
        # Use 'options' for common/full settings; layer pattern-specific into 'noise_options'.
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
                "noise_pattern": ([
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
                ], {"tooltip": "Pattern used to generate the mask. Auto-tuned for rough/noisy results."}),
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
                "perlin_scale": max(8.0, smin / 14.0),
                "perlin_octaves": 4,
                "perlin_persistence": 0.6,
                "perlin_lacunarity": 2.2,
            })
        elif pattern == "poisson_blue_mask":
            rpx = max(4.0, smin / 28.0)
            params.update({
                "poisson_radius_px": float(rpx),
                "poisson_softness": float(max(2.0, rpx * 0.75)),
            })
        elif pattern == "worley_edges":
            params.update({
                "worley_points_per_kpx": 4.0,
                "worley_metric": "L2",
                "worley_edge_sharpness": 1.2,
            })
        elif pattern == "black_noise":
            params.update({
                "black_bins_per_kpx": 1024,
            })
        elif pattern == "green_noise":
            params.update({
                "green_center_frac": 0.4,
                "green_bandwidth_frac": 0.2,
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
                "ring_bandwidth_frac": 0.08,
            })
        elif pattern == "tile_oriented_lines":
            ts = int(max(8, smin // 32))
            params.update({
                "tile_line_tile_size": ts,
                "tile_line_freq_cyc_px": 0.5,
                "tile_line_jitter": 0.3,
            })
        elif pattern == "dot_screen_jitter":
            cs = int(max(6, smin // 48))
            params.update({
                "dot_cell_size": cs,
                "dot_jitter_px": 2.0,
                "dot_fill_ratio": 0.35,
            })
        elif pattern == "velvet_noise":
            params.update({
                "velvet_taps_per_kpx": 16,
            })
        elif pattern == "checker":
            params.update({
                "checker_size": int(max(4, smin // 32)),
            })
        elif pattern == "bayer":
            params.update({
                "bayer_size": 8 if smin < 256 else 16,
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


NODE_CLASS_MAPPINGS = {
    "LatentAffineOptions": WASLatentAffineOptions,
    "LatentMaskedAffine": WASLatentAffine,
    "LatentAffineSimple": WASLatentAffineSimple,
    "LatentAffineCommonOptions": WASLatentAffineCommonOptions,
    "CrossHatchOptions": WASCrossHatchOptions,
    "HighpassWhiteOptions": WASHighpassWhiteOptions,
    "RingNoiseOptions": WASRingNoiseOptions,
    "PoissonBlueOptions": WASPoissonBlueOptions,
    "WorleyEdgesOptions": WASWorleyEdgesOptions,
    "TileLinesOptions": WASTileLinesOptions,
    "DotScreenOptions": WASDotScreenOptions,
    "GreenNoiseOptions": WASGreenNoiseOptions,
    "BlackNoiseOptions": WASBlackNoiseOptions,
    "PerlinOptions": WASPerlinOptions,
    "VelvetOptions": WASVelvetOptions,
    "CheckerOptions": WASCheckerOptions,
    "BayerOptions": WASBayerOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentAffineOptions": "Latent Affine Super Options",
    "LatentMaskedAffine": "Latent Affine",
    "LatentAffineSimple": "Latent Affine Simple",
    "LatentAffineCommonOptions": "Latent Affine Common Options",
    "CrossHatchOptions": "Affine Cross-Hatch Noise Options",
    "HighpassWhiteOptions": "Affine High-pass White Noise Options",
    "RingNoiseOptions": "Affine Ring Noise Noise Options",
    "PoissonBlueOptions": "Affine Poisson Blue Noise Options",
    "WorleyEdgesOptions": "Affine Worley Edges Noise Options",
    "TileLinesOptions": "Affine Tile Lines Noise Options",
    "DotScreenOptions": "Affine Dot Screen Noise Options",
    "GreenNoiseOptions": "Affine Green Noise Options",
    "BlackNoiseOptions": "Affine Black Noise Options",
    "PerlinOptions": "Affine Perlin Noise Options",
    "VelvetOptions": "Affine Velvet Noise Options",
    "CheckerOptions": "Affine Checker Noise Options",
    "BayerOptions": "Affine Bayer Noise Options",
}
