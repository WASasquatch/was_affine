import comfy
import torch
import torch.nn.functional as F

from ..modules.usdu import run_usdu_pipeline

from .affine_nodes import (
    PATTERN_CHOICES_LIST,
)


class WASUltimateCustomAdvancedAffineNoUpscaleLatent:
    CATEGORY = "latent/sampling"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT", {"tooltip": "Input latents dict with key 'samples'. Supports [B,C,H,W] (images) or [B,C,F,H,W] (video)."}),
                "model": ("MODEL", {"tooltip": "Diffusion model to sample with."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for base sampler (noise)."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. Can be a single float value or a per-step list. Short lists repeat last value."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Base sampler algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "Scheduler for sigma schedule."}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise fraction (<=1)."}),
                "affine_interval": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Apply affine every N steps (1 = every step)."}),
                "max_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.0001, "tooltip": "Scale at schedule peak: 1 + (max_scale-1)*t."}),
                "max_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.0001, "tooltip": "Bias at schedule peak: max_bias*t."}),
                "pattern": (PATTERN_CHOICES_LIST, {"default": "white_noise", "tooltip": "Affine mask pattern."}),
                "affine_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for affine mask generation."}),
                "affine_seed_increment": ("BOOLEAN", {"default": False, "tooltip": "Increment affine seed after each application (temporal)."}),
                "affine_schedule": ("DICT", {"tooltip": "WASAffineScheduleOptions dict."}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1, "tooltip": "Process the LATENT batch in chunks of this size to reduce peak VRAM."}),
            },
            "optional": {
                "noise": ("NOISE", {"tooltip": "Optional world-aligned noise generator for step 0; if omitted, uses Comfy's default prepare_noise."}),
                "external_mask": ("IMAGE", {"tooltip": "Optional external mask to gate affine (adapted to latent space internally)."}),
                "options": ("DICT", {"tooltip": "Base options for Affine (common/full options)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides layered onto 'options'."}),
                "temporal_mode": (["static", "per_frame"], {"default": "static", "tooltip": "Temporal behavior of the affine mask for video latents."}),
                "merge_frames_in_batch": ("BOOLEAN", {"default": True, "tooltip": "If output latent is 5D [B,C,F,H,W], keep frames; concatenation happens along batch automatically."}),
                "deterministic_noise": ("BOOLEAN", {"default": False, "tooltip": "Generate batching-invariant noise locally using the node's seed; ignores NOISE input when enabled."}),
                "global_noise_mode": ("BOOLEAN", {"default": False, "tooltip": "Force local deterministic noise for entire run (ignores NOISE input)."}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enable detailed logging of shapes and progress for debugging."}),
            },
        }

    @classmethod
    def upscale(
        cls,
        latents,
        model,
        positive,
        negative,
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
        batch_size,
        external_mask=None,
        options=None,
        noise_options=None,
        noise=None,
        temporal_mode="static",
        merge_frames_in_batch=True,
        deterministic_noise=False,
        global_noise_mode=False,
        verbose=False,
    ):
        return run_usdu_pipeline(
            latents=latents,
            model=model,
            positive=positive,
            negative=negative,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            noise=noise,
            affine_interval=affine_interval,
            max_scale=max_scale,
            max_bias=max_bias,
            pattern=pattern,
            affine_seed=affine_seed,
            affine_seed_increment=affine_seed_increment,
            affine_schedule=affine_schedule,
            batch_size=batch_size,
            external_mask=external_mask,
            options=options,
            noise_options=noise_options,
            temporal_mode=temporal_mode,
            merge_frames_in_batch=merge_frames_in_batch,
            deterministic_noise=deterministic_noise,
            global_noise_mode=global_noise_mode,
            verbose=verbose,
        )

class WASUltimateCustomAdvancedAffineNoUpscale:
    CATEGORY = "image/upscaling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input IMAGE or VIDEO tensor. Supports [B,H,W,C] or [B,F,H,W,C]."}),
                "model": ("MODEL", {"tooltip": "Diffusion model to sample with."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
                "vae": ("VAE", {"tooltip": "VAE used to encode/decode latents during USDU."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for base sampler (noise)."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. Can be a single float or a per-step list (short lists repeat last value)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Base sampler algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "Scheduler for sigma schedule."}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise fraction (<=1)."}),
            },
            "optional": {
                "noise": ("NOISE", {"tooltip": "Optional world-aligned noise generator for step 0; if omitted, uses Comfy's default prepare_noise."}),
                "external_mask": ("IMAGE", {"tooltip": "Optional external mask to gate affine."}),
                "options": ("DICT", {"tooltip": "Base options for Affine (common/full options)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides layered onto 'options'."}),
                "affine_interval": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Apply affine every N steps (1 = every step)."}),
                "max_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.0001, "tooltip": "Scale at schedule peak: 1 + (max_scale-1)*t."}),
                "max_bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.0001, "tooltip": "Bias at schedule peak: max_bias*t."}),
                "pattern": (PATTERN_CHOICES_LIST, {"default": "white_noise", "tooltip": "Affine mask pattern."}),
                "affine_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for affine mask generation (separate from sampler seed)."}),
                "affine_seed_increment": ("BOOLEAN", {"default": False, "tooltip": "Increment affine seed after each application (temporal)."}),
                "affine_schedule": ("DICT", {"tooltip": "WASAffineScheduleOptions dict; interpreted over total steps with repeat-last behavior."}),
                "tile_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16, "tooltip": "IMAGE-space sampling tile width in pixels. 0 disables IMAGE tiling (single pass)."}),
                "tile_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16, "tooltip": "IMAGE-space sampling tile height in pixels. 0 disables IMAGE tiling (single pass)."}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 2048, "step": 1, "tooltip": "IMAGE-space overlap between sampling tiles (pixels)."}),
                "feather_sigma": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 0.5, "tooltip": "Feathering (Gaussian sigma, in pixels) for seam blending during latent accumulation."}),
                "merge_frames_in_batch": ("BOOLEAN", {"default": True, "tooltip": "If decoded IMAGE is 5D [B,F,H,W,C], merge F into batch for concat."}),
                "tiled_decode": ("BOOLEAN", {"default": False, "tooltip": "Use VAE tiled decode to reduce VRAM spikes for large outputs/video."}),
                "tiled_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Target output tile size (pixels) for VAE tiled decode."}),
                "tiled_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "Output-space overlap (pixels) for tiled decode."}),
                "tiled_temporal_size": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 1, "tooltip": "Temporal window size (frames) for video tiled decode. 0 disables temporal tiling."}),
                "tiled_temporal_overlap": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1, "tooltip": "Temporal overlap (frames) for video tiled decode."}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enable detailed logging for debugging."}),
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
        noise=None,
        external_mask=None,
        options=None,
        noise_options=None,
        affine_interval=1,
        max_scale=1.2,
        max_bias=0.0,
        pattern="white_noise",
        affine_seed=0,
        affine_seed_increment=False,
        affine_schedule=None,
        tile_width=0,
        tile_height=0,
        tile_overlap=64,
        feather_sigma=8.0,
        merge_frames_in_batch=True,
        tiled_decode=False,
        tiled_tile_size=512,
        tiled_overlap=64,
        tiled_temporal_size=64,
        tiled_temporal_overlap=8,
        verbose=False,
    ):
        import torch
        import math
        is_video = image.dim() == 5
        if is_video:
            b, f, h, w, c = image.shape
        else:
            b, h, w, c = image.shape

        # If tiling disabled or video input, fall back to single-pass latent sampling
        if int(tile_width) <= 0 or int(tile_height) <= 0 or is_video:
            with torch.no_grad():
                latent = vae.encode(image)
            lat_dict = latent if isinstance(latent, dict) else {"samples": latent}
            out_lat, = run_usdu_pipeline(
                latents=lat_dict,
                model=model,
                positive=positive,
                negative=negative,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                noise=noise,
                affine_interval=affine_interval,
                max_scale=max_scale,
                max_bias=max_bias,
                pattern=pattern,
                affine_seed=affine_seed,
                affine_seed_increment=affine_seed_increment,
                affine_schedule=affine_schedule if isinstance(affine_schedule, dict) else ({}),
                batch_size=b,
                external_mask=external_mask,
                options=options,
                noise_options=noise_options,
                temporal_mode="static",
                merge_frames_in_batch=merge_frames_in_batch,
                deterministic_noise=False,
                global_noise_mode=False,
                verbose=verbose,
            )
            merged_tensor = out_lat["samples"] if isinstance(out_lat, dict) else out_lat
        else:
            # IMAGE-space sampling tiling path (4D IMAGE only)
            tw = max(1, int(tile_width))
            th = max(1, int(tile_height))
            ov = max(0, int(tile_overlap))
            # Determine latent scale from VAE (fallback 8)
            try:
                s_comp = int(vae.spacial_compression_decode())
            except Exception:
                s_comp = 8
            # Compute latent mosaic dimensions
            H_lat = math.ceil(h / s_comp)
            W_lat = math.ceil(w / s_comp)
            full_lat = None
            full_wts = None
            # Iterate tiles
            for y0 in range(0, h, th - ov if th - ov > 0 else th):
                y1 = min(h, y0 + th)
                if y1 - y0 <= 0:
                    continue
                for x0 in range(0, w, tw - ov if tw - ov > 0 else tw):
                    x1 = min(w, x0 + tw)
                    if x1 - x0 <= 0:
                        continue
                    tile_img = image[:, y0:y1, x0:x1, :]
                    with torch.no_grad():
                        lat_tile = vae.encode(tile_img)
                    lat_tile = lat_tile if isinstance(lat_tile, dict) else {"samples": lat_tile}
                    lat_samples = lat_tile["samples"]
                    # Sample this latent tile
                    out_lat_tile, = run_usdu_pipeline(
                        latents=lat_tile,
                        model=model,
                        positive=positive,
                        negative=negative,
                        seed=seed,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        denoise=denoise,
                        noise=noise,
                        affine_interval=affine_interval,
                        max_scale=max_scale,
                        max_bias=max_bias,
                        pattern=pattern,
                        affine_seed=affine_seed,
                        affine_seed_increment=affine_seed_increment,
                        affine_schedule=affine_schedule if isinstance(affine_schedule, dict) else ({}),
                        batch_size=lat_samples.shape[0],
                        external_mask=external_mask,
                        options=options,
                        noise_options=noise_options,
                        temporal_mode="static",
                        merge_frames_in_batch=False,
                        deterministic_noise=False,
                        global_noise_mode=False,
                        verbose=verbose,
                    )
                    lat_out_tile = out_lat_tile["samples"] if isinstance(out_lat_tile, dict) else out_lat_tile
                    # Allocate mosaic on first tile
                    if isinstance(lat_out_tile, torch.Tensor) and lat_out_tile.dim() == 5:
                        b_tmp, c_tmp, f_tmp, h_tmp, w_tmp = lat_out_tile.shape
                        lat_out_tile = lat_out_tile.view(b_tmp * f_tmp, c_tmp, h_tmp, w_tmp)
                    if full_lat is None:
                        bsz, cL, hL, wL = lat_out_tile.shape
                        full_lat = torch.zeros((bsz, cL, H_lat, W_lat), device=lat_out_tile.device, dtype=lat_out_tile.dtype)
                        full_wts = torch.zeros((1, 1, H_lat, W_lat), device=lat_out_tile.device, dtype=lat_out_tile.dtype)
                    # Compute latent-space coords and clamp
                    oy0 = y0 // s_comp
                    ox0 = x0 // s_comp
                    hLt, wLt = lat_out_tile.shape[2], lat_out_tile.shape[3]
                    oy1 = min(oy0 + hLt, H_lat)
                    ox1 = min(ox0 + wLt, W_lat)
                    thL = oy1 - oy0
                    twL = ox1 - ox0
                    tile_crop = lat_out_tile[:, :, :thL, :twL]
                    # Build feather weights in latent space
                    ovL = max(0, ov // s_comp)
                    ov_h = min(ovL, thL // 2)
                    ov_w = min(ovL, twL // 2)
                    wmask = torch.ones((1, 1, thL, twL), device=tile_crop.device, dtype=tile_crop.dtype)
                    if ov_w > 0:
                        # Left edge
                        ramp = torch.linspace(0, 1, ov_w, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, 1, ov_w)
                        if ox0 > 0:
                            wmask[:, :, :, :ov_w] *= ramp
                        # Right edge
                        if ox1 < W_lat:
                            ramp = torch.linspace(1, 0, ov_w, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, 1, ov_w)
                            wmask[:, :, :, -ov_w:] *= ramp
                    if ov_h > 0:
                        # Top edge
                        ramp = torch.linspace(0, 1, ov_h, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, ov_h, 1)
                        if oy0 > 0:
                            wmask[:, :, :ov_h, :] *= ramp
                        # Bottom edge
                        if oy1 < H_lat:
                            ramp = torch.linspace(1, 0, ov_h, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, ov_h, 1)
                            wmask[:, :, -ov_h:, :] *= ramp
                    # Accumulate
                    full_lat[:, :, oy0:oy1, ox0:ox1] += tile_crop * wmask
                    full_wts[:, :, oy0:oy1, ox0:ox1] += wmask
            merged_tensor = full_lat / (full_wts + 1e-8)
        try:
            t_comp_fn = getattr(vae, "temporal_compression_decode", None)
            require_5d = False
            if callable(t_comp_fn):
                try:
                    require_5d = t_comp_fn() is not None
                except Exception:
                    require_5d = False
            if isinstance(merged_tensor, torch.Tensor):
                if merged_tensor.dim() == 4 and require_5d:
                    merged_tensor = merged_tensor.unsqueeze(2)
                elif merged_tensor.dim() == 5 and not require_5d:
                    b0, c0, f0, h0, w0 = merged_tensor.shape
                    merged_tensor = merged_tensor.reshape(b0 * f0, c0, h0, w0)
            if tiled_decode:
                tile_size = int(tiled_tile_size)
                overlap = int(tiled_overlap)
                temporal_size = int(tiled_temporal_size)
                temporal_overlap = int(tiled_temporal_overlap)
                if tile_size < overlap * 4:
                    overlap = tile_size // 4
                t_comp = vae.temporal_compression_decode()
                if t_comp is not None:
                    temporal_size_adj = max(2, temporal_size // t_comp)
                    temporal_overlap_adj = max(1, min(temporal_size_adj // 2, temporal_overlap // max(1, t_comp)))
                else:
                    temporal_size_adj = None
                    temporal_overlap_adj = None
                s_comp = vae.spacial_compression_decode()
                tile_x = max(1, tile_size // max(1, s_comp))
                tile_y = max(1, tile_size // max(1, s_comp))
                overlap_lat = max(0, overlap // max(1, s_comp))
                with torch.no_grad():
                    images = vae.decode_tiled(merged_tensor,
                                              tile_x=tile_x, tile_y=tile_y, overlap=overlap_lat,
                                              tile_t=temporal_size_adj, overlap_t=temporal_overlap_adj)
                if isinstance(images, torch.Tensor) and images.dim() == 5:
                    out_img = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                else:
                    out_img = images
            else:
                with torch.no_grad():
                    out_img = vae.decode(merged_tensor)
            if merge_frames_in_batch and out_img.dim() == 5:
                b_out, f_out, h_out, w_out, c_out = out_img.shape
                out_img = out_img.view(b_out * f_out, h_out, w_out, c_out)
        except Exception as e:
            raise RuntimeError(f"VAE.decode failed: {e}")

        return (out_img,)


class WASUltimateCustomAdvancedAffineCustomLatent(WASUltimateCustomAdvancedAffineNoUpscaleLatent):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["custom_sampler"] = ("SAMPLER", {"tooltip": "Optional custom SAMPLER; if provided, used instead of sampler_name."})
        base["optional"]["custom_sigmas"] = ("SIGMAS", {"tooltip": "Optional custom SIGMAS; if provided, overrides scheduler/steps/denoise."})
        return base

    @classmethod
    def upscale(
        cls,
        latents,
        model,
        positive,
        negative,
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
        batch_size,
        external_mask=None,
        options=None,
        noise_options=None,
        noise=None,
        temporal_mode="static",
        custom_sampler=None,
        custom_sigmas=None,
        merge_frames_in_batch=True,
        deterministic_noise=False,
        global_noise_mode=False,
        verbose=False,
    ):
        return run_usdu_pipeline(
            latents=latents,
            model=model,
            positive=positive,
            negative=negative,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            noise=noise,
            affine_interval=affine_interval,
            max_scale=max_scale,
            max_bias=max_bias,
            pattern=pattern,
            affine_seed=affine_seed,
            affine_seed_increment=affine_seed_increment,
            affine_schedule=affine_schedule,
            batch_size=batch_size,
            external_mask=external_mask,
            options=options,
            noise_options=noise_options,
            temporal_mode=temporal_mode,
            merge_frames_in_batch=merge_frames_in_batch,
            deterministic_noise=deterministic_noise,
            global_noise_mode=global_noise_mode,
            verbose=verbose,
            custom_sampler=custom_sampler,
            custom_sigmas=custom_sigmas,
        )

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
        noise=None,
        external_mask=None,
        options=None,
        noise_options=None,
        affine_interval=1,
        max_scale=1.2,
        max_bias=0.0,
        pattern="white_noise",
        affine_seed=0,
        affine_seed_increment=False,
        affine_schedule=None,
        tile_width=0,
        tile_height=0,
        tile_overlap=64,
        feather_sigma=8.0,
        merge_frames_in_batch=True,
        tiled_decode=False,
        tiled_tile_size=512,
        tiled_overlap=64,
        tiled_temporal_size=64,
        tiled_temporal_overlap=8,
        verbose=False,
        custom_sampler=None,
        custom_sigmas=None,
    ):
        import torch
        import math
        is_video = image.dim() == 5
        if is_video:
            b, f, h, w, c = image.shape
        else:
            b, h, w, c = image.shape

        # If tiling disabled or video input, fall back to single-pass latent sampling
        if int(tile_width) <= 0 or int(tile_height) <= 0 or is_video:
            with torch.no_grad():
                latent = vae.encode(image)
            lat_dict = latent if isinstance(latent, dict) else {"samples": latent}
            out_lat, = run_usdu_pipeline(
                latents=lat_dict,
                model=model,
                positive=positive,
                negative=negative,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                noise=noise,
                affine_interval=affine_interval,
                max_scale=max_scale,
                max_bias=max_bias,
                pattern=pattern,
                affine_seed=affine_seed,
                affine_seed_increment=affine_seed_increment,
                affine_schedule=affine_schedule if isinstance(affine_schedule, dict) else ({}),
                batch_size=b,
                external_mask=external_mask,
                options=options,
                noise_options=noise_options,
                temporal_mode="static",
                merge_frames_in_batch=merge_frames_in_batch,
                deterministic_noise=False,
                global_noise_mode=False,
                verbose=verbose,
                custom_sampler=custom_sampler,
                custom_sigmas=custom_sigmas,
            )
            merged_tensor = out_lat["samples"] if isinstance(out_lat, dict) else out_lat
        else:
            # IMAGE-space sampling tiling path (4D IMAGE only)
            tw = max(1, int(tile_width))
            th = max(1, int(tile_height))
            ov = max(0, int(tile_overlap))
            # Determine latent scale from VAE (fallback 8)
            try:
                s_comp = int(vae.spacial_compression_decode())
            except Exception:
                s_comp = 8
            # Compute latent mosaic dimensions
            H_lat = math.ceil(h / s_comp)
            W_lat = math.ceil(w / s_comp)
            full_lat = None
            full_wts = None
            # Iterate tiles
            for y0 in range(0, h, th - ov if th - ov > 0 else th):
                y1 = min(h, y0 + th)
                if y1 - y0 <= 0:
                    continue
                for x0 in range(0, w, tw - ov if tw - ov > 0 else tw):
                    x1 = min(w, x0 + tw)
                    if x1 - x0 <= 0:
                        continue
                    tile_img = image[:, y0:y1, x0:x1, :]
                    with torch.no_grad():
                        lat_tile = vae.encode(tile_img)
                    lat_tile = lat_tile if isinstance(lat_tile, dict) else {"samples": lat_tile}
                    lat_samples = lat_tile["samples"]
                    # Sample this latent tile
                    out_lat_tile, = run_usdu_pipeline(
                        latents=lat_tile,
                        model=model,
                        positive=positive,
                        negative=negative,
                        seed=seed,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        denoise=denoise,
                        noise=noise,
                        affine_interval=affine_interval,
                        max_scale=max_scale,
                        max_bias=max_bias,
                        pattern=pattern,
                        affine_seed=affine_seed,
                        affine_seed_increment=affine_seed_increment,
                        affine_schedule=affine_schedule if isinstance(affine_schedule, dict) else ({}),
                        batch_size=lat_samples.shape[0],
                        external_mask=external_mask,
                        options=options,
                        noise_options=noise_options,
                        temporal_mode="static",
                        merge_frames_in_batch=False,
                        deterministic_noise=False,
                        global_noise_mode=False,
                        verbose=verbose,
                        custom_sampler=custom_sampler,
                        custom_sigmas=custom_sigmas,
                    )
                    lat_out_tile = out_lat_tile["samples"] if isinstance(out_lat_tile, dict) else out_lat_tile
                    # Allocate mosaic on first tile
                    # Flatten potential temporal dimension into batch if present
                    if isinstance(lat_out_tile, torch.Tensor) and lat_out_tile.dim() == 5:
                        b_tmp, c_tmp, f_tmp, h_tmp, w_tmp = lat_out_tile.shape
                        lat_out_tile = lat_out_tile.view(b_tmp * f_tmp, c_tmp, h_tmp, w_tmp)
                    if full_lat is None:
                        bsz, cL, hL, wL = lat_out_tile.shape
                        full_lat = torch.zeros((bsz, cL, H_lat, W_lat), device=lat_out_tile.device, dtype=lat_out_tile.dtype)
                        full_wts = torch.zeros((1, 1, H_lat, W_lat), device=lat_out_tile.device, dtype=lat_out_tile.dtype)
                    # Compute latent-space coords and clamp
                    oy0 = y0 // s_comp
                    ox0 = x0 // s_comp
                    hLt, wLt = lat_out_tile.shape[2], lat_out_tile.shape[3]
                    oy1 = min(oy0 + hLt, H_lat)
                    ox1 = min(ox0 + wLt, W_lat)
                    thL = oy1 - oy0
                    twL = ox1 - ox0
                    tile_crop = lat_out_tile[:, :, :thL, :twL]
                    # Build feather weights in latent space
                    ovL = max(0, ov // s_comp)
                    ov_h = min(ovL, thL // 2)
                    ov_w = min(ovL, twL // 2)
                    wmask = torch.ones((1, 1, thL, twL), device=tile_crop.device, dtype=tile_crop.dtype)
                    if ov_w > 0:
                        # Left edge
                        ramp = torch.linspace(0, 1, ov_w, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, 1, ov_w)
                        if ox0 > 0:
                            wmask[:, :, :, :ov_w] *= ramp
                        # Right edge
                        if ox1 < W_lat:
                            ramp = torch.linspace(1, 0, ov_w, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, 1, ov_w)
                            wmask[:, :, :, -ov_w:] *= ramp
                    if ov_h > 0:
                        # Top edge
                        ramp = torch.linspace(0, 1, ov_h, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, ov_h, 1)
                        if oy0 > 0:
                            wmask[:, :, :ov_h, :] *= ramp
                        # Bottom edge
                        if oy1 < H_lat:
                            ramp = torch.linspace(1, 0, ov_h, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, ov_h, 1)
                            wmask[:, :, -ov_h:, :] *= ramp
                    # Accumulate
                    full_lat[:, :, oy0:oy1, ox0:ox1] += tile_crop * wmask
                    full_wts[:, :, oy0:oy1, ox0:ox1] += wmask
            merged_tensor = full_lat / (full_wts + 1e-8)

        # Decode merged latent tensor
        try:
            t_comp_fn = getattr(vae, "temporal_compression_decode", None)
            require_5d = False
            if callable(t_comp_fn):
                try:
                    require_5d = t_comp_fn() is not None
                except Exception:
                    require_5d = False
            if isinstance(merged_tensor, torch.Tensor):
                if merged_tensor.dim() == 4 and require_5d:
                    merged_tensor = merged_tensor.unsqueeze(2)
                elif merged_tensor.dim() == 5 and not require_5d:
                    b0, c0, f0, h0, w0 = merged_tensor.shape
                    merged_tensor = merged_tensor.reshape(b0 * f0, c0, h0, w0)
            if tiled_decode:
                tile_size = int(tiled_tile_size)
                overlap = int(tiled_overlap)
                temporal_size = int(tiled_temporal_size)
                temporal_overlap = int(tiled_temporal_overlap)
                if tile_size < overlap * 4:
                    overlap = tile_size // 4
                t_comp = vae.temporal_compression_decode()
                if t_comp is not None:
                    temporal_size_adj = max(2, temporal_size // t_comp)
                    temporal_overlap_adj = max(1, min(temporal_size_adj // 2, temporal_overlap // max(1, t_comp)))
                else:
                    temporal_size_adj = None
                    temporal_overlap_adj = None
                s_comp = vae.spacial_compression_decode()
                tile_x = max(1, tile_size // max(1, s_comp))
                tile_y = max(1, tile_size // max(1, s_comp))
                overlap_lat = max(0, overlap // max(1, s_comp))
                with torch.no_grad():
                    images = vae.decode_tiled(merged_tensor,
                                              tile_x=tile_x, tile_y=tile_y, overlap=overlap_lat,
                                              tile_t=temporal_size_adj, overlap_t=temporal_overlap_adj)
                if isinstance(images, torch.Tensor) and images.dim() == 5:
                    out_img = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                else:
                    out_img = images
            else:
                with torch.no_grad():
                    out_img = vae.decode(merged_tensor)
            if merge_frames_in_batch and out_img.dim() == 5:
                b_out, f_out, h_out, w_out, c_out = out_img.shape
                out_img = out_img.view(b_out * f_out, h_out, w_out, c_out)
        except Exception as e:
            raise RuntimeError(f"VAE.decode failed: {e}")

        return (out_img,)


class WASUltimateCustomAdvancedAffine:
    CATEGORY = "image/upscaling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input IMAGE or VIDEO tensor. Supports [B,H,W,C] or [B,F,H,W,C]."}),
                "model": ("MODEL", {"tooltip": "Diffusion model to sample with."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
                "vae": ("VAE", {"tooltip": "VAE used to encode/decode latents during USDU."}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional image-space upscaler model (e.g., ESRGAN/RealESRGAN)."}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "tooltip": "Image-space upscale factor before USDU."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for base sampler (noise)."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. Can be a single float or a per-step list (short lists repeat last value)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Base sampler algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "Scheduler for sigma schedule."}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise fraction (<=1)."}),
                "tiled_decode": ("BOOLEAN", {"default": False, "tooltip": "Use VAE tiled decode to reduce VRAM spikes for large outputs/video."}),
            },
            "optional": {
                "noise": ("NOISE", {"tooltip": "Optional world-aligned noise generator for step 0; if omitted, uses Comfy's default prepare_noise."}),
                "external_mask": ("IMAGE", {"tooltip": "Optional external mask to gate affine."}),
                "options": ("DICT", {"tooltip": "Base options for Affine (common/full options)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides layered onto 'options'."}),
                "temporal_mode": (["static", "per_frame"], {"default": "static", "tooltip": "Temporal behavior of the affine mask."}),
                "merge_frames_in_batch": ("BOOLEAN", {"default": True, "tooltip": "If decoded IMAGE is 5D [B,F,H,W,C], merge F into batch for concat."}),
                "deterministic_noise": ("BOOLEAN", {"default": False, "tooltip": "Generate batching-invariant noise locally using the node's seed; ignores NOISE input when enabled."}),
                "global_noise_mode": ("BOOLEAN", {"default": False, "tooltip": "Force local deterministic noise for entire run (ignores NOISE input)."}),
                "tiled_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Target output tile size (pixels) for VAE tiled decode."}),
                "tiled_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "Output-space overlap (pixels) for tiled decode."}),
                "tiled_temporal_size": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 1, "tooltip": "Temporal window size (frames) for video tiled decode. 0 disables temporal tiling."}),
                "tiled_temporal_overlap": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1, "tooltip": "Temporal overlap (frames) for video tiled decode."}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enable detailed logging for debugging."}),
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
        upscale_model,
        upscale_factor,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        tiled_decode,
        affine_interval,
        max_scale,
        max_bias,
        pattern,
        affine_seed,
        affine_seed_increment,
        affine_schedule,
        batch_size,
        external_mask=None,
        options=None,
        noise_options=None,
        noise=None,
        temporal_mode="static",
        merge_frames_in_batch=True,
        deterministic_noise=False,
        global_noise_mode=False,
        tiled_tile_size=512,
        tiled_overlap=64,
        tiled_temporal_size=64,
        tiled_temporal_overlap=8,
        verbose=False,
    ):
        is_video = image.dim() == 5
        if is_video:
            b, f, h, w, c = image.shape
            new_h = int(h * upscale_factor)
            new_w = int(w * upscale_factor)
            try:
                from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
                upscaler = ImageUpscaleWithModel()
                upscaled_frames = []
                for frame_idx in range(f):
                    frame = image[:, frame_idx]
                    model_up = upscaler.upscale(upscale_model, frame)[0]
                    if model_up.shape[1] != new_h or model_up.shape[2] != new_w:
                        t = model_up.permute(0, 3, 1, 2)
                        t = F.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)
                        model_up = t.permute(0, 2, 3, 1)
                    upscaled_frames.append(model_up)
                upscaled_image = torch.stack(upscaled_frames, dim=1)
            except Exception:
                img_reshaped = image.view(b * f, h, w, c)
                t = img_reshaped.permute(0, 3, 1, 2)
                t = F.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)
                upscaled_image = t.permute(0, 2, 3, 1).view(b, f, new_h, new_w, c)
        else:
            b, h, w, c = image.shape
            new_h = int(h * upscale_factor)
            new_w = int(w * upscale_factor)
            try:
                from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
                upscaler = ImageUpscaleWithModel()
                model_up = upscaler.upscale(upscale_model, image)[0]
                if model_up.shape[1] != new_h or model_up.shape[2] != new_w:
                    t = model_up.permute(0, 3, 1, 2)
                    t = F.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    model_up = t.permute(0, 2, 3, 1)
                upscaled_image = model_up
            except Exception:
                t = image.permute(0, 3, 1, 2)
                t = F.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)
                upscaled_image = t.permute(0, 2, 3, 1)

        if verbose:
            print(f"[WAS Affine][Ultimate] Upscaled IMAGE to {new_h}x{new_w}")

        with torch.no_grad():
            latent = vae.encode(upscaled_image)
        lat_dict = latent if isinstance(latent, dict) else {"samples": latent}

        out_lat, = run_usdu_pipeline(
            latents=lat_dict,
            model=model,
            positive=positive,
            negative=negative,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            noise=noise,
            affine_interval=affine_interval,
            max_scale=max_scale,
            max_bias=max_bias,
            pattern=pattern,
            affine_seed=affine_seed,
            affine_seed_increment=affine_seed_increment,
            affine_schedule=affine_schedule,
            batch_size=batch_size,
            external_mask=external_mask,
            options=options,
            noise_options=noise_options,
            temporal_mode=temporal_mode,
            merge_frames_in_batch=merge_frames_in_batch,
            deterministic_noise=deterministic_noise,
            global_noise_mode=global_noise_mode,
            verbose=verbose,
        )

        merged_tensor = out_lat["samples"] if isinstance(out_lat, dict) else out_lat
        if verbose:
            print(f"[WAS Affine][Ultimate] Decoding output latents shape: {tuple(merged_tensor.shape)}")

        try:
            if tiled_decode:
                tile_size = int(tiled_tile_size)
                overlap = int(tiled_overlap)
                temporal_size = int(tiled_temporal_size)
                temporal_overlap = int(tiled_temporal_overlap)
                if tile_size < overlap * 4:
                    overlap = tile_size // 4
                if temporal_size < temporal_overlap * 2:
                    temporal_overlap = max(1, temporal_overlap // 2)
                t_comp = vae.temporal_compression_decode()
                if t_comp is not None:
                    temporal_size_adj = max(2, temporal_size // t_comp)
                    temporal_overlap_adj = max(1, min(temporal_size_adj // 2, temporal_overlap // max(1, t_comp)))
                else:
                    temporal_size_adj = None
                    temporal_overlap_adj = None
                s_comp = vae.spacial_compression_decode()
                tile_x = max(1, tile_size // max(1, s_comp))
                tile_y = max(1, tile_size // max(1, s_comp))
                overlap_lat = max(0, overlap // max(1, s_comp))
                with torch.no_grad():
                    images = vae.decode_tiled(merged_tensor,
                                              tile_x=tile_x, tile_y=tile_y, overlap=overlap_lat,
                                              tile_t=temporal_size_adj, overlap_t=temporal_overlap_adj)
                if isinstance(images, torch.Tensor) and images.dim() == 5:
                    out_img = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                else:
                    out_img = images
            else:
                with torch.no_grad():
                    out_img = vae.decode(merged_tensor)
            if merge_frames_in_batch and out_img.dim() == 5:
                b_out, f_out, h_out, w_out, c_out = out_img.shape
                out_img = out_img.view(b_out * f_out, h_out, w_out, c_out)
        except Exception as e:
            raise RuntimeError(f"VAE.decode failed: {e}")

        return (out_img,)


NODE_CLASS_MAPPINGS = {
    "WASUltimateCustomAdvancedAffineNoUpscaleLatent": WASUltimateCustomAdvancedAffineNoUpscaleLatent,
    "WASUltimateCustomAdvancedAffineCustomLatent": WASUltimateCustomAdvancedAffineCustomLatent,
    "WASUltimateCustomAdvancedAffineNoUpscale": WASUltimateCustomAdvancedAffineNoUpscale,
    "WASUltimateCustomAdvancedAffineCustom": WASUltimateCustomAdvancedAffineCustom,
    "WASUltimateCustomAdvancedAffine": WASUltimateCustomAdvancedAffine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WASUltimateCustomAdvancedAffineNoUpscaleLatent": "Ultimate Affine KSampler [Latent] (No Upscale)",
    "WASUltimateCustomAdvancedAffineCustomLatent": "Ultimate Affine KSampler [Latent] (Custom)",
    "WASUltimateCustomAdvancedAffineNoUpscale": "Ultimate Affine KSampler (No Upscale)",
    "WASUltimateCustomAdvancedAffineCustom": "Ultimate Affine KSampler (Custom)",
    "WASUltimateCustomAdvancedAffine": "Ultimate Affine KSampler",
}