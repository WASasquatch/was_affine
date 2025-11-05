import comfy

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
                        out_img = vae.decode_tiled(merged_tensor,
                                                  tile_x=tile_x, tile_y=tile_y, overlap=overlap_lat,
                                                  tile_t=temporal_size_adj, overlap_t=temporal_overlap_adj)
                else:
                    with torch.no_grad():
                        out_img = vae.decode(merged_tensor)
                
                if out_img.dim() == 5:
                    if merge_frames_in_batch:
                        b_out, f_out, h_out, w_out, c_out = out_img.shape
                        out_img = out_img.view(b_out * f_out, h_out, w_out, c_out)
            except Exception as e:
                raise RuntimeError(f"VAE.decode failed: {e}")
        else:
            tw = max(1, int(tile_width))
            th = max(1, int(tile_height))
            ov = max(0, int(tile_overlap))
            
            full_img = None
            full_wts = None
            tile_count = 0
            for y0 in range(0, h, th - ov if th - ov > 0 else th):
                y1 = min(h, y0 + th)
                if y1 - y0 <= 0:
                    continue
                for x0 in range(0, w, tw - ov if tw - ov > 0 else tw):
                    x1 = min(w, x0 + tw)
                    if x1 - x0 <= 0:
                        continue
                    
                    tile_count += 1
                    if verbose:
                        print(f"[WAS Affine][USDU Tiling] Processing tile {tile_count} at ({y0}:{y1}, {x0}:{x1})")
                    
                    tile_img = image[:, y0:y1, x0:x1, :]
                    
                    with torch.no_grad():
                        lat_tile = vae.encode(tile_img)
                    lat_tile = lat_tile if isinstance(lat_tile, dict) else {"samples": lat_tile}
                    lat_samples = lat_tile["samples"]
                    
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
                    
                    with torch.no_grad():
                        if tiled_decode:
                            tile_size = int(tiled_tile_size)
                            overlap = int(tiled_overlap)
                            if tile_size < overlap * 4:
                                overlap = tile_size // 4
                            try:
                                s_comp = int(vae.spacial_compression_decode())
                            except Exception:
                                s_comp = 8
                            tile_x = max(1, tile_size // max(1, s_comp))
                            tile_y = max(1, tile_size // max(1, s_comp))
                            overlap_lat = max(0, overlap // max(1, s_comp))
                            decoded_tile = vae.decode_tiled(lat_out_tile,
                                                           tile_x=tile_x, tile_y=tile_y, overlap=overlap_lat)
                        else:
                            decoded_tile = vae.decode(lat_out_tile)
                    
                    if decoded_tile.dim() == 5:
                        b_tmp, f_tmp, h_tmp, w_tmp, c_tmp = decoded_tile.shape
                        decoded_tile = decoded_tile.view(b_tmp * f_tmp, h_tmp, w_tmp, c_tmp)
                    
                    if full_img is None:
                        bsz, h_out, w_out, c_out = decoded_tile.shape
                        full_img = torch.zeros((bsz, h, w, c_out), device=decoded_tile.device, dtype=decoded_tile.dtype)
                        full_wts = torch.zeros((1, h, w, 1), device=decoded_tile.device, dtype=decoded_tile.dtype)
                    
                    th_dec, tw_dec = decoded_tile.shape[1], decoded_tile.shape[2]
                    
                    oy1 = min(y0 + th_dec, h)
                    ox1 = min(x0 + tw_dec, w)
                    th_out = oy1 - y0
                    tw_out = ox1 - x0
                    
                    tile_crop = decoded_tile[:, :th_out, :tw_out, :]
                    
                    ov_h = min(ov, th_out // 2)
                    ov_w = min(ov, tw_out // 2)
                    wmask = torch.ones((1, th_out, tw_out, 1), device=tile_crop.device, dtype=tile_crop.dtype)
                    
                    if ov_w > 0:
                        if x0 > 0:
                            ramp = torch.linspace(0, 1, ov_w, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, ov_w, 1)
                            wmask[:, :, :ov_w, :] *= ramp
                        if ox1 < w:
                            ramp = torch.linspace(1, 0, ov_w, device=tile_crop.device, dtype=tile_crop.dtype).view(1, 1, ov_w, 1)
                            wmask[:, :, -ov_w:, :] *= ramp
                    
                    if ov_h > 0:
                        if y0 > 0:
                            ramp = torch.linspace(0, 1, ov_h, device=tile_crop.device, dtype=tile_crop.dtype).view(1, ov_h, 1, 1)
                            wmask[:, :ov_h, :, :] *= ramp
                        if oy1 < h:
                            ramp = torch.linspace(1, 0, ov_h, device=tile_crop.device, dtype=tile_crop.dtype).view(1, ov_h, 1, 1)
                            wmask[:, -ov_h:, :, :] *= ramp
                    
                    full_img[:, y0:oy1, x0:ox1, :] += tile_crop * wmask
                    full_wts[:, y0:oy1, x0:ox1, :] += wmask
            
            out_img = full_img / (full_wts + 1e-8)
            
            if verbose:
                print(f"[WAS Affine][USDU Tiling] Completed {tile_count} tiles, output shape: {tuple(out_img.shape)}")

        return (out_img,)


NODE_CLASS_MAPPINGS = {
    "WASUltimateCustomAdvancedAffineNoUpscaleLatent": WASUltimateCustomAdvancedAffineNoUpscaleLatent,
    "WASUltimateCustomAdvancedAffineNoUpscale": WASUltimateCustomAdvancedAffineNoUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WASUltimateCustomAdvancedAffineNoUpscaleLatent": "Ultimate Affine KSampler [Latent]",
    "WASUltimateCustomAdvancedAffineNoUpscale": "Ultimate Affine KSampler",
}