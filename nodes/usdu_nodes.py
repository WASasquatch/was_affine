import comfy
import torch

from ..modules.utils import get_cfg_for_step
from ..modules.usdu import _build_sigmas_from_model

from .affine_nodes import (
    PATTERN_CHOICES_LIST,
    WASAffineCustomAdvanced,
)


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
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Tile width in pixels."}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Tile height in pixels."}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "Mask blur (px) for tile feathering."}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8, "tooltip": "Tile padding/overlap size (px)."}),
                # Misc
                "tiled_decode": ("BOOLEAN", {"default": False, "tooltip": "Use ComfyUI's built-in VAE tiled decode (compression-aware). Helps reduce VRAM spikes on large images or video VAEs."}),
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
                # Batching
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1, "tooltip": "Process the IMAGE batch in chunks of this size to reduce peak VRAM. Split along the batch dimension and run encode→sample→decode sequentially."}),
            },
            "optional": {
                "external_mask": ("IMAGE", {"tooltip": "Optional external mask to gate affine."}),
                "options": ("DICT", {"tooltip": "Base options for Affine (common/full options)."}),
                "noise_options": ("DICT", {"tooltip": "Pattern-specific overrides layered onto 'options'."}),
                "temporal_mode": (["static", "per_frame"], {"default": "static", "tooltip": "Temporal behavior of the affine mask."}),
                "merge_frames_in_batch": ("BOOLEAN", {"default": True, "tooltip": "If decoded IMAGE is 5D [B,F,H,W,C], merge F into batch so batches with different frame counts can be concatenated safely."}),
                "deterministic_noise": ("BOOLEAN", {"default": False, "tooltip": "Generate batching-invariant noise locally using the node's seed; ignores NOISE input when enabled."}),
                "global_noise_mode": ("BOOLEAN", {"default": False, "tooltip": "Force local deterministic noise for entire run (ignores NOISE input). Ensures batching-invariant noise."}),
                "overlap_blend_count": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Crossfade this many images between consecutive batches to hide boundary shifts. 0 disables."}),
                "overlap_blend_curve": (["cosine", "linear"], {"default": "cosine", "tooltip": "Curve for crossfade between batches."}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enable detailed logging of shapes, tiles, and progress for debugging."}),
                # Tiled decode tunables
                "tiled_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Target output tile size in pixels for VAE tiled decode (pre-compression)."}),
                "tiled_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "Output-space overlap in pixels for tiled decode (pre-compression)."}),
                "tiled_temporal_size": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 1, "tooltip": "Temporal window size for video tiled decode (frames, pre-compression). 0 disables temporal tiling."}),
                "tiled_temporal_overlap": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1, "tooltip": "Temporal overlap for video tiled decode (frames, pre-compression)."}),
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
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        tiled_decode,
        noise,
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
        temporal_mode="static",
        merge_frames_in_batch=True,
        deterministic_noise=False,
        global_noise_mode=False,
        overlap_blend_count=0,
        overlap_blend_curve="cosine",
        tiled_tile_size=512,
        tiled_overlap=64,
        tiled_temporal_size=64,
        tiled_temporal_overlap=8,
        verbose=False,
    ):
        def dbg(msg):
            print(f"[WAS Affine][NoUpscale] {msg}")

        dbg(f"Input IMAGE shape: {tuple(image.shape)}")
        # Helper: process one image batch through encode -> tiled sampling -> decode
        def _process_image_batch(img_batch):
            try:
                latent_local = vae.encode(img_batch)
            except Exception as e:
                raise RuntimeError(f"VAE.encode failed: {e}")
            return latent_local

        try:
            g = comfy.samplers.CFGGuider(model)
            g.set_conds(positive, negative)
            cfg_value = get_cfg_for_step(cfg, 0, steps)
            g.set_cfg(cfg_value)
            sampler_obj = comfy.samplers.sampler_object(sampler_name)
            sigmas = _build_sigmas_from_model(model, scheduler, int(steps), float(denoise))
        except Exception as e:
            raise RuntimeError(f"Failed to prepare sampler: {e}")

        final_images = []
        ref_out_shape = None
        total = image.shape[0]
        bs = max(1, int(batch_size))
        for b0 in range(0, total, bs):
            b1 = min(total, b0 + bs)
            img_batch = image[b0:b1]
            dbg(f"Batch {b0}:{b1} img_batch shape: {tuple(img_batch.shape)}")
            latent = _process_image_batch(img_batch)
            x_full = latent["samples"] if isinstance(latent, dict) else latent
            dbg(f"Encoded latent shape: {tuple(x_full.shape)} | device: {x_full.device} | dtype: {x_full.dtype}")

            # Align latent device with model_patcher device, matching base node behavior
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
            dbg(f"Latent aligned to device: {x_full.device}")
            
            if len(x_full.shape) == 5:
                b, cL, fL, hL, wL = x_full.shape
                is_video = True
            elif len(x_full.shape) == 4:
                b, cL, hL, wL = x_full.shape
                is_video = False
            else:
                raise ValueError(f"Unsupported latent shape: {x_full.shape}. Expected 4D or 5D tensor.")
            dbg(f"is_video={is_video}, latent L dims: hL={hL}, wL={wL}{', fL='+str(fL) if is_video else ''}")
        
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
            dbg(f"Pixel dims: H={Hpx if 'Hpx' in locals() else 'N/A'}, W={Wpx if 'Wpx' in locals() else 'N/A'}, scale_h={scale_h:.3f}, scale_w={scale_w:.3f}")

            def to_lat(v_px: int, scale: float):
                return max(1, int(round(float(v_px) / scale)))

            twL = to_lat(int(tile_width), scale_w)
            thL = to_lat(int(tile_height), scale_h)
            padL = to_lat(int(tile_padding), (scale_h + scale_w) * 0.5)
            blurL = to_lat(int(mask_blur), (scale_h + scale_w) * 0.5)

            # Generate noise: deterministic local when enabled; else prefer NOISE input, fallback to randn_like
            if deterministic_noise or global_noise_mode:
                dbg("Deterministic noise generation enabled (batching-invariant)")
                gen = torch.Generator(device=x_full.device)
                bsz = x_full.shape[0]
                if is_video:
                    # x_full: [B, C, F, H, W]
                    _, cL, fL, hL, wL = x_full.shape
                    full_noise = torch.empty_like(x_full)
                    for bi in range(bsz):
                        global_b = b0 + bi
                        for fi in range(fL):
                            per_seed = int(seed) + global_b * 1000003 + fi
                            gen.manual_seed(per_seed)
                            n = torch.randn((1, cL, 1, hL, wL), device=x_full.device, dtype=x_full.dtype, generator=gen)
                            full_noise[bi:bi+1, :, fi:fi+1, :, :] = n
                else:
                    # x_full: [B, C, H, W]
                    _, cL, hL, wL = x_full.shape
                    full_noise = torch.empty_like(x_full)
                    for bi in range(bsz):
                        global_b = b0 + bi
                        per_seed = int(seed) + global_b * 1000003
                        gen.manual_seed(per_seed)
                        n = torch.randn((1, cL, hL, wL), device=x_full.device, dtype=x_full.dtype, generator=gen)
                        full_noise[bi:bi+1, :, :, :] = n
            else:
                try:
                    full_noise = noise.generate_noise(latent)
                    if isinstance(full_noise, torch.Tensor):
                        full_noise = full_noise.to(device=x_full.device, dtype=x_full.dtype)
                    else:
                        full_noise = torch.randn_like(x_full)
                except Exception:
                    full_noise = torch.randn_like(x_full)
            dbg(f"Noise shape: {tuple(full_noise.shape)}")

            out_acc = torch.zeros_like(x_full)
            if is_video:
                w_acc = torch.zeros((b, 1, fL, hL, wL), device=x_full.device, dtype=x_full.dtype)
            else:
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
            dbg(f"Tiling L-space: thL={thL}, twL={twL}, padL={padL}, blurL={blurL}, stride_y={stride_y}, stride_x={stride_x}")
            for by in range(0, hL, stride_y):
                for bx in range(0, wL, stride_x):
                    y0 = max(0, by - padL)
                    x0 = max(0, bx - padL)
                    y1 = min(hL, by + thL + padL)
                    x1 = min(wL, bx + twL + padL)
                    if (y1 - y0) <= 1 or (x1 - x0) <= 1:
                        continue
                    if verbose:
                        dbg(f"Tile L-range: y={y0}:{y1} x={x0}:{x1}")

                    if is_video:
                        tile_lat = {"samples": x_full[:, :, :, y0:y1, x0:x1]}
                    else:
                        tile_lat = {"samples": x_full[:, :, y0:y1, x0:x1]}

                    class _TileNoise:
                        def __init__(self, base_noise, y0, y1, x0, x1, is_video):
                            self.base = base_noise
                            self.seed = 0
                            self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
                            self.is_video = is_video

                        def generate_noise(self, input_latent):
                            if self.is_video:
                                return self.base[:, :, :, self.y0:self.y1, self.x0:self.x1]
                            else:
                                return self.base[:, :, self.y0:self.y1, self.x0:self.x1]

                    tnoise = _TileNoise(full_noise, y0, y1, x0, x1, is_video)

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
                    expected_shape = (x_full[:, :, :, y0:y1, x0:x1].shape if is_video else x_full[:, :, y0:y1, x0:x1].shape)
                    dbg(f"tile_out shape: {tuple(tile_out.shape)} expected: {tuple(expected_shape)}")
                    # Safety: ensure tile_out matches expected shape
                    exp = (x_full[:, :, :, y0:y1, x0:x1] if is_video else x_full[:, :, y0:y1, x0:x1])
                    if tile_out.shape != exp.shape:
                        raise RuntimeError(f"Affine sample tile_out shape {tuple(tile_out.shape)} does not match expected {tuple(exp.shape)} at y={y0}:{y1}, x={x0}:{x1}")

                    w2d = feather_mask(hL, wL, y0, y1, x0, x1).unsqueeze(0).unsqueeze(0)
                    if is_video:
                        w2d = w2d.unsqueeze(2)  # Add frame dimension for 5D
                    w2d = w2d.to(device=x_full.device, dtype=x_full.dtype)
                    
                    if out_acc.device != x_full.device:
                        out_acc = out_acc.to(device=x_full.device, dtype=x_full.dtype)
                    if w_acc.device != x_full.device:
                        w_acc = w_acc.to(device=x_full.device, dtype=x_full.dtype)

                    if is_video:
                        out_acc[:, :, :, y0:y1, x0:x1] += tile_out * w2d
                        w_acc[:, :, :, y0:y1, x0:x1] += w2d
                    else:
                        out_acc[:, :, y0:y1, x0:x1] += tile_out * w2d
                        w_acc[:, :, y0:y1, x0:x1] += w2d

            w_safe = torch.where(w_acc > 0, w_acc, torch.ones_like(w_acc))
            merged = out_acc / w_safe
            merged = torch.where(w_acc > 0, merged, x_full)
            dbg(f"Merged latent shape: {tuple(merged.shape)}")

            # Explicitly log before decode to show progress and avoid perceived hang
            dbg(f"Decoding merged latent... shape={tuple(merged.shape)}, device={merged.device}, dtype={merged.dtype}")
            try:
                # Use ComfyUI built-in tiled decode if enabled, else direct decode
                if tiled_decode:
                    tile_size = int(tiled_tile_size)
                    overlap = int(tiled_overlap)
                    temporal_size = int(tiled_temporal_size)
                    temporal_overlap = int(tiled_temporal_overlap)
                    # Enforce sane relationships
                    if tile_size < overlap * 4:
                        overlap = tile_size // 4
                    if temporal_size < temporal_overlap * 2:
                        temporal_overlap = max(1, temporal_overlap // 2)
                    # Adjust for VAE compression
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
                    dbg(f"tiled_decode=True -> using vae.decode_tiled(tile={tile_size}, overlap={overlap}, t_size={temporal_size}, t_ov={temporal_overlap}) -> tile_x={tile_x}, tile_y={tile_y}, overlap_lat={overlap_lat}, tile_t={temporal_size_adj}, overlap_t={temporal_overlap_adj}")
                    images = vae.decode_tiled(merged["samples"] if isinstance(merged, dict) else merged,
                                              tile_x=tile_x, tile_y=tile_y, overlap=overlap_lat,
                                              tile_t=temporal_size_adj, overlap_t=temporal_overlap_adj)
                    # images: [B,H,W,C] or [B,F,H,W,C] -> if 5D, combine batches
                    if isinstance(images, torch.Tensor) and images.dim() == 5:
                        out_img = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    else:
                        out_img = images
                else:
                    out_img = vae.decode(merged)
                dbg(f"Decoded out_img shape: {tuple(out_img.shape)}")
                if merge_frames_in_batch and out_img.dim() == 5:
                    b_out, f_out, h_out, w_out, c_out = out_img.shape
                    out_img = out_img.view(b_out * f_out, h_out, w_out, c_out)
                    dbg(f"Merged frames into batch for concat: now {tuple(out_img.shape)}")
            except Exception as e:
                raise RuntimeError(f"VAE.decode failed: {e}")

            # Cross-batch overlap blending and shape consistency
            if ref_out_shape is None:
                ref_out_shape = tuple(out_img.shape[1:])
            else:
                if tuple(out_img.shape[1:]) != ref_out_shape:
                    raise RuntimeError(
                        f"Decoded IMAGE shape mismatch across batches: expected * x {ref_out_shape}, got * x {tuple(out_img.shape[1:])}. "
                        f"Ensure all inputs in the overall IMAGE batch have identical spatial sizes and channels."
                    )
                if overlap_blend_count and len(final_images) > 0:
                    prev = final_images[-1]
                    curr = out_img
                    k = min(int(overlap_blend_count), prev.shape[0], curr.shape[0])
                    if k > 0:
                        dbg(f"Crossfading at batch join with k={k}, curve={overlap_blend_curve}")
                        if overlap_blend_curve == "cosine":
                            import math
                            alphas = [0.5 - 0.5*math.cos(math.pi*(i+1)/(k+1)) for i in range(k)]
                        else:
                            alphas = [(i+1)/(k+1) for i in range(k)]
                        # Blend last k of prev with first k of curr; keep all frames (no dropping)
                        for i, a in enumerate(alphas):
                            idx_prev = prev.shape[0] - k + i
                            prev[idx_prev] = prev[idx_prev] * (1.0 - a) + curr[i] * a
                        # Keep current batch frames intact to avoid frame loss
            final_images.append(out_img)

        # Concatenate all batch outputs along batch dimension
        if len(final_images) == 1:
            return (final_images[0],)
        out_all = torch.cat(final_images, dim=0)
        dbg(f"Concatenated output shape: {tuple(out_all.shape)} from {len(final_images)} batches")
        return (out_all,)


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
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        tiled_decode,
        noise,
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
        temporal_mode="static",
        custom_sampler=None,
        custom_sigmas=None,
        merge_frames_in_batch=True,
        deterministic_noise=False,
        global_noise_mode=False,
        overlap_blend_count=0,
        overlap_blend_curve="cosine",
        tiled_tile_size=512,
        tiled_overlap=64,
        tiled_temporal_size=64,
        tiled_temporal_overlap=8,
        verbose=False,
    ):
        def dbg(msg):
            print(f"[WAS Affine][Custom] {msg}")
        dbg(f"Input IMAGE shape: {tuple(image.shape)}")
        try:
            g = comfy.samplers.CFGGuider(model)
            g.set_conds(positive, negative)
            cfg_value = get_cfg_for_step(cfg, 0, steps)
            g.set_cfg(cfg_value)
            sampler_obj = custom_sampler if custom_sampler is not None else comfy.samplers.sampler_object(sampler_name)
            sigmas = custom_sigmas if custom_sigmas is not None else _build_sigmas_from_model(model, scheduler, int(steps), float(denoise))
        except Exception as e:
            raise RuntimeError(f"Failed to prepare sampler: {e}")

        final_images = []
        ref_out_shape = None
        total = image.shape[0]
        bs = max(1, int(batch_size))
        for b0 in range(0, total, bs):
            b1 = min(total, b0 + bs)
            img_batch = image[b0:b1]
            dbg(f"Batch {b0}:{b1} img_batch shape: {tuple(img_batch.shape)}")
            try:
                latent = vae.encode(img_batch)
            except Exception as e:
                raise RuntimeError(f"VAE.encode failed: {e}")

            x_full = latent["samples"] if isinstance(latent, dict) else latent
            dbg(f"Encoded latent shape: {tuple(x_full.shape)} | device: {x_full.device} | dtype: {x_full.dtype}")

            if len(x_full.shape) == 5:
                b, cL, fL, hL, wL = x_full.shape
                is_video = True
            elif len(x_full.shape) == 4:
                b, cL, hL, wL = x_full.shape
                is_video = False
            else:
                raise ValueError(f"Unsupported latent shape: {x_full.shape}. Expected 4D or 5D tensor.")

            try:
                if len(img_batch.shape) == 5 and is_video:
                    _, _, Hpx, Wpx, _ = img_batch.shape
                elif len(img_batch.shape) == 4:
                    _, Hpx, Wpx, _ = img_batch.shape
                else:
                    raise ValueError(f"Unsupported image shape: {img_batch.shape}")
                scale_h = max(1.0, float(Hpx) / float(hL))
                scale_w = max(1.0, float(Wpx) / float(wL))
            except Exception:
                scale_h = scale_w = 8.0
            dbg(f"Pixel dims: H={Hpx if 'Hpx' in locals() else 'N/A'}, W={Wpx if 'Wpx' in locals() else 'N/A'}, scale_h={scale_h:.3f}, scale_w={scale_w:.3f}")

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
            if is_video:
                w_acc = torch.zeros((b, 1, fL, hL, wL), device=x_full.device, dtype=x_full.dtype)
            else:
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

                    if is_video:
                        tile_lat = {"samples": x_full[:, :, :, y0:y1, x0:x1]}
                    else:
                        tile_lat = {"samples": x_full[:, :, y0:y1, x0:x1]}

                    class _TileNoise:
                        def __init__(self, base_noise, y0, y1, x0, x1, is_video):
                            self.base = base_noise
                            self.seed = 0
                            self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
                            self.is_video = is_video

                        def generate_noise(self, input_latent):
                            if self.is_video:
                                return self.base[:, :, :, self.y0:self.y1, self.x0:self.x1]
                            else:
                                return self.base[:, :, self.y0:self.y1, self.x0:self.x1]

                    tnoise = _TileNoise(full_noise, y0, y1, x0, x1, is_video)

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
                    if verbose:
                        expected_shape = (x_full[:, :, :, y0:y1, x0:x1].shape if is_video else x_full[:, :, y0:y1, x0:x1].shape)
                        dbg(f"tile_out shape: {tuple(tile_out.shape)} expected: {tuple(expected_shape)}")
                    # Safety: ensure tile_out matches expected shape
                    exp = (x_full[:, :, :, y0:y1, x0:x1] if is_video else x_full[:, :, y0:y1, x0:x1])
                    if tile_out.shape != exp.shape:
                        raise RuntimeError(f"Affine sample tile_out shape {tuple(tile_out.shape)} does not match expected {tuple(exp.shape)} at y={y0}:{y1}, x={x0}:{x1}")

                    w2d = feather_mask(hL, wL, y0, y1, x0, x1).unsqueeze(0).unsqueeze(0)
                    if is_video:
                        w2d = w2d.unsqueeze(2)  # Add frame dimension for 5D
                    w2d = w2d.to(device=x_full.device, dtype=x_full.dtype)

                    if out_acc.device != x_full.device:
                        out_acc = out_acc.to(device=x_full.device, dtype=x_full.dtype)
                    if w_acc.device != x_full.device:
                        w_acc = w_acc.to(device=x_full.device, dtype=x_full.dtype)

                    if is_video:
                        out_acc[:, :, :, y0:y1, x0:x1] += tile_out * w2d
                        w_acc[:, :, :, y0:y1, x0:x1] += w2d
                    else:
                        out_acc[:, :, y0:y1, x0:x1] += tile_out * w2d
                        w_acc[:, :, y0:y1, x0:x1] += w2d

            w_safe = torch.where(w_acc > 0, w_acc, torch.ones_like(w_acc))
            merged = out_acc / w_safe
            merged = torch.where(w_acc > 0, merged, x_full)

            dbg(f"Decoding merged latent... shape={tuple(merged.shape)}, device={merged.device}, dtype={merged.dtype}")
            try:
                # Use ComfyUI built-in tiled decode if enabled, else direct decode
                if tiled_decode:
                    tile_size = 512
                    overlap = 64
                    temporal_size = 64
                    temporal_overlap = 8
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
                    dbg(f"tiled_decode=True -> using vae.decode_tiled(tile={tile_size}, overlap={overlap}, t_size={temporal_size}, t_ov={temporal_overlap}) -> tile_x={tile_x}, tile_y={tile_y}, overlap_lat={overlap_lat}, tile_t={temporal_size_adj}, overlap_t={temporal_overlap_adj}")
                    images = vae.decode_tiled(merged["samples"] if isinstance(merged, dict) else merged,
                                              tile_x=tile_x, tile_y=tile_y, overlap=overlap_lat,
                                              tile_t=temporal_size_adj, overlap_t=temporal_overlap_adj)
                    if isinstance(images, torch.Tensor) and images.dim() == 5:
                        out_img = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    else:
                        out_img = images
                else:
                    out_img = vae.decode(merged)
                dbg(f"Decoded out_img shape: {tuple(out_img.shape)}")
            except Exception as e:
                raise RuntimeError(f"VAE.decode failed: {e}")

            # Cross-batch overlap blending and shape consistency
            if ref_out_shape is None:
                ref_out_shape = tuple(out_img.shape[1:])
            else:
                if tuple(out_img.shape[1:]) != ref_out_shape:
                    raise RuntimeError(
                        f"Decoded IMAGE shape mismatch across batches: expected * x {ref_out_shape}, got * x {tuple(out_img.shape[1:])}. "
                        f"Ensure all inputs in the overall IMAGE batch have identical spatial sizes and channels."
                    )
                if overlap_blend_count and len(final_images) > 0:
                    prev = final_images[-1]
                    curr = out_img
                    k = min(int(overlap_blend_count), prev.shape[0], curr.shape[0])
                    if k > 0:
                        dbg(f"Crossfading at batch join with k={k}, curve={overlap_blend_curve}")
                        if overlap_blend_curve == "cosine":
                            import math
                            alphas = [0.5 - 0.5*math.cos(math.pi*(i+1)/(k+1)) for i in range(k)]
                        else:
                            alphas = [(i+1)/(k+1) for i in range(k)]
                        for i, a in enumerate(alphas):
                            idx_prev = prev.shape[0] - k + i
                            prev[idx_prev] = prev[idx_prev] * (1.0 - a) + curr[i] * a
                        # Keep current batch frames intact to avoid frame loss
            final_images.append(out_img)

        if len(final_images) == 1:
            return (final_images[0],)
        out_all = torch.cat(final_images, dim=0)
        return (out_all,)


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
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        tiled_decode,
        noise,
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
        temporal_mode="static",
        merge_frames_in_batch=True,
        deterministic_noise=False,
        global_noise_mode=False,
        overlap_blend_count=0,
        overlap_blend_curve="cosine",
        tiled_tile_size=512,
        tiled_overlap=64,
        tiled_temporal_size=64,
        tiled_temporal_overlap=8,
        verbose=False,
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
            tile_width,
            tile_height,
            mask_blur,
            tile_padding,
            tiled_decode,
            noise,
            affine_interval,
            max_scale,
            max_bias,
            pattern,
            affine_seed,
            affine_seed_increment,
            affine_schedule,
            batch_size,
            external_mask=external_mask,
            options=options,
            noise_options=noise_options,
            temporal_mode=temporal_mode,
            merge_frames_in_batch=merge_frames_in_batch,
            deterministic_noise=deterministic_noise,
            global_noise_mode=global_noise_mode,
            overlap_blend_count=overlap_blend_count,
            overlap_blend_curve=overlap_blend_curve,
            tiled_tile_size=tiled_tile_size,
            tiled_overlap=tiled_overlap,
            tiled_temporal_size=tiled_temporal_size,
            tiled_temporal_overlap=tiled_temporal_overlap,
            verbose=verbose,
        )


NODE_CLASS_MAPPINGS = {
    "WASUltimateCustomAdvancedAffineNoUpscale": WASUltimateCustomAdvancedAffineNoUpscale,
    "WASUltimateCustomAdvancedAffineCustom": WASUltimateCustomAdvancedAffineCustom,
    "WASUltimateCustomAdvancedAffine": WASUltimateCustomAdvancedAffine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WASUltimateCustomAdvancedAffineNoUpscale": "Affine KSampler (No Upscale)",
    "WASUltimateCustomAdvancedAffineCustom": "Affine KSampler (Custom)",
    "WASUltimateCustomAdvancedAffine": "Affine KSampler",
}