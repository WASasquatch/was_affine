import torch

import comfy

from .utils import get_cfg_for_step
from ..nodes.affine_nodes import WASAffineCustomAdvanced

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

def run_usdu_pipeline(
    *,
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
    noise=None,
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
    verbose=False,
    custom_sampler=None,
    custom_sigmas=None,
):
    """
    Run the USDU affine pipeline.
    """
    def dbg(msg):
        if verbose:
            print(f"[WAS Affine][USDU] {msg}")

    x_in = latents["samples"] if isinstance(latents, dict) else latents
    dbg(f"Input LATENT shape: {tuple(x_in.shape)}")

    try:
        g = comfy.samplers.CFGGuider(model)
        g.set_conds(positive, negative)
        cfg_value = get_cfg_for_step(cfg, 0, steps)
        g.set_cfg(cfg_value)
        sampler_obj = custom_sampler if custom_sampler is not None else comfy.samplers.sampler_object(sampler_name)
        sigmas = custom_sigmas if custom_sigmas is not None else _build_sigmas_from_model(model, scheduler, int(steps), float(denoise))
    except Exception as e:
        raise RuntimeError(f"Failed to prepare sampler: {e}")

    final_latents = []
    total = int(x_in.shape[0])
    bs = max(1, int(batch_size))
    dbg(f"Running USDU latent pipeline with batch_size={bs} on total={total}")

    for b0 in range(0, total, bs):
        b1 = min(total, b0 + bs)
        x_full = x_in[b0:b1]
        latent = {"samples": x_full}
        dbg(f"Batch {b0}:{b1} latent shape: {tuple(x_full.shape)} | device: {x_full.device} | dtype: {x_full.dtype}")

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

        # Noise for this batch
        if deterministic_noise or global_noise_mode:
            dbg("Deterministic noise generation enabled (batching-invariant)")
            gen = torch.Generator(device=x_full.device)
            bsz = x_full.shape[0]
            if x_full.dim() == 5:
                _, cL, fL, hL, wL = x_full.shape
                full_noise = torch.empty_like(x_full)
                for i in range(bsz):
                    per_seed = int(seed) + (b0 + i) * 1000003
                    gen.manual_seed(per_seed)
                    n = torch.randn((1, cL, 1, hL, wL), device=x_full.device, dtype=x_full.dtype, generator=gen)
                    full_noise[i:i+1] = n.expand(1, cL, fL, hL, wL)
            else:
                _, cL, hL, wL = x_full.shape
                full_noise = torch.empty_like(x_full)
                for i in range(bsz):
                    per_seed = int(seed) + (b0 + i) * 1000003
                    gen.manual_seed(per_seed)
                    n = torch.randn((1, cL, hL, wL), device=x_full.device, dtype=x_full.dtype, generator=gen)
                    full_noise[i:i+1] = n
        else:
            try:
                if noise is not None:
                    full_noise = noise.generate_noise(latent)
                else:
                    import comfy.sample as _comfy_sample
                    batch_inds = latent.get("batch_index") if isinstance(latent, dict) else None
                    base = _comfy_sample.prepare_noise(x_full, int(seed), batch_inds)
                    full_noise = base.to(device=x_full.device, dtype=x_full.dtype)
                if isinstance(full_noise, torch.Tensor):
                    full_noise = full_noise.to(device=x_full.device, dtype=x_full.dtype)
                else:
                    full_noise = torch.randn_like(x_full)
            except Exception:
                full_noise = torch.randn_like(x_full)
        dbg(f"Noise shape: {tuple(full_noise.shape)}")

        class _FullNoise:
            def __init__(self, base):
                self.base = base
            def generate_noise(self, input_latent):
                return self.base

        tnoise = _FullNoise(full_noise)
        with torch.no_grad():
            out_lat_full, _ = WASAffineCustomAdvanced.sample(
                tnoise,
                g,
                sampler_obj,
                sigmas,
                {"samples": x_full},
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
        merged = out_lat_full
        merged_tensor = merged["samples"] if isinstance(merged, dict) else merged
        dbg(f"Merged latent shape (batch result): {tuple(merged_tensor.shape)}")
        if merge_frames_in_batch and merged_tensor.dim() == 5:
            pass
        final_latents.append(merged_tensor)

    if len(final_latents) == 1:
        return ({"samples": final_latents[0]},)
    out_all = torch.cat(final_latents, dim=0)
    dbg(f"Concatenated latent output shape: {tuple(out_all.shape)} from {len(final_latents)} batches")
    return ({"samples": out_all},)
