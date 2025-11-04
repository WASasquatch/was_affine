import comfy


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
    affine_interval=1,
    max_scale=1.2,
    max_bias=0.0,
    pattern="white_noise",
    affine_seed=0,
    affine_seed_increment=False,
    affine_schedule=None,
    batch_size=16,
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
    """Run USDU pipeline with affine sampling."""
    from ..nodes.affine_nodes import WASAffineKSamplerAdvanced
    
    if custom_sampler is not None and custom_sigmas is not None:
        pass
    
    result = WASAffineKSamplerAdvanced.sample(
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latents,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        affine_interval=affine_interval,
        add_noise=True,
        max_scale=max_scale,
        max_bias=max_bias,
        pattern=pattern,
        affine_seed=affine_seed,
        affine_seed_increment=affine_seed_increment,
        affine_schedule=affine_schedule if affine_schedule else {},
        external_mask=external_mask,
        options=options,
        noise_options=noise_options,
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise=False,
        merge_inactive_steps=True,
    )
    
    if isinstance(result, tuple) and len(result) >= 1:
        return (result[0],)
    return (result,)
