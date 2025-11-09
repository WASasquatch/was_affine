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
    affine_interval=1,
    max_scale=1.2,
    max_bias=0.0,
    pattern="white_noise",
    affine_seed=0,
    affine_seed_increment=False,
    affine_schedule=None,
    external_mask=None,
    options=None,
    noise_options=None,
    batch_size=0,
    verbose=False,
):
    """Run USDU pipeline with affine sampling.
    
    Args:
        batch_size: If > 0, splits the batch dimension into chunks of this size
                   for memory-efficient processing. 0 = process all at once.
        verbose: Enable progress logging.
    """
    import torch
    from ..nodes.affine_nodes import WASAffineKSamplerAdvanced
    
    # Extract latent samples from dict if needed
    if isinstance(latents, dict):
        lat_samples = latents["samples"]
    else:
        lat_samples = latents
    
    # Determine batch dimension size
    # Latents are [B,C,H,W] for images or [B,C,F,H,W] for video
    total_batch = lat_samples.shape[0]
    
    # If batch_size is 0 or >= total_batch, process all at once
    if batch_size <= 0 or batch_size >= total_batch:
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
    
    # Batch processing: split into chunks
    if verbose:
        print(f"[WAS Affine][USDU] Processing {total_batch} batches in chunks of {batch_size}")
    
    batch_results = []
    for batch_start in range(0, total_batch, batch_size):
        batch_end = min(batch_start + batch_size, total_batch)
        
        if verbose:
            print(f"[WAS Affine][USDU] Processing batch {batch_start}:{batch_end} of {total_batch}")
        
        # Extract batch chunk
        batch_chunk = lat_samples[batch_start:batch_end]
        batch_dict = {"samples": batch_chunk}
        
        # Process this batch chunk
        result = WASAffineKSamplerAdvanced.sample(
            model=model,
            positive=positive,
            negative=negative,
            latent_image=batch_dict,
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
        
        # Extract result
        if isinstance(result, tuple) and len(result) >= 1:
            result = result[0]
        
        if isinstance(result, dict):
            batch_results.append(result["samples"])
        else:
            batch_results.append(result)
    
    # Concatenate all batch results along batch dimension
    merged_samples = torch.cat(batch_results, dim=0)
    
    if verbose:
        print(f"[WAS Affine][USDU] Batch processing complete. Final shape: {tuple(merged_samples.shape)}")
    
    return ({"samples": merged_samples},)
