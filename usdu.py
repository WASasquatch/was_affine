import comfy

_SEAM_FIX_CHOICES = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]

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
