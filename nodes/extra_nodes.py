import torch
import torch.nn.functional as F

from contextlib import nullcontext


import comfy.sample as comfy_sample_mod


class WASWANVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "VAE used to decode the latents. Must correspond to the model that produced the latents. Respects the VAE's device/dtype and internal time/space scaling."}),
                "latents": ("LATENT", {"tooltip": "Latent dict with key 'samples' shaped [B,C,F,H,W]. F=frames (F=1 for images). H/W are latent-space, not pixel-space."}),
                "horizontal_tiles": ("INT", {"default": 1, "min": 1, "max": 12, "tooltip": "Tiles across WIDTH in latent space."}),
                "vertical_tiles": ("INT", {"default": 1, "min": 1, "max": 12, "tooltip": "Tiles across HEIGHT in latent space."}),
                "overlap": ("INT", {"default": 1, "min": 0, "max": 16, "tooltip": "Overlap in LATENT pixels. Output overlap = overlap * VAE scale."}),
                "last_frame_fix": ("BOOLEAN", {"default": False, "tooltip": "Append last latent frame before decode, trim after. For video VAEs."}),
            },
            "optional": {
                "frames_per_pass": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Limit frames per pass (0 = all)."}),
                "batch_per_pass": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Limit batch per pass (0 = all)."}),
                "max_decode_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 400.0, "step": 0.5, "tooltip": "VRAM budget ≈ decH * decW * out_frames * batch / 1e6. 0 = off."}),
                "accumulate_dtype": (["fp32", "fp16", "bf16"], {"default": "fp32", "tooltip": "Accumulator dtype for blending."}),
                "output_dtype": (["auto", "fp16", "bf16", "fp32"], {"default": "auto", "tooltip": "Final NHWC dtype ('auto' = fp16 on CUDA, else fp32)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def _amp_dtype(self):
        if not torch.cuda.is_available():
            return None
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _dtype_from_choice(self, choice, fallback):
        if choice == "fp16":
            return torch.float16
        if choice == "bf16":
            return torch.bfloat16
        if choice == "fp32":
            return torch.float32
        return fallback

    def _format_decoded(self, dec, Bb, out_Fp):
        if dec.dim() == 5:
            if dec.shape[-1] in (1, 3, 4):
                B2, T2, H2, W2, C2 = dec.shape
                return dec, B2, T2, H2, W2, C2
            if dec.shape[2] in (1, 3, 4):
                dec = dec.permute(0, 1, 3, 4, 2).contiguous()
                B2, T2, H2, W2, C2 = dec.shape
                return dec, B2, T2, H2, W2, C2
            raise RuntimeError("Unsupported 5D decode layout")
        if dec.dim() == 4:
            if dec.shape[-1] in (1, 3, 4):
                N, H2, W2, C2 = dec.shape
                if N % out_Fp != 0:
                    raise RuntimeError(f"decode batch {N} not divisible by out_Fp {out_Fp}")
                B2 = N // out_Fp
                dec = dec.view(B2, out_Fp, H2, W2, C2)
                return dec, B2, out_Fp, H2, W2, C2
            if dec.shape[1] in (1, 3, 4):
                N, C2, H2, W2 = dec.shape
                if N % out_Fp != 0:
                    raise RuntimeError(f"decode batch {N} not divisible by out_Fp {out_Fp}")
                B2 = N // out_Fp
                dec = dec.permute(0, 2, 3, 1).contiguous()
                dec = dec.view(B2, out_Fp, H2, W2, C2)
                return dec, B2, out_Fp, H2, W2, C2
            raise RuntimeError(f"Unsupported 4D decode layout {dec.shape}")
        raise RuntimeError(f"Decode returned unsupported rank {dec.dim()}")

    @torch.inference_mode()
    def decode(
        self,
        vae,
        latents,
        horizontal_tiles,
        vertical_tiles,
        overlap,
        last_frame_fix,
        frames_per_pass=0,
        batch_per_pass=0,
        max_decode_megapixels=0.0,
        accumulate_dtype="fp32",
        output_dtype="auto",
    ):
        samples = latents["samples"]
        if last_frame_fix:
            samples = torch.cat([samples, samples[:, :, -1:, :, :]], dim=2)

        B, C, F, H, W = samples.shape
        t_sf, w_sf, h_sf = vae.downscale_index_formula
        out_F = 1 + (F - 1) * t_sf
        out_H, out_W = H * h_sf, W * w_sf

        base_tile_h = (H + (vertical_tiles - 1) * overlap) // vertical_tiles
        base_tile_w = (W + (horizontal_tiles - 1) * overlap) // horizontal_tiles

        device = samples.device
        acc_dtype = self._dtype_from_choice(accumulate_dtype, torch.float32)
        out_dtype_t = self._dtype_from_choice(output_dtype, None)
        if out_dtype_t is None:
            out_dtype_t = torch.float16 if torch.cuda.is_available() else torch.float32

        output = torch.zeros((B, out_F, out_H, out_W, 3), device=device, dtype=acc_dtype)
        weights = torch.zeros((B, out_F, out_H, out_W, 1), device=device, dtype=acc_dtype)

        overlap_out_h = overlap * h_sf
        overlap_out_w = overlap * w_sf

        def _ramp(n, a, b):
            if n <= 0:
                return None
            return torch.linspace(a, b, n, device=device, dtype=acc_dtype)

        h_left = _ramp(overlap_out_w, 0, 1)
        h_right = _ramp(overlap_out_w, 1, 0)
        v_top = _ramp(overlap_out_h, 0, 1)
        v_bot = _ramp(overlap_out_h, 1, 0)

        amp_dtype = self._amp_dtype()
        amp_ctx = nullcontext() if amp_dtype is None else (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if hasattr(torch, "amp") else torch.cuda.amp.autocast(dtype=amp_dtype)
        )

        def plan_pass_sizes():
            nonlocal frames_per_pass, batch_per_pass
            if frames_per_pass <= 0:
                frames_per_pass = F
            if batch_per_pass <= 0:
                batch_per_pass = B
            if max_decode_megapixels <= 0:
                return frames_per_pass, batch_per_pass
            max_oH = min((base_tile_h + overlap) * h_sf, out_H)
            max_oW = min((base_tile_w + overlap) * w_sf, out_W)
            tile_mp = (max_oH * max_oW) / 1_000_000.0
            def out_frames(ff): return 1 + (ff - 1) * t_sf
            ff, bb = frames_per_pass, batch_per_pass
            while ff > 1 or bb > 1:
                if tile_mp * out_frames(ff) * bb <= max_decode_megapixels + 1e-6:
                    break
                if ff >= bb and ff > 1:
                    ff = max(1, ff // 2)
                elif bb > 1:
                    bb = max(1, bb // 2)
                else:
                    break
            return max(ff, 1), max(bb, 1)

        frames_pass, batch_pass = plan_pass_sizes()

        def chunks(n, step):
            i = 0
            while i < n:
                j = min(i + step, n)
                yield i, j
                i = j

        with amp_ctx:
            for b0, b1 in chunks(B, batch_pass):
                for f0, f1 in chunks(F, frames_pass):
                    sub = samples[b0:b1, :, f0:f1, :, :]
                    Bb, _, Fp, _, _ = sub.shape
                    out_Fp = 1 + (Fp - 1) * t_sf
                    of0 = f0 * t_sf
                    of1 = min(of0 + out_Fp, out_F)
                    frames_clip = of1 - of0
                    if frames_clip <= 0:
                        continue

                    for v in range(vertical_tiles):
                        for h in range(horizontal_tiles):
                            x0 = h * (base_tile_w - overlap)
                            y0 = v * (base_tile_h - overlap)
                            x1 = min(x0 + base_tile_w, W) if h < horizontal_tiles - 1 else W
                            y1 = min(y0 + base_tile_h, H) if v < vertical_tiles - 1 else H

                            tile = sub[:, :, :, y0:y1, x0:x1]
                            dec = vae.decode(tile)
                            dec5, B2, T2, H2, W2, C2 = self._format_decoded(dec, Bb, out_Fp)
                            if B2 != Bb:
                                raise RuntimeError(f"decode batch mismatch: got {B2}, expected {Bb}")

                            Tuse = min(frames_clip, T2)
                            if Tuse <= 0:
                                continue

                            oy0 = y0 * h_sf
                            ox0 = x0 * w_sf
                            if oy0 >= out_H or ox0 >= out_W:
                                continue
                            validH = min(H2, out_H - oy0)
                            validW = min(W2, out_W - ox0)
                            if validH <= 0 or validW <= 0:
                                continue

                            dv = dec5[:, :Tuse, :validH, :validW, :]

                            tw = torch.ones((Bb, Tuse, validH, validW, 1), device=device, dtype=acc_dtype)
                            ow = min(overlap_out_w, validW)
                            oh = min(overlap_out_h, validH)
                            if ow > 0:
                                if h > 0:
                                    tw[:, :, :, :ow, 0] *= h_left[:ow].view(1, 1, 1, -1)
                                if h < horizontal_tiles - 1:
                                    tw[:, :, :, -ow:, 0] *= h_right[:ow].view(1, 1, 1, -1)
                            if oh > 0:
                                if v > 0:
                                    tw[:, :, :oh, :, 0] *= v_top[:oh].view(1, 1, -1, 1)
                                if v < vertical_tiles - 1:
                                    tw[:, :, -oh:, :, 0] *= v_bot[:oh].view(1, 1, -1, 1)

                            oy1 = oy0 + validH
                            ox1 = ox0 + validW
                            ot1 = of0 + Tuse
                            output[b0:b1, of0:ot1, oy0:oy1, ox0:ox1, :] += dv.to(acc_dtype) * tw
                            weights[b0:b1, of0:ot1, oy0:oy1, ox0:ox1, :] += tw

        output = output / (weights + 1e-8)
        if last_frame_fix:
            # Trim last t_sf frames before flattening
            output = output[:, :-t_sf, :, :, :]
            out_F = out_F - t_sf
        # Flatten batch and frame dimensions for ComfyUI IMAGE format [N, H, W, C]
        output = output.view(B * out_F, out_H, out_W, 3).to(out_dtype_t)
        return (output,)


class WASWANVAEEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "VAE used to encode the images."}),
                "images": ("IMAGE", {"tooltip": "Input images in ComfyUI format [N, H, W, C]. N can be a batch of images or frames."}),
                "batch_mode": (["images", "frames"], {"default": "images", "tooltip": "images: treat N as separate image batches [B=N, F=1]. frames: treat N as video frames [B=1, F=N]."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "latent"

    @torch.inference_mode()
    def encode(self, vae, images, batch_mode="images"):
        """
        Encode images to WAN-compatible latents with proper batch/frame dimensions.
        
        Args:
            vae: VAE model
            images: [N, H, W, C] tensor
            batch_mode: "images" = [B=N, C, F=1, H, W], "frames" = [B=1, C, F=N, H, W]
        """
        if not isinstance(images, torch.Tensor):
            raise ValueError("images must be a torch.Tensor")
        
        if images.dim() != 4:
            raise ValueError(f"images must be 4D [N, H, W, C], got {images.dim()}D")
        
        N, H, W, C = images.shape
        
        if C not in (1, 3, 4):
            raise ValueError(f"images must have 1, 3, or 4 channels, got {C}")
        
        if batch_mode == "images":
            print(f"[WASWANVAEEncode] Encoding {N} images individually as batches")
            latent_list = []
            for i in range(N):
                img_single = images[i:i+1]
                lat = vae.encode(img_single)
                if lat.dim() == 4:
                    lat = lat.unsqueeze(2)
                elif lat.dim() != 5:
                    raise RuntimeError(f"VAE encode returned unexpected {lat.dim()}D tensor")
                latent_list.append(lat)
            
            latent_samples = torch.cat(latent_list, dim=0)
            print(f"[WASWANVAEEncode] Encoded {N} images as batches: {tuple(latent_samples.shape)} [B, C, F, H, W]")
            
        elif batch_mode == "frames":
            print(f"[WASWANVAEEncode] Encoding {N} frames as video: input shape {tuple(images.shape)} [N, H, W, C]")
            latent_samples = vae.encode(images)
            if latent_samples.dim() == 4:
                latent_samples = latent_samples.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
            elif latent_samples.dim() != 5:
                raise RuntimeError(f"VAE encode returned unexpected {latent_samples.dim()}D tensor")
            
            print(f"[WASWANVAEEncode] Encoded latent shape: {tuple(latent_samples.shape)} [B, C, F, H, W]")
        else:
            raise ValueError(f"Invalid batch_mode: {batch_mode}")
        
        return ({"samples": latent_samples},)


class WASLatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT", {"tooltip": "Latent dict with key 'samples' shaped [B,C,H,W]. Also accepts [B,C,1,H,W] (e.g., Qwen) and will squeeze the singleton frame dimension automatically."}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "tooltip": "Upscale factor in LATENT space."}),
            },
            "optional": {
                "mode": (["hybrid", "bicubic", "bilinear", "nearest"], {"tooltip": "Hybrid = LP bicubic(+AA) + HP nearest with variance match."}),
                "antialias": ("BOOLEAN", {"default": True, "tooltip": "Antialias for bilinear/bicubic where supported."}),
                "hp_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Gaussian σ for LP/HP split. 0 disables split."}),
                "renoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Optional additional HP noise (0–1)."}),
                "current_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.0001, "tooltip": "If >0, scales renoise by this sigma."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "RNG seed for renoise. 0 = nondeterministic."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/resize"

    def _channelwise_std(self, x):
        return x.flatten(2).std(dim=-1, unbiased=False).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-8)

    def _gauss_kernel(self, C, sigma, device, dtype):
        if sigma <= 0:
            return None
        k = int(2 * round(3 * float(sigma)) + 1)
        if k < 3:
            k = 3
        x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = (g / g.sum()).to(dtype)
        k2 = torch.outer(g, g)
        k2 = (k2 / k2.sum()).to(dtype)
        w = k2.view(1, 1, k, k).repeat(C, 1, 1, 1)
        return w, k // 2

    def _blur(self, x, sigma):
        C = x.shape[1]
        k = self._gauss_kernel(C, sigma, x.device, x.dtype)
        if k is None:
            return x
        w, pad = k
        return F.conv2d(x, w, padding=pad, groups=C)

    def _interp(self, x, size, mode, align_corners=False, antialias=False):
        if mode in ("bilinear", "bicubic"):
            try:
                return F.interpolate(x, size=size, mode=mode, align_corners=align_corners, antialias=antialias)
            except TypeError:
                return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
        return F.interpolate(x, size=size, mode=mode)

    def _randn_like_seeded(self, x, seed: int):
        if seed == 0:
            return torch.randn_like(x)
        devices = [x.device.index] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            if x.is_cuda:
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)
            return torch.randn_like(x)

    @torch.inference_mode()
    def apply(self, latents, scale_by, mode="hybrid", antialias=True, hp_sigma=0.5,
              renoise=0.0, current_sigma=0.0, seed=0):
        x = latents["samples"]
        # Normalize shapes: accept [B,C,H,W] and [B,C,1,H,W] (Qwen with singleton frame)
        if x.dim() == 5:
            # Expected layout [B,C,F,H,W]
            B5, C5, F5, H5, W5 = x.shape
            if F5 == 1:
                x = x[:, :, 0, :, :]
            else:
                raise RuntimeError(
                    f"WASLatentUpscale expects image latents, got 5D with F>1 (video) shape {tuple(x.shape)}. "
                    "Use a video-aware latent resize node or decode per-frame."
                )
        assert x.dim() == 4, f"Expected [B,C,H,W] after normalization, got {tuple(x.shape)}"
        B, C, H, W = x.shape
        H2 = int(round(H * float(scale_by)))
        W2 = int(round(W * float(scale_by)))

        if hp_sigma > 0:
            lp = self._blur(x, hp_sigma)
            hp = x - lp
        else:
            lp = x
            hp = torch.zeros_like(x)

        if mode == "hybrid":
            lp_up = self._interp(lp, (H2, W2), mode="bicubic", align_corners=False, antialias=antialias)
            hp_up = self._interp(hp, (H2, W2), mode="nearest")
        else:
            lp_up = self._interp(lp, (H2, W2), mode=mode, align_corners=False, antialias=antialias)
            hp_up = self._interp(hp, (H2, W2), mode=mode, align_corners=False, antialias=antialias if mode in ("bilinear", "bicubic") else False)

        std_hp = self._channelwise_std(hp)
        std_hp2 = self._channelwise_std(hp_up)
        gain = (std_hp / std_hp2).clamp(0.0, 10.0)
        hp_up = hp_up * gain

        if renoise > 0.0:
            eps = self._randn_like_seeded(hp_up, seed)
            if current_sigma > 0.0:
                hp_up = hp_up + renoise * float(current_sigma) * eps
            else:
                eps_std = self._channelwise_std(eps)
                hp_up = hp_up + eps * (renoise * std_hp / eps_std)

        y = lp_up + hp_up
        return ({"samples": y},)


class WASMultiBandNoiseApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Existing latent to shape. Shape is auto-detected (4D image or 5D video)."}),
            },
            "optional": {
                "frames_per_batch": ("INT", {"default": 0, "min": 0, "max": 128, "tooltip": "For 5D video latents, process this many frames per batch (0 = all)."}),
                "enable_multiband": ("BOOLEAN", {"default": True, "tooltip": "Enable frequency-domain shaping of low/mid/high bands."}),
                "gain_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Low band gain."}),
                "gain_mid": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Mid band gain."}),
                "gain_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "High band gain."}),
                "band_xover_low": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Low/Mid crossover (normalized radius)."}),
                "band_xover_high": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mid/High crossover (normalized radius)."}),
                "region_mask": ("IMAGE", {"tooltip": "Optional mask [M,H,W,C] to apply alternate band gains inside the region."}),
                "region_gain_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Alternate low-band gain in region."}),
                "region_gain_mid": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Alternate mid-band gain in region."}),
                "region_gain_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Alternate high-band gain in region."}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Feather radius (latent pixels) for mask blending."}),
                "normalize_energy": ("BOOLEAN", {"default": False, "tooltip": "Normalize output to original std after shaping."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "noise/generation"

    def _radial_map(self, Hf, Wf, device, dtype):
        ky = torch.arange(Hf, device=device, dtype=dtype)
        ky = torch.minimum(ky, (Hf - ky)) / max(Hf / 2.0, 1.0)
        kx = torch.arange(Wf, device=device, dtype=dtype) / max(Wf - 1.0, 1.0)
        r = torch.sqrt(ky.view(Hf, 1) ** 2 + kx.view(1, Wf) ** 2)
        return r.clamp(0.0, 1.0)

    def _band_gains_map(self, Hf, Wf, device, dtype, gl, gm, gh, xo_l, xo_h):
        r = self._radial_map(Hf, Wf, device, dtype)
        l_mask = (r <= xo_l).to(dtype)
        h_mask = (r >= xo_h).to(dtype)
        m_mask = 1.0 - l_mask - h_mask
        gain = l_mask * gl + m_mask * gm + h_mask * gh
        return gain

    def _fft_shape(self, x, gl, gm, gh, xo_l, xo_h):
        N, C, H, W = x.shape
        X = torch.fft.rfftn(x, dim=(-2, -1))
        Hf, Wf = X.shape[-2], X.shape[-1]
        gain = self._band_gains_map(Hf, Wf, x.device, x.dtype, gl, gm, gh, xo_l, xo_h)
        X = X * gain.view(1, 1, Hf, Wf)
        y = torch.fft.irfftn(X, s=(H, W), dim=(-2, -1))
        return y

    @torch.inference_mode()
    def apply(self, latent, frames_per_batch=0, enable_multiband=True,
              gain_low=1.0, gain_mid=1.0, gain_high=1.0, band_xover_low=0.25, band_xover_high=0.6,
              region_mask=None, region_gain_low=1.0, region_gain_mid=1.0, region_gain_high=1.0,
              feather_px=0, normalize_energy=False):
        samples = latent["samples"]

        def shape_block(x_block, gl, gm, gh, xo_l, xo_h):
            if not enable_multiband:
                return x_block
            return self._fft_shape(x_block, gl, gm, gh, xo_l, xo_h)

        def energy_norm(ref, cur):
            if not normalize_energy:
                return cur
            ref_std = ref.flatten(2).std(dim=-1, unbiased=False).mean().clamp_min(1e-8)
            cur_std = cur.flatten(2).std(dim=-1, unbiased=False).mean().clamp_min(1e-8)
            return cur * (ref_std / cur_std)

        if samples.dim() == 4:
            x_lat = samples
            shaped_lat = shape_block(x_lat, float(gain_low), float(gain_mid), float(gain_high), float(band_xover_low), float(band_xover_high))
            if region_mask is not None:
                m = region_mask.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
                m = F.interpolate(m, size=(samples.shape[-2], samples.shape[-1]), mode="bilinear", align_corners=False)
                if feather_px and feather_px > 0:
                    k = int(feather_px) * 2 + 1
                    pad = k // 2
                    w = torch.ones((1, 1, k, k), device=m.device, dtype=m.dtype) / (k * k)
                    m = F.conv2d(m, w, padding=pad)
                if m.shape[0] == 1 and samples.shape[0] > 1:
                    m = m.repeat(samples.shape[0], 1, 1, 1)
                elif m.shape[0] != samples.shape[0]:
                    m = m[:1].repeat(samples.shape[0], 1, 1, 1)
                m = m.clamp(0.0, 1.0)
                alt_lat = shape_block(x_lat, float(region_gain_low), float(region_gain_mid), float(region_gain_high), float(band_xover_low), float(band_xover_high))
                shaped_lat = shaped_lat * (1.0 - m) + alt_lat * m
            shaped_lat = energy_norm(samples, shaped_lat)
            return ({"samples": shaped_lat},)

        # 5D
        B5, C5, F5, H5, W5 = samples.shape
        step = int(frames_per_batch) if int(frames_per_batch) > 0 else F5
        parts = []
        for f0 in range(0, F5, step):
            f1 = min(f0 + step, F5)
            xlat_blk = samples[:, :, f0:f1, :, :]
            xlat2 = xlat_blk.permute(0, 2, 1, 3, 4).contiguous().view((B5 * (f1 - f0), C5, H5, W5))
            shaped_lat_blk = shape_block(xlat2, float(gain_low), float(gain_mid), float(gain_high), float(band_xover_low), float(band_xover_high))
            if region_mask is not None:
                M = region_mask.shape[0]
                masks = []
                for fi in range(f0, f1):
                    idx = 0 if M == 1 else min(fi, M - 1)
                    m2d = region_mask[idx:idx+1].mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
                    m2d = F.interpolate(m2d, size=(H5, W5), mode="bilinear", align_corners=False)
                    if feather_px and feather_px > 0:
                        k = int(feather_px) * 2 + 1
                        pad = k // 2
                        w = torch.ones((1, 1, k, k), device=m2d.device, dtype=m2d.dtype) / (k * k)
                        m2d = F.conv2d(m2d, w, padding=pad)
                    masks.append(m2d)
                mstack = torch.cat(masks, dim=0).clamp(0.0, 1.0).repeat(B5, 1, 1, 1)
                alt_lat_blk = shape_block(xlat2, float(region_gain_low), float(region_gain_mid), float(region_gain_high), float(band_xover_low), float(band_xover_high))
                shaped_lat_blk = shaped_lat_blk * (1.0 - mstack) + alt_lat_blk * mstack
            shaped_lat_blk = shaped_lat_blk.view(B5, (f1 - f0), C5, H5, W5).permute(0, 2, 1, 3, 4).contiguous()
            parts.append(shaped_lat_blk)
        out = torch.cat(parts, dim=2)
        out = energy_norm(samples, out)
        return ({"samples": out},)


NODE_CLASS_MAPPINGS = {
    "WASWANVAEDecode": WASWANVAEDecode,
    "WASWANVAEEncode": WASWANVAEEncode,
    "WASLatentUpscale": WASLatentUpscale,
    "WASMultiBandNoiseApply": WASMultiBandNoiseApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASWANVAEDecode": "Wan VAE Decode",
    "WASWANVAEEncode": "Wan VAE Encode",
    "WASLatentUpscale": "Latent Upscale (WIP Exp.)",
    "WASMultiBandNoiseApply": "Multi-Band Latent Apply",
}
