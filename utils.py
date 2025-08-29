import math
import torch
import torch.nn.functional as F

def gaussian_kernel1d(size: int, sigma: float, device, dtype):
    """
    Generate a 1D Gaussian kernel.
    
    Args:
        size: Size of the kernel.
        sigma: Standard deviation for the kernel.
        device: Device for the kernel.
        dtype: Data type for the kernel.
    """
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k


def gaussian_blur(img: torch.Tensor, ksize: int, sigma: float):
    """
    Apply Gaussian blur to an image tensor.
    
    Args:
        img: Input image tensor of shape (N, C, H, W).
        ksize: Kernel size for blurring.
        sigma: Standard deviation for blurring.
    """
    if ksize <= 1 or sigma <= 0:
        return img
    dtype = img.dtype
    device = img.device
    k1d = gaussian_kernel1d(ksize | 1, sigma, device, dtype)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    c = img.shape[1]
    img = F.conv2d(img, kx.expand(c, 1, 1, -1), padding=(0, (ksize | 1) // 2), groups=c)
    img = F.conv2d(img, ky.expand(c, 1, -1, 1), padding=(((ksize | 1) // 2), 0), groups=c)
    return img


def perlin_noise(h, w, scale, octaves, persistence, lacunarity, seed, device, dtype):
    """
    Generate Perlin noise.
    
    Args:
        h: Height of the noise.
        w: Width of the noise.
        scale: Scale of the noise.
        octaves: Number of octaves for the noise.
        persistence: Persistence of the noise.
        lacunarity: Lacunarity of the noise.
        seed: Seed for the noise.
        device: Device for the noise.
        dtype: Data type for the noise.
    """
    if scale <= 0:
        scale = 1.0
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    def _gradients(gh, gw):
        theta = torch.rand((gh + 1, gw + 1), generator=rng, device=device, dtype=dtype) * 2 * math.pi
        return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    def _perlin2d(size_y, size_x, freq_y, freq_x):
        gh = int(math.floor(freq_y)) + 1
        gw = int(math.floor(freq_x)) + 1
        g = _gradients(gh, gw)
        y = torch.linspace(0, freq_y, steps=size_y, device=device, dtype=dtype)
        x = torch.linspace(0, freq_x, steps=size_x, device=device, dtype=dtype)
        yi = y.floor().long()
        xi = x.floor().long()
        yf = (y - yi).unsqueeze(1).expand(size_y, size_x)
        xf = (x - xi).unsqueeze(0).expand(size_y, size_x)

        def _g(ix, iy):
            return g[iy.clamp_max(gh), ix.clamp_max(gw)]

        x0 = xi.unsqueeze(0).expand(size_y, size_x)
        y0 = yi.unsqueeze(1).expand(size_y, size_x)
        g00 = _g(x0, y0)
        g10 = _g(x0 + 1, y0)
        g01 = _g(x0, y0 + 1)
        g11 = _g(x0 + 1, y0 + 1)

        d00 = torch.stack((xf, yf), dim=-1)
        d10 = torch.stack((xf - 1.0, yf), dim=-1)
        d01 = torch.stack((xf, yf - 1.0), dim=-1)
        d11 = torch.stack((xf - 1.0, yf - 1.0), dim=-1)

        n00 = (g00 * d00).sum(dim=-1)
        n10 = (g10 * d10).sum(dim=-1)
        n01 = (g01 * d01).sum(dim=-1)
        n11 = (g11 * d11).sum(dim=-1)

        def _fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
        u = _fade(xf)
        v = _fade(yf)
        nx0 = n00 * (1 - u) + n10 * u
        nx1 = n01 * (1 - u) + n11 * u
        nxy = nx0 * (1 - v) + nx1 * v
        return nxy

    yfreq = max(h / scale, 1.0)
    xfreq = max(w / scale, 1.0)
    amp = 1.0
    freq_mul = 1.0
    total = torch.zeros((h, w), device=device, dtype=dtype)
    maxamp = 0.0
    for _ in range(max(1, octaves)):
        total += amp * _perlin2d(h, w, yfreq * freq_mul, xfreq * freq_mul)
        maxamp += amp
        amp *= persistence
        freq_mul *= lacunarity
    total = total / maxamp if maxamp > 0 else total
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total


def bayer_matrix(size: int, device, dtype):
    """
    Generate a simple Bayer matrix.
    
    Args:
        size: Size of the matrix.
        device: Device for the matrix.
        dtype: Data type for the matrix.
    """
    base2 = torch.tensor([[0, 2], [3, 1]], device=device, dtype=dtype)
    if size <= 2:
        m = base2
    elif size <= 4:
        m = torch.cat([torch.cat([4*base2+0, 4*base2+2], dim=1),
                       torch.cat([4*base2+3, 4*base2+1], dim=1)], dim=0)
    else:
        m2 = torch.cat([torch.cat([4*base2+0, 4*base2+2], dim=1),
                        torch.cat([4*base2+3, 4*base2+1], dim=1)], dim=0)
        rh = size // m2.shape[0] + (1 if size % m2.shape[0] else 0)
        rw = size // m2.shape[1] + (1 if size % m2.shape[1] else 0)
        m = m2.repeat(rh, rw)[:size, :size]
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    return m
