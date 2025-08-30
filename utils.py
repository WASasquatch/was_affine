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


def _radial_frequency_grid(h: int, w: int, device, dtype):
    """
    Build a radial spatial-frequency grid in cycles/pixel for 2D FFT shaping.

    Returns tensor of shape (h, w) with radius r = sqrt(fx^2 + fy^2).
    """
    fy = torch.fft.fftfreq(h, d=1.0, device=device)
    fx = torch.fft.fftfreq(w, d=1.0, device=device)
    # Use float dtype for operations, then cast to requested dtype at the end
    Fy = fy.reshape(h, 1).to(torch.float64)
    Fx = fx.reshape(1, w).to(torch.float64)
    r = torch.sqrt(Fx * Fx + Fy * Fy)
    return r.to(dtype=dtype, device=device)


def spectral_noise(h: int, w: int, beta: float, seed: int, device, dtype):
    """
    Generate 2D spatial noise with power spectral density S(f) ∝ f^beta.

    beta = 0   -> white
    beta = -1  -> pink (1/f)
    beta = -2  -> brown/red (1/f^2)
    beta = +1  -> blue (f)
    beta = +2  -> violet/purple (f^2)
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    base = torch.randn((h, w), generator=rng, device=device, dtype=dtype)
    F = torch.fft.fft2(base)
    r = _radial_frequency_grid(h, w, device, dtype)
    r = torch.clamp(r, min=1e-6)
    amp = torch.pow(r, beta * 0.5)
    F_shaped = F * amp
    x = torch.fft.ifft2(F_shaped).real
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x


def _rand_generator(seed: int, device):
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return g


def _chunked_pairwise_min_dists(px: torch.Tensor, pts: torch.Tensor, k: int = 1, chunk_pts: int = 256, chunk_px: int = 65536):
    """
    Compute k smallest distances from each pixel to points with chunking.
    px: (P,2), pts: (N,2)
    Returns f1 (P,), optional f2 (P,) if k>=2
    """
    P = px.shape[0]
    device = px.device
    dtype = px.dtype
    f = [torch.full((P,), float("inf"), device=device, dtype=dtype) for _ in range(k)]
    for pi in range(0, P, chunk_px):
        pchunk = px[pi:pi+chunk_px]  # (Pc,2)
        # current bests for this slice
        best = [torch.full((pchunk.shape[0],), float("inf"), device=device, dtype=dtype) for _ in range(k)]
        for i in range(0, pts.shape[0], chunk_pts):
            chunk = pts[i:i+chunk_pts]  # (C,2)
            # distances squared
            d2 = (pchunk[:, None, 0] - chunk[None, :, 0])**2 + (pchunk[:, None, 1] - chunk[None, :, 1])**2
            d = torch.sqrt(d2 + 1e-12)
            # merge with current bests
            merged = torch.cat([d] + [b.view(-1, 1) for b in best], dim=1)
            vals, _ = torch.topk(merged, k=k, dim=1, largest=False)
            for j in range(k):
                best[j] = vals[:, j]
        # write back
        for j in range(k):
            f[j][pi:pi+best[j].shape[0]] = best[j]
    return tuple(f)


def poisson_blue_mask_2d(h: int, w: int, radius_px: float, softness: float, seed: int, device, dtype):
    """
    Generate a Poisson-disk point pattern, then map distance-to-nearest into [0,1].
    softness controls how distance maps to mask: m = 1 - exp(-D / softness).
    """
    g = _rand_generator(seed, device)
    # Bridson dart throwing with grid accel
    r = max(1.0, float(radius_px))
    cell = r / math.sqrt(2.0)
    gw, gh = int(math.ceil(w / cell)), int(math.ceil(h / cell))
    grid = -torch.ones((gh, gw), device=device, dtype=torch.int32)
    points = []

    def grid_coords(x, y):
        return int(y // cell), int(x // cell)

    def fits(x, y):
        gi, gj = grid_coords(x, y)
        i0, i1 = max(0, gi - 2), min(gh - 1, gi + 2)
        j0, j1 = max(0, gj - 2), min(gw - 1, gj + 2)
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                idx = int(grid[ii, jj].item())
                if idx >= 0:
                    px, py = points[idx]
                    if (px - x) ** 2 + (py - y) ** 2 < r * r:
                        return False
        return True

    # initial point
    x0 = torch.rand((), generator=g, device=device).item() * w
    y0 = torch.rand((), generator=g, device=device).item() * h
    points.append((x0, y0))
    gi, gj = grid_coords(x0, y0)
    grid[gi, gj] = 0
    active = [0]

    k = 30
    max_points = int(4.0 * (h * w) / (r * r + 1e-6))  # heuristic cap
    while active and len(points) < max_points:
        ai = int(active[torch.randint(0, len(active), (1,), generator=g, device=device).item()])
        cx, cy = points[ai]
        placed = False
        for _ in range(k):
            rad = r * (1.0 + torch.rand((), generator=g, device=device).item())
            ang = 2 * math.pi * torch.rand((), generator=g, device=device).item()
            nx = cx + rad * math.cos(ang)
            ny = cy + rad * math.sin(ang)
            if 0 <= nx < w and 0 <= ny < h and fits(nx, ny):
                points.append((nx, ny))
                gi, gj = grid_coords(nx, ny)
                grid[gi, gj] = len(points) - 1
                active.append(len(points) - 1)
                placed = True
                break
        if not placed:
            active.remove(ai)

    if len(points) == 0:
        return torch.zeros((h, w), device=device, dtype=dtype)

    pts = torch.tensor(points, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    px = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    (f1,) = _chunked_pairwise_min_dists(px, pts, k=1)
    D = f1.reshape(h, w)
    m = 1.0 - torch.exp(-D / max(1e-6, float(softness)))
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    return m


def worley_edges_2d(h: int, w: int, points_per_kpx: float, metric: str, edge_sharpness: float, seed: int, device, dtype):
    """
    Worley cellular edges via F1/F2 distances. Edge strength ~ 1 - (F2-F1)/norm.
    metric: 'L2' or 'L1'.
    """
    g = _rand_generator(seed, device)
    npts = max(1, int(points_per_kpx * (h * w) / 1000.0))
    pts = torch.stack([
        torch.rand((npts,), generator=g, device=device, dtype=dtype) * w,
        torch.rand((npts,), generator=g, device=device, dtype=dtype) * h,
    ], dim=1)
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    px = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    # Distance function
    def dist_fn(a, b):
        if metric.upper() == "L1":
            d = torch.abs(a[:, None, :] - b[None, :, :]).sum(dim=2)
        else:
            d = torch.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(dim=2) + 1e-12)
        return d

    # chunked keep two smallest distances per pixel
    P = px.shape[0]
    f1 = torch.full((P,), float("inf"), device=device, dtype=dtype)
    f2 = torch.full((P,), float("inf"), device=device, dtype=dtype)
    chunk_pts = 256
    chunk_px = 65536
    for pi in range(0, P, chunk_px):
        pchunk = px[pi:pi+chunk_px]
        bf1 = torch.full((pchunk.shape[0],), float("inf"), device=device, dtype=dtype)
        bf2 = torch.full((pchunk.shape[0],), float("inf"), device=device, dtype=dtype)
        for i in range(0, pts.shape[0], chunk_pts):
            chunk = pts[i:i+chunk_pts]
            d = dist_fn(pchunk, chunk)
            merged = torch.cat([d, bf1.view(-1, 1), bf2.view(-1, 1)], dim=1)
            vals, _ = torch.topk(merged, k=2, dim=1, largest=False)
            bf1, bf2 = vals[:, 0], vals[:, 1]
        f1[pi:pi+bf1.shape[0]] = bf1
        f2[pi:pi+bf2.shape[0]] = bf2

    F1 = f1.reshape(h, w)
    F2 = f2.reshape(h, w)
    # Edge measure: small gap between F2 and F1 -> on boundaries
    gap = (F2 - F1)
    gap = gap / (gap.max() + 1e-12)
    edges = 1.0 - gap
    # sharpen
    s = max(0.1, float(edge_sharpness))
    edges = torch.clamp(edges, 0.0, 1.0) ** s
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-12)
    return edges


def tile_oriented_lines_2d(h: int, w: int, tile_size: int, freq_cyc_px: float, jitter: float, seed: int, device, dtype):
    """
    Randomized oriented sinusoid per tile, stitched together; normalized to [0,1].
    """
    s = max(2, int(tile_size))
    g = _rand_generator(seed, device)
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    out = torch.zeros((h, w), device=device, dtype=dtype)
    two_pi = 2.0 * math.pi
    f = float(freq_cyc_px)
    for y0 in range(0, h, s):
        for x0 in range(0, w, s):
            y1 = min(h, y0 + s)
            x1 = min(w, x0 + s)
            th = torch.rand((), generator=g, device=device, dtype=dtype).item() * two_pi
            ph = (torch.rand((), generator=g, device=device, dtype=dtype).item() - 0.5) * two_pi * float(jitter)
            sl = slice(y0, y1)
            sc = slice(x0, x1)
            proj = (xx[sl, sc] - x0) * math.cos(th) + (yy[sl, sc] - y0) * math.sin(th)
            gr = torch.sin(two_pi * f * proj + ph)
            out[sl, sc] = gr
    out = (out - out.min()) / (out.max() - out.min() + 1e-12)
    return out


def dot_screen_jitter_2d(h: int, w: int, cell_size: int, jitter_px: float, fill_ratio: float, seed: int, device, dtype):
    """
    Halftone dot lattice with per-cell jitter of center and radius.
    fill_ratio ~ area fraction per cell (0..1).
    """
    cs = max(2, int(cell_size))
    g = _rand_generator(seed, device)
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    mask = torch.zeros((h, w), device=device, dtype=dtype)
    base_r = 0.5 * math.sqrt(fill_ratio) * cs
    for y0 in range(0, h, cs):
        for x0 in range(0, w, cs):
            y1 = min(h, y0 + cs)
            x1 = min(w, x0 + cs)
            jx = (torch.rand((), generator=g, device=device, dtype=dtype).item() - 0.5) * 2.0 * float(jitter_px)
            jy = (torch.rand((), generator=g, device=device, dtype=dtype).item() - 0.5) * 2.0 * float(jitter_px)
            cx = x0 + cs * 0.5 + jx
            cy = y0 + cs * 0.5 + jy
            rj = base_r * (0.8 + 0.4 * torch.rand((), generator=g, device=device, dtype=dtype).item())
            sl = slice(y0, y1)
            sc = slice(x0, x1)
            d2 = (xx[sl, sc] - cx) ** 2 + (yy[sl, sc] - cy) ** 2
            mask[sl, sc] = torch.where(d2 <= rj * rj, torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype))
    # Slight blur to smooth hard edges
    mask = gaussian_blur(mask.unsqueeze(0).unsqueeze(0), ksize=3, sigma=0.6).squeeze(0).squeeze(0)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-12)
    return mask


def _grid(h: int, w: int, device, dtype):
    yy = torch.arange(h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, w).expand(h, w)
    return xx, yy


def cross_hatch_2d(
    h: int,
    w: int,
    freq_cyc_px: float,
    angles_deg: tuple,
    square: bool,
    phase_jitter: float,
    supersample: int,
    seed: int,
    device,
    dtype,
):
    """
    Generate cross-hatch gratings at given frequency (cycles/pixel) and angles.
    Optionally square-wave, phase jitter, and supersample then downsample.
    Returns normalized [0,1].
    """
    s = max(1, int(supersample))
    H, W = h * s, w * s
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    xx, yy = _grid(H, W, device, dtype)

    two_pi = 2.0 * math.pi
    f = float(freq_cyc_px)
    # Allow aliasing if f > 0.5 (user-controlled)
    acc = torch.zeros((H, W), device=device, dtype=dtype)
    ad = angles_deg if isinstance(angles_deg, (list, tuple)) else [0.0, 90.0]
    for a in ad:
        th = math.radians(float(a))
        # Phase jitter in [0, 2π*phase_jitter)
        phi = 0.0
        if phase_jitter > 0.0:
            phi = torch.rand((), generator=rng, device=device, dtype=dtype).item() * two_pi * float(phase_jitter)
        proj = xx * math.cos(th) + yy * math.sin(th)
        gr = torch.sin(two_pi * f * proj + phi)
        if square:
            gr = torch.sign(gr)
        acc = acc + gr
    # Normalize to [0,1]
    if len(ad) > 0:
        acc = acc / float(len(ad))
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-12)

    if s > 1:
        acc = acc.unsqueeze(0).unsqueeze(0)
        # Light blur before downsample to reduce aliasing
        acc = gaussian_blur(acc, ksize=3, sigma=0.6)
        acc = F.interpolate(acc, size=(h, w), mode="area")
        acc = acc.squeeze(0).squeeze(0)
        acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-12)
    return acc


def highpass_white_2d(h: int, w: int, cutoff_frac: float, order: int, seed: int, device, dtype):
    """
    White noise shaped by Butterworth high-pass filter.
    cutoff_frac: 0..1 fraction of Nyquist radius.
    order: filter order (>=1).
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    base = torch.randn((h, w), generator=rng, device=device, dtype=dtype)
    F = torch.fft.fft2(base)
    r = _radial_frequency_grid(h, w, device, dtype)
    r = torch.clamp(r, min=1e-6)
    rmax = r.max().clamp(min=1e-6)
    rc = torch.as_tensor(cutoff_frac, device=device, dtype=dtype).clamp(1e-6, 1.0) * rmax
    n = max(1, int(order))
    # Butterworth high-pass magnitude response
    H = 1.0 / (1.0 + torch.pow(rc / r, 2 * n))
    H = H.to(F.dtype)
    F_hp = F * H
    x = torch.fft.ifft2(F_hp).real
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x


def ring_noise_2d(h: int, w: int, center_frac: float, bandwidth_frac: float, seed: int, device, dtype):
    """
    Narrow annulus band-pass white noise around center_frac of Nyquist radius.
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    base = torch.randn((h, w), generator=rng, device=device, dtype=dtype)
    F = torch.fft.fft2(base)
    r = _radial_frequency_grid(h, w, device, dtype)
    rmax = r.max().clamp(min=1e-6)
    c = torch.as_tensor(center_frac, device=device, dtype=dtype).clamp(0.0, 1.0) * rmax
    bw = torch.as_tensor(bandwidth_frac, device=device, dtype=dtype).clamp(1e-6, 1.0) * rmax
    # Gaussian ring
    sigma = bw
    mask = torch.exp(-0.5 * ((r - c) / sigma) ** 2)
    F_ring = F * mask
    x = torch.fft.ifft2(F_ring).real
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x


def pink_noise_2d(h: int, w: int, seed: int, device, dtype):
    return spectral_noise(h, w, beta=-1.0, seed=seed, device=device, dtype=dtype)


def brown_noise_2d(h: int, w: int, seed: int, device, dtype):
    # Also known as red noise
    return spectral_noise(h, w, beta=-2.0, seed=seed, device=device, dtype=dtype)


def blue_noise_2d(h: int, w: int, seed: int, device, dtype):
    return spectral_noise(h, w, beta=+1.0, seed=seed, device=device, dtype=dtype)


def violet_noise_2d(h: int, w: int, seed: int, device, dtype):
    # Also called purple noise
    return spectral_noise(h, w, beta=+2.0, seed=seed, device=device, dtype=dtype)


def velvet_noise(h: int, w: int, tap_count: int, seed: int, device, dtype):
    """
    Generate 2D velvet noise: sparse random ±1 impulses, returned in [0,1].

    tap_count: number of impulses to place. Values are clipped to [1, h*w].
    """
    tap_count = int(max(1, min(tap_count, h * w)))
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    v = torch.zeros((h, w), device=device, dtype=dtype)
    perm = torch.randperm(h * w, generator=rng, device=device)[:tap_count]
    signs = (torch.randint(0, 2, (tap_count,), generator=rng, device=device) * 2 - 1).to(dtype)
    flat = v.view(-1)
    flat[perm] = signs

    # Map from {-1, 0, +1} to [0,1] with 0.5 baseline
    v = (v + 1.0) * 0.5
    return v


def green_noise_2d(h: int, w: int, seed: int, device, dtype, center_frac: float = 0.35, bandwidth_frac: float = 0.15):
    """
    Generate green noise as a mid-frequency band-pass shaped white noise.

    center_frac, bandwidth_frac are relative to max radial frequency (rmax). Typical values:
      center_frac ~ 0.3-0.4, bandwidth_frac ~ 0.1-0.2
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    base = torch.randn((h, w), generator=rng, device=device, dtype=dtype)
    F = torch.fft.fft2(base)
    r = _radial_frequency_grid(h, w, device, dtype)
    rmax = r.max().clamp(min=1e-6)
    c = torch.as_tensor(center_frac, device=device, dtype=dtype).clamp(0.0, 1.0) * rmax
    bw = torch.as_tensor(bandwidth_frac, device=device, dtype=dtype).clamp(1e-6, 1.0) * rmax
    sigma = bw
    wmask = torch.exp(-0.5 * ((r - c) / sigma) ** 2)
    F_bp = F * wmask
    x = torch.fft.ifft2(F_bp).real
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x


def black_noise_2d(h: int, w: int, seed: int, device, dtype, bins: int = 2048):
    """
    Generate black noise as sparse narrowband content: keep a small number of
    frequency bins (in rFFT domain), zero elsewhere, then inverse transform.
    """
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    # rFFT spectrum shape for real inputs
    spec_shape = (h, w // 2 + 1)
    total_bins = spec_shape[0] * spec_shape[1]
    k = int(max(1, min(bins, total_bins - 1)))  # exclude DC ideally

    spec = torch.zeros(spec_shape, device=device, dtype=torch.complex64)
    idx = torch.randperm(total_bins, generator=rng, device=device)
    idx = idx[idx != 0]
    idx = idx[:k]

    real = torch.randn((k,), generator=rng, device=device)
    imag = torch.randn((k,), generator=rng, device=device)
    vals = torch.complex(real, imag)

    flat = spec.view(-1)
    flat[idx] = vals
    spec = flat.view(spec_shape)

    x = torch.fft.irfft2(spec, s=(h, w))
    x = x.to(dtype)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x
