"""TurboQuant KV Cache — Precomputed Lloyd-Max Centroids.

Lloyd-Max optimal scalar quantization for N(0, sigma) where sigma = 1/sqrt(d)
after WHT rotation. Each coordinate of the rotated vector is approximately
N(0, 1/sqrt(d)), so we precompute centroids for this distribution.

Reference: Google TurboQuant (ICLR 2026, arXiv:2504.19874)
- WHT rotation → Lloyd-Max scalar quantization → packed storage
- MSE-only variant (no QJL residual — better for softmax attention)
"""

import torch
import math
from typing import Tuple

# ---------------------------------------------------------------------------
# Lloyd-Max algorithm for 1D Gaussian
# ---------------------------------------------------------------------------

def _lloyd_max_gaussian(n_levels: int, sigma: float = 1.0,
                        max_iter: int = 200, tol: float = 1e-12
                        ) -> Tuple[list, list]:
    """Compute Lloyd-Max optimal quantizer for N(0, sigma^2).

    Returns:
        centroids: list of n_levels reconstruction values
        boundaries: list of (n_levels - 1) decision boundaries
    """
    from scipy.stats import norm
    from scipy.integrate import quad
    import numpy as np

    dist = norm(loc=0, scale=sigma)

    # Initialize centroids uniformly across [-3sigma, 3sigma]
    centroids = np.linspace(-3 * sigma, 3 * sigma, n_levels)

    for _ in range(max_iter):
        # Update boundaries (midpoints between centroids)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2
                      for i in range(n_levels - 1)]

        # Update centroids (conditional expectation in each partition)
        edges = [-np.inf] + boundaries + [np.inf]
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            # E[X | lo < X < hi] = integral(x * pdf(x), lo, hi) / P(lo < X < hi)
            num, _ = quad(lambda x: x * dist.pdf(x), lo, hi)
            den = dist.cdf(hi) - dist.cdf(lo)
            if den > 1e-15:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else centroids[i]

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Final boundaries
    boundaries = [(centroids[i] + centroids[i + 1]) / 2
                  for i in range(n_levels - 1)]

    return centroids.tolist(), boundaries


def _precompute_all() -> dict:
    """Precompute centroids for all supported (bits, head_dim) combos.

    Returns dict keyed by (bits, head_dim) → (centroids, boundaries).
    """
    results = {}
    for head_dim in [128, 256]:
        sigma = 1.0 / math.sqrt(head_dim)
        for bits, n_levels in [(3, 8), (4, 16)]:
            c, b = _lloyd_max_gaussian(n_levels, sigma)
            results[(bits, head_dim)] = (c, b)
    return results


# ---------------------------------------------------------------------------
# Hardcoded centroids (precomputed via _precompute_all)
# Run `python -m vllm.v1.attention.ops.turboquant_centroids` to regenerate.
# ---------------------------------------------------------------------------

# fmt: off

# 3-bit (8 levels), head_dim=128, sigma = 1/sqrt(128) ≈ 0.08839
# MSE=2.700e-04, SNR=14.6 dB
CENTROIDS_3BIT_D128 = [
    -0.19020693, -0.11878592, -0.06682206, -0.02166347,
     0.02166347,  0.06682206,  0.11878592,  0.19020693,
]
BOUNDARIES_3BIT_D128 = [
    -0.15449642, -0.09280399, -0.04424276,
     0.0,
     0.04424276,  0.09280399,  0.15449642,
]

# 4-bit (16 levels), head_dim=128, sigma = 1/sqrt(128) ≈ 0.08839
# MSE=7.425e-05, SNR=20.2 dB
CENTROIDS_4BIT_D128 = [
    -0.24156297, -0.18291530, -0.14305551, -0.11107301,
    -0.08332376, -0.05807442, -0.03431444, -0.01135392,
     0.01135392,  0.03431444,  0.05807442,  0.08332376,
     0.11107301,  0.14305551,  0.18291530,  0.24156297,
]
BOUNDARIES_4BIT_D128 = [
    -0.21223914, -0.16298541, -0.12706426, -0.09719839,
    -0.07069909, -0.04619443, -0.02283418,
     0.0,
     0.02283418,  0.04619443,  0.07069909,  0.09719839,
     0.12706426,  0.16298541,  0.21223914,
]

# 3-bit (8 levels), head_dim=256, sigma = 1/sqrt(256) = 0.0625
# MSE=1.350e-04, SNR=14.6 dB
CENTROIDS_3BIT_D256 = [
    -0.13449661, -0.08399433, -0.04725033, -0.01531839,
     0.01531839,  0.04725033,  0.08399433,  0.13449661,
]
BOUNDARIES_3BIT_D256 = [
    -0.10924547, -0.06562233, -0.03128436,
     0.0,
     0.03128436,  0.06562233,  0.10924547,
]

# 4-bit (16 levels), head_dim=256, sigma = 1/sqrt(256) = 0.0625
# MSE=3.713e-05, SNR=20.2 dB
CENTROIDS_4BIT_D256 = [
    -0.17081081, -0.12934065, -0.10115552, -0.07854048,
    -0.05891880, -0.04106481, -0.02426397, -0.00802843,
     0.00802843,  0.02426397,  0.04106481,  0.05891880,
     0.07854048,  0.10115552,  0.12934065,  0.17081081,
]
BOUNDARIES_4BIT_D256 = [
    -0.15007573, -0.11524809, -0.08984800, -0.06872964,
    -0.04999181, -0.03266439, -0.01614620,
     0.0,
     0.01614620,  0.03266439,  0.04999181,  0.06872964,
     0.08984800,  0.11524809,  0.15007573,
]

# fmt: on

# ---------------------------------------------------------------------------
# Lookup tables as tensors (created once, moved to device on first use)
# ---------------------------------------------------------------------------

_CENTROID_TABLES: dict = {}
_BOUNDARY_TABLES: dict = {}

_RAW = {
    (3, 128): (CENTROIDS_3BIT_D128, BOUNDARIES_3BIT_D128),
    (4, 128): (CENTROIDS_4BIT_D128, BOUNDARIES_4BIT_D128),
    (3, 256): (CENTROIDS_3BIT_D256, BOUNDARIES_3BIT_D256),
    (4, 256): (CENTROIDS_4BIT_D256, BOUNDARIES_4BIT_D256),
}


def get_centroids(bits: int, head_dim: int,
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Get centroid lookup table as a tensor.

    Returns shape (n_levels,) tensor.
    """
    key = (bits, head_dim, device, dtype)
    if key not in _CENTROID_TABLES:
        raw_c, _ = _RAW[(bits, head_dim)]
        _CENTROID_TABLES[key] = torch.tensor(raw_c, device=device, dtype=dtype)
    return _CENTROID_TABLES[key]


def get_boundaries(bits: int, head_dim: int,
                   device: torch.device = torch.device("cpu"),
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Get decision boundary tensor for quantization.

    Returns shape (n_levels - 1,) tensor.
    """
    key = (bits, head_dim, device, dtype)
    if key not in _BOUNDARY_TABLES:
        _, raw_b = _RAW[(bits, head_dim)]
        _BOUNDARY_TABLES[key] = torch.tensor(raw_b, device=device, dtype=dtype)
    return _BOUNDARY_TABLES[key]


# ---------------------------------------------------------------------------
# CLI: regenerate hardcoded values
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Regenerating Lloyd-Max centroids for TurboQuant KV cache...\n")
    results = _precompute_all()
    for (bits, head_dim), (centroids, boundaries) in sorted(results.items()):
        sigma = 1.0 / math.sqrt(head_dim)
        n = 2 ** bits
        print(f"# {bits}-bit ({n} levels), head_dim={head_dim}, "
              f"sigma = 1/sqrt({head_dim}) = {sigma:.5f}")
        print(f"CENTROIDS_{bits}BIT_D{head_dim} = [")
        for i in range(0, n, 4):
            vals = ", ".join(f"{centroids[j]:13.8f}" for j in range(i, min(i + 4, n)))
            print(f"    {vals},")
        print("]")
        print(f"BOUNDARIES_{bits}BIT_D{head_dim} = [")
        for i in range(0, n - 1, 4):
            vals = ", ".join(f"{boundaries[j]:13.8f}" for j in range(i, min(i + 4, n - 1)))
            print(f"    {vals},")
        print("]\n")

    # Verify symmetry and MSE
    import numpy as np
    from scipy.stats import norm

    print("--- Verification ---")
    for (bits, head_dim), (centroids, boundaries) in sorted(results.items()):
        sigma = 1.0 / math.sqrt(head_dim)
        c = np.array(centroids)
        # Check symmetry
        assert np.allclose(c, -c[::-1], atol=1e-10), f"Not symmetric: {bits}bit d{head_dim}"
        # Compute MSE via integration
        from scipy.integrate import quad
        dist = norm(0, sigma)
        edges = [-np.inf] + boundaries + [np.inf]
        mse = 0.0
        for i in range(len(centroids)):
            lo, hi = edges[i], edges[i + 1]
            val, _ = quad(lambda x: (x - centroids[i])**2 * dist.pdf(x), lo, hi)
            mse += val
        bpv = bits  # bits per value
        ratio = sigma**2 / mse  # SNR
        print(f"  {bits}-bit d={head_dim}: MSE={mse:.2e}, "
              f"sigma²={sigma**2:.2e}, SNR={10*math.log10(ratio):.1f} dB")
