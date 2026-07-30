#!/usr/bin/env python3
"""Offline numpy pre-check for a compensated bf16x2 split rmsnorm_matmul.

Zero remote spend. Reproduces the EXACT input the remote gate scores (the adapter
seeds np.random.seed(42) before drawing x then w; every profiler seed reuses that
same draw), computes the fp32 reference the way the NKIBench numpy reference does,
then models an idealized bf16x2 compensated split-matmul and reports the worst
NKIBench-style relative-L2 against the fp32 reference.

The split keeps each fp32 operand as two bf16 limbs and accumulates three bf16
products in fp32:
    x_hi = bf16(x),  x_lo = bf16(x - x_hi)      (round-to-nearest-even)
    w_hi = bf16(w),  w_lo = bf16(w - w_hi)
    prod ~= x_hi@w_hi + x_hi@w_lo + x_lo@w_hi   (drop x_lo@w_lo)
Each matmul's inputs are bf16 but the products are accumulated in fp32 (models the
fp32 PSUM accumulation the hardware kernel would use). This is the IDEALIZED case:
numpy round-to-nearest-even limb construction + exact fp32 accumulation is at least
as accurate as the hardware, so an offline result at/above the gate means hardware
almost-certainly fails. It is a practical no-spend gate, not an impossibility proof.

Relative-L2 is the NKIBench metric: ||v_k - v_r||_2 / ||v_r||_2 over the flattened
output (the same quantity verify.py's l2_norm gate thresholds at 2e-5).
"""

from __future__ import annotations

import numpy as np

M, N, K = 4096, 2048, 1024
INPUT_SEED = 42          # adapter/nkibench_case.py DEFAULT_INPUT_SEED
REL_TOL = 2e-5           # adapter DEFAULT_REL_TOL (the NKIBench gate)


def to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """Round fp32 -> bfloat16 (round-to-nearest-even), returned as fp32 values.

    bf16 keeps the fp32 sign+8-bit exponent and truncates the 23-bit mantissa to 7
    explicit bits, i.e. it drops the low 16 bits of the fp32 bit pattern. RNE adds a
    tie-to-even rounding bias before truncating:
        lsb          = bit 16 (the least significant bit that survives)
        rounding_bias = 0x7FFF + lsb
        bf16_bits    = (uint32(x) + rounding_bias) >> 16
    Mantissa carry into the exponent is handled correctly by the integer add. Inputs
    here are O(1) normal values (no Inf/NaN), so the NaN/Inf edge cases do not arise;
    an assert guards that assumption.
    """
    x = np.asarray(x, dtype=np.float32)
    assert np.all(np.isfinite(x)), "bf16 RNE helper assumes finite O(1) inputs"
    u = x.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = (u + bias) >> np.uint32(16)
    bf16_as_u32 = rounded.astype(np.uint32) << np.uint32(16)
    return bf16_as_u32.view(np.float32)


def split_bf16x2(x: np.ndarray):
    """Two-limb bf16 split of an fp32 array: x ~= x_hi + x_lo, both bf16-valued fp32.

    x_hi = bf16(x); the residual (x - x_hi) is exact in fp32 for O(1) magnitudes, and
    x_lo = bf16(residual) keeps its top 8 mantissa bits -> ~16 effective mantissa bits.
    """
    x = np.asarray(x, dtype=np.float32)
    x_hi = to_bf16_rne(x)
    residual = (x - x_hi).astype(np.float32)
    x_lo = to_bf16_rne(residual)
    return x_hi, x_lo


def draw_inputs(seed: int):
    """Draw (x, w) exactly as the adapter's seeded natural get_inputs() does.

    reference get_inputs(): x ~ normal(0,1,(M,K)) then w ~ normal(0,1,(K,N)), both
    fp32, in that order, after a single np.random.seed(seed). This matches
    adapter/nkibench_case.py, which seeds once and reuses the draw for every seed.
    """
    np.random.seed(seed)
    x = np.random.normal(loc=0.0, scale=1.0, size=(M, K)).astype(np.float32)
    w = np.random.normal(loc=0.0, scale=1.0, size=(K, N)).astype(np.float32)
    return x, w


def reference_forward(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The NKIBench numpy reference: RMSNorm over K (divide-by-K inside rms), then matmul.

    Mirrors reference/rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py exactly.
    """
    squared = np.square(x)
    scaled_square = np.divide(squared, K)
    rms_sum = np.sum(scaled_square, axis=1, keepdims=True)
    rms_norm = np.sqrt(rms_sum)
    normalized = np.divide(x, rms_norm)
    return np.matmul(normalized, w)


def inv_rms_per_row(x: np.ndarray) -> np.ndarray:
    """Per-row inv_rms = 1/sqrt(mean_k(x^2)); the kernel's folded rsqrt(sumsq*1/K)."""
    sumsq = np.sum(np.square(x), axis=1, keepdims=True)
    return (1.0 / np.sqrt(sumsq * np.float32(1.0 / K))).astype(np.float32)


def rel_l2(v_k: np.ndarray, v_r: np.ndarray) -> float:
    """NKIBench relative-L2 over the flattened output: ||v_k - v_r||_2 / ||v_r||_2."""
    num = np.linalg.norm((v_k - v_r).ravel().astype(np.float64))
    den = np.linalg.norm(v_r.ravel().astype(np.float64))
    return float(num / den)


def fp32_control_postscale(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """fp32 control that mirrors the kernel's post-scale commutation (validates model).

    Matmul RAW x in fp32, then scale each row by inv_rms — algebraically equal to the
    reference (normalize then matmul), differing only by fp32 reassociation. Reproducing
    the reference to ~1e-7 confirms seed/draw-order/dtype/formula all match before the
    bf16x2 number is trusted.
    """
    raw = np.matmul(x.astype(np.float32), w.astype(np.float32))
    return (raw * inv_rms_per_row(x)).astype(np.float32)


def bf16x2_postscale(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Idealized bf16x2 3-product split on RAW x (post-scale by fp32 inv_rms).

    Faithful to a device kernel built on the post-scale eviction-fold base: split raw x
    and w into bf16 limbs, accumulate the 3 products in fp32 (models fp32 PSUM), then
    apply the exact fp32 per-row inv_rms at eviction.
    """
    x_hi, x_lo = split_bf16x2(x)
    w_hi, w_lo = split_bf16x2(w)
    # Each product: bf16-valued inputs, fp32 accumulation (np.matmul in fp32).
    prod = (np.matmul(x_hi, w_hi) + np.matmul(x_hi, w_lo)
            + np.matmul(x_lo, w_hi)).astype(np.float32)
    return (prod * inv_rms_per_row(x)).astype(np.float32)


def bf16x2_prescale(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Idealized bf16x2 3-product split on NORMALIZED x (pre-scale variant).

    Alternative kernel design: normalize x in fp32 first, then split the normalized
    activation and w into bf16 limbs. Reported alongside the post-scale variant so the
    worst-of-both is the number the no-spend gate sees.
    """
    normalized = (x * inv_rms_per_row(x)).astype(np.float32)
    x_hi, x_lo = split_bf16x2(normalized)
    w_hi, w_lo = split_bf16x2(w)
    return (np.matmul(x_hi, w_hi) + np.matmul(x_hi, w_lo)
            + np.matmul(x_lo, w_hi)).astype(np.float32)


def bf16x2_full4_postscale(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reference-only: full 4-product split (keeps x_lo@w_lo) to size the dropped term."""
    x_hi, x_lo = split_bf16x2(x)
    w_hi, w_lo = split_bf16x2(w)
    prod = (np.matmul(x_hi, w_hi) + np.matmul(x_hi, w_lo)
            + np.matmul(x_lo, w_hi) + np.matmul(x_lo, w_lo)).astype(np.float32)
    return (prod * inv_rms_per_row(x)).astype(np.float32)


def plain_bf16(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reference-only: plain single-limb bf16 matmul (the rejected route) for scale."""
    x_hi = to_bf16_rne(x)
    w_hi = to_bf16_rne(w)
    return (np.matmul(x_hi, w_hi).astype(np.float32) * inv_rms_per_row(x)).astype(np.float32)


def _unit_test_bf16():
    """Sanity-check the RNE helper on values with known bf16 roundings."""
    # 1.0 is exact in bf16.
    assert to_bf16_rne(np.float32(1.0)) == np.float32(1.0)
    # 1 + 2^-8 rounds to nearest-even. bf16 step at 1.0 is 2^-7; 2^-8 is a half-ulp tie
    # -> ties to even (1.0, whose mantissa lsb is 0).
    v = np.float32(1.0 + 2.0 ** -8)
    assert to_bf16_rne(v) == np.float32(1.0), to_bf16_rne(v)
    # 1 + 3*2^-9 is just above the tie -> rounds up to 1 + 2^-7.
    v = np.float32(1.0 + 3.0 * 2.0 ** -9)
    assert to_bf16_rne(v) == np.float32(1.0 + 2.0 ** -7), to_bf16_rne(v)
    # Split reconstructs to ~16-bit accuracy.
    rng = np.random.default_rng(0)
    a = rng.normal(size=10000).astype(np.float32)
    a_hi, a_lo = split_bf16x2(a)
    rel = np.linalg.norm((a - (a_hi + a_lo)).astype(np.float64)) / np.linalg.norm(a.astype(np.float64))
    assert rel < 3e-5, rel
    return rel


def main():
    self_test_rel = _unit_test_bf16()
    print(f"[self-test] bf16 RNE helper OK; bf16x2 reconstruction rel-L2 on N(0,1) = {self_test_rel:.3e}")
    print(f"[config] M={M} N={N} K={K}  input_seed={INPUT_SEED}  gate rel_tol={REL_TOL:.1e}\n")

    # --- Primary: the exact scored input (seed 42) ---
    x, w = draw_inputs(INPUT_SEED)
    ref = reference_forward(x, w)

    ctrl = fp32_control_postscale(x, w)
    ctrl_rel = rel_l2(ctrl, ref)

    post = bf16x2_postscale(x, w)
    pre = bf16x2_prescale(x, w)
    full4 = bf16x2_full4_postscale(x, w)
    plain = plain_bf16(x, w)

    post_rel = rel_l2(post, ref)
    pre_rel = rel_l2(pre, ref)
    full4_rel = rel_l2(full4, ref)
    plain_rel = rel_l2(plain, ref)

    print(f"[seed {INPUT_SEED}] fp32 post-scale CONTROL vs reference rel-L2 = {ctrl_rel:.3e}"
          f"   (validates seed/draw-order/dtype/formula; expect ~1e-7)")
    print(f"[seed {INPUT_SEED}] plain bf16 (rejected)        rel-L2 = {plain_rel:.3e}"
          f"   (~4e-3 scale check)")
    print(f"[seed {INPUT_SEED}] bf16x2 3-product post-scale  rel-L2 = {post_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 3-product pre-scale   rel-L2 = {pre_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 4-product (keeps lo*lo) rel-L2 = {full4_rel:.3e}"
          f"   (sizes the dropped x_lo*w_lo term)\n")

    worst_primary = max(post_rel, pre_rel)

    # --- Robustness: extra synthetic seeds (NOT the scored input, but same distribution) ---
    extra_seeds = [0, 21, 63, 84, 123, 2024]
    extra = []
    for s in extra_seeds:
        xs, ws = draw_inputs(s)
        rs = reference_forward(xs, ws)
        p_post = rel_l2(bf16x2_postscale(xs, ws), rs)
        p_pre = rel_l2(bf16x2_prescale(xs, ws), rs)
        extra.append((s, p_post, p_pre))
        print(f"[seed {s:4d}] bf16x2 post-scale = {p_post:.3e}   pre-scale = {p_pre:.3e}")

    worst_all = max([worst_primary] + [max(p, q) for _, p, q in extra])

    print("\n" + "=" * 64)
    print(f"fp32 control rel-L2 (must be << gate, validates model): {ctrl_rel:.3e}")
    print(f"WORST bf16x2 3-product rel-L2 (scored seed 42):         {worst_primary:.3e}")
    print(f"WORST bf16x2 3-product rel-L2 (all seeds):              {worst_all:.3e}")
    print(f"NKIBench gate:                                          {REL_TOL:.1e}")
    verdict = "COMFORTABLY BELOW (<=1.5e-5)" if worst_all <= 1.5e-5 else (
        "MARGINAL/ABOVE (>1.5e-5)")
    print(f"no-spend decision input: worst={worst_all:.3e}  ->  {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
