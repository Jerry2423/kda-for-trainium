#!/usr/bin/env python3
"""Offline numpy pre-check for a compensated bf16x2 split add_rmsnorm_matmul.

Zero remote spend. Reproduces the EXACT input the remote gate scores (the adapter
seeds np.random.seed(42) before drawing x, w, eps, z, g in that order; every
profiler seed reuses that same draw), computes the fp32 reference the way the
NKIBench numpy reference does, then models an idealized bf16x2 compensated
split-matmul and reports the worst NKIBench-style relative-L2 against the fp32
reference.

This op is the near-exact sibling of rmsnorm_matmul, with three deltas:
  1. residual add  a = x + z        before the norm
  2. per-K learned scale  y *= g[k] (g does NOT commute past the matmul)
  3. +eps inside the rsqrt

The matmul the PE array runs is  a @ w', where  w'[k,n] = g[k] * w[k,n]  (g folded
into the resident weight, a per-partition scale on the [k_in, n] weight tile), and
the per-row inv_rms[m] = 1/sqrt(mean_k(a^2) + eps) is applied post-scale at eviction
(it commutes with the matmul). The split keeps each fp32 operand as two bf16 limbs
and accumulates three bf16 products in fp32, dropping the negligible lo*lo term:
    a_hi = bf16(a),   a_lo = bf16(a - a_hi)          (round-to-nearest-even)
    w'_hi = bf16(w'), w'_lo = bf16(w' - w'_hi)
    prod ~= a_hi@w'_hi + a_hi@w'_lo + a_lo@w'_hi

Two g placements are compared so the worst is the number the no-spend gate sees:
  - g-into-w  : split w' = g*w   (matches the cheap resident-weight fold)
  - g-on-act  : split y = a*g    (v1's activation-side placement)

Idealized numpy RNE limb construction + exact fp32 accumulation is at least as
accurate as the hardware, so an offline result at/above the gate means the device
almost-certainly fails. It is a practical no-spend gate, not an impossibility proof.

CONTEXT (why this matters MORE here than for the sibling): the on-device fp32 v1
already measures rel-L2 = 1.46e-5, only ~1.37x under the 2e-5 gate -- the trn2 PE
array is bf16-native and its "fp32" matmul is itself a multi-pass bf16 emulation.
So the bf16x2 margin is tighter here than the sibling's ~4.5e-6; gate carefully.
"""

from __future__ import annotations

import numpy as np

M, N, K = 4096, 2048, 1024
INPUT_SEED = 42          # adapter/nkibench_case.py DEFAULT_INPUT_SEED
REL_TOL = 2e-5           # adapter DEFAULT_REL_TOL (the NKIBench gate)
EPS = np.float32(1e-5)


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
    """Two-limb bf16 split of an fp32 array: x ~= x_hi + x_lo, both bf16-valued fp32."""
    x = np.asarray(x, dtype=np.float32)
    x_hi = to_bf16_rne(x)
    residual = (x - x_hi).astype(np.float32)
    x_lo = to_bf16_rne(residual)
    return x_hi, x_lo


def draw_inputs(seed: int):
    """Draw (x, w, z, g) exactly as the adapter's seeded natural get_inputs() does.

    reference get_inputs(): x ~ normal(0,1,(M,K)), w ~ normal(0,1,(K,N)), eps=1e-5
    (no draw), z ~ normal(0,1,(M,K)), g ~ normal(0,1,(K,)), all fp32, in that order,
    after a single np.random.seed(seed). eps is a python float so it does NOT advance
    the RNG; the draw order is x -> w -> z -> g.
    """
    np.random.seed(seed)
    x = np.random.normal(loc=0.0, scale=1.0, size=(M, K)).astype(np.float32)
    w = np.random.normal(loc=0.0, scale=1.0, size=(K, N)).astype(np.float32)
    z = np.random.normal(loc=0.0, scale=1.0, size=(M, K)).astype(np.float32)
    g = np.random.normal(loc=0.0, scale=1.0, size=(K,)).astype(np.float32)
    return x, w, z, g


def reference_forward(x, w, z, g) -> np.ndarray:
    """The NKIBench numpy reference for add_rmsnorm_matmul (exact mirror)."""
    y = (x + z).astype(np.float32)
    t = np.square(y)
    t = np.divide(t, K)
    t = np.sum(t, axis=-1, keepdims=True)
    t = (t + EPS).astype(np.float32)
    y = y / np.sqrt(t)
    y = y * g
    return np.matmul(y, w).astype(np.float32)


def inv_rms_per_row(a: np.ndarray) -> np.ndarray:
    """Per-row inv_rms = 1/sqrt(mean_k(a^2) + eps); the kernel's post-scale factor."""
    sumsq = np.sum(np.square(a), axis=1, keepdims=True)
    mean_eps = (sumsq * np.float32(1.0 / K) + EPS).astype(np.float32)
    return (1.0 / np.sqrt(mean_eps)).astype(np.float32)


def rel_l2(v_k: np.ndarray, v_r: np.ndarray) -> float:
    """NKIBench relative-L2 over the flattened output: ||v_k - v_r||_2 / ||v_r||_2."""
    num = np.linalg.norm((v_k - v_r).ravel().astype(np.float64))
    den = np.linalg.norm(v_r.ravel().astype(np.float64))
    return float(num / den)


def fp32_control(x, w, z, g) -> np.ndarray:
    """fp32 control mirroring the kernel's post-scale + g-into-w commutation.

    Matmul RAW a=x+z against w'=g*w in fp32, then scale each row by inv_rms.
    Algebraically equal to the reference; reproducing it to ~1e-7 confirms
    seed/draw-order/dtype/formula/eps all match before the bf16x2 number is trusted.
    """
    a = (x + z).astype(np.float32)
    wp = (g[:, None] * w).astype(np.float32)
    raw = np.matmul(a, wp).astype(np.float32)
    return (raw * inv_rms_per_row(a)).astype(np.float32)


def bf16x2_g_into_w(x, w, z, g) -> np.ndarray:
    """Idealized bf16x2 3-product split with g FOLDED INTO w (the cheap resident fold)."""
    a = (x + z).astype(np.float32)
    wp = (g[:, None] * w).astype(np.float32)
    a_hi, a_lo = split_bf16x2(a)
    w_hi, w_lo = split_bf16x2(wp)
    prod = (np.matmul(a_hi, w_hi) + np.matmul(a_hi, w_lo)
            + np.matmul(a_lo, w_hi)).astype(np.float32)
    return (prod * inv_rms_per_row(a)).astype(np.float32)


def bf16x2_g_on_act(x, w, z, g) -> np.ndarray:
    """Idealized bf16x2 3-product split with g applied to the ACTIVATION (v1's placement)."""
    a = (x + z).astype(np.float32)
    y = (a * g).astype(np.float32)          # g broadcast over the free axis
    y_hi, y_lo = split_bf16x2(y)
    w_hi, w_lo = split_bf16x2(w)
    prod = (np.matmul(y_hi, w_hi) + np.matmul(y_hi, w_lo)
            + np.matmul(y_lo, w_hi)).astype(np.float32)
    return (prod * inv_rms_per_row(a)).astype(np.float32)


def bf16x2_full4_g_into_w(x, w, z, g) -> np.ndarray:
    """Reference-only: full 4-product split (keeps lo*lo) to size the dropped term."""
    a = (x + z).astype(np.float32)
    wp = (g[:, None] * w).astype(np.float32)
    a_hi, a_lo = split_bf16x2(a)
    w_hi, w_lo = split_bf16x2(wp)
    prod = (np.matmul(a_hi, w_hi) + np.matmul(a_hi, w_lo)
            + np.matmul(a_lo, w_hi) + np.matmul(a_lo, w_lo)).astype(np.float32)
    return (prod * inv_rms_per_row(a)).astype(np.float32)


def plain_bf16_g_into_w(x, w, z, g) -> np.ndarray:
    """Reference-only: plain single-limb bf16 matmul (the rejected route) for scale."""
    a = (x + z).astype(np.float32)
    wp = (g[:, None] * w).astype(np.float32)
    a_hi = to_bf16_rne(a)
    w_hi = to_bf16_rne(wp)
    return (np.matmul(a_hi, w_hi).astype(np.float32) * inv_rms_per_row(a)).astype(np.float32)


def _unit_test_bf16():
    """Sanity-check the RNE helper on values with known bf16 roundings."""
    assert to_bf16_rne(np.float32(1.0)) == np.float32(1.0)
    v = np.float32(1.0 + 2.0 ** -8)                      # half-ulp tie -> to even (1.0)
    assert to_bf16_rne(v) == np.float32(1.0), to_bf16_rne(v)
    v = np.float32(1.0 + 3.0 * 2.0 ** -9)                # just above tie -> rounds up
    assert to_bf16_rne(v) == np.float32(1.0 + 2.0 ** -7), to_bf16_rne(v)
    rng = np.random.default_rng(0)
    a = rng.normal(size=10000).astype(np.float32)
    a_hi, a_lo = split_bf16x2(a)
    rel = np.linalg.norm((a - (a_hi + a_lo)).astype(np.float64)) / np.linalg.norm(a.astype(np.float64))
    assert rel < 3e-5, rel
    return rel


def main():
    self_test_rel = _unit_test_bf16()
    print(f"[self-test] bf16 RNE helper OK; bf16x2 reconstruction rel-L2 on N(0,1) = {self_test_rel:.3e}")
    print(f"[config] M={M} N={N} K={K}  input_seed={INPUT_SEED}  eps={float(EPS):.0e}  "
          f"gate rel_tol={REL_TOL:.1e}\n")

    x, w, z, g = draw_inputs(INPUT_SEED)
    ref = reference_forward(x, w, z, g)

    ctrl_rel = rel_l2(fp32_control(x, w, z, g), ref)
    plain_rel = rel_l2(plain_bf16_g_into_w(x, w, z, g), ref)
    giw_rel = rel_l2(bf16x2_g_into_w(x, w, z, g), ref)
    goa_rel = rel_l2(bf16x2_g_on_act(x, w, z, g), ref)
    full4_rel = rel_l2(bf16x2_full4_g_into_w(x, w, z, g), ref)

    print(f"[seed {INPUT_SEED}] fp32 CONTROL vs reference        rel-L2 = {ctrl_rel:.3e}"
          f"   (validates seed/draw-order/dtype/eps; expect ~1e-7)")
    print(f"[seed {INPUT_SEED}] plain bf16 (rejected)            rel-L2 = {plain_rel:.3e}"
          f"   (~4e-3 scale check)")
    print(f"[seed {INPUT_SEED}] bf16x2 3-product g-into-w        rel-L2 = {giw_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 3-product g-on-activation rel-L2 = {goa_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 4-product (keeps lo*lo)   rel-L2 = {full4_rel:.3e}"
          f"   (sizes the dropped lo*lo term)\n")

    worst_primary = max(giw_rel, goa_rel)

    extra_seeds = [0, 21, 63, 84, 123, 2024]
    extra = []
    for s in extra_seeds:
        xs, ws, zs, gs = draw_inputs(s)
        rs = reference_forward(xs, ws, zs, gs)
        p_giw = rel_l2(bf16x2_g_into_w(xs, ws, zs, gs), rs)
        p_goa = rel_l2(bf16x2_g_on_act(xs, ws, zs, gs), rs)
        extra.append((s, p_giw, p_goa))
        print(f"[seed {s:4d}] bf16x2 g-into-w = {p_giw:.3e}   g-on-act = {p_goa:.3e}")

    worst_giw = max([giw_rel] + [p for _, p, _ in extra])
    worst_all = max([worst_primary] + [max(p, q) for _, p, q in extra])

    print("\n" + "=" * 66)
    print(f"fp32 control rel-L2 (must be << gate, validates model): {ctrl_rel:.3e}")
    print(f"WORST bf16x2 g-into-w rel-L2 (all seeds):               {worst_giw:.3e}")
    print(f"WORST bf16x2 3-product rel-L2 (all seeds, both g):      {worst_all:.3e}")
    print(f"on-device fp32 v1 rel-L2 (reference datum):             1.46e-5")
    print(f"NKIBench gate:                                          {REL_TOL:.1e}")
    verdict = "COMFORTABLY BELOW (<=1.5e-5)" if worst_all <= 1.5e-5 else (
        "MARGINAL/ABOVE (>1.5e-5)")
    print(f"no-spend decision input: worst={worst_all:.3e}  ->  {verdict}")
    print("=" * 66)


if __name__ == "__main__":
    main()
