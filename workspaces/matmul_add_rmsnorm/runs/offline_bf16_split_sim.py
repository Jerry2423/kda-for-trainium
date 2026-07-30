#!/usr/bin/env python3
"""Offline numpy pre-check for a compensated bf16x2 split matmul_add_rmsnorm.

Zero remote spend. Reproduces the EXACT input the remote gate scores (the adapter
seeds np.random.seed(seed) before drawing x, w, eps, z, g in that order; every
profiler seed reuses that same draw), computes the fp32 reference the way the
NKIBench numpy reference does, then models an idealized bf16x2 compensated
split-matmul and reports the worst NKIBench-style relative-L2 against the fp32
reference.

This op is the MIRROR of the rmsnorm siblings: they do norm -> GEMM (bf16 error
enters ONLY the matmul; inv_rms is computed from the exact fp32 activation and
commutes out as a post-scale). THIS op does GEMM -> add -> norm, so:

    y[m,n]     = sum_k x[m,k] * w[k,n]  +  z[m,n]        # matmul (bf16x2) then +z
    inv_rms[m] = 1 / sqrt( mean_n( y[m,n]^2 ) + eps )    # per-row, over N
    out[m,n]   = y[m,n] * g[n] * inv_rms[m]              # per-N g, per-row 1/rms

The bf16 matmul error lands in y, and y feeds BOTH the norm (inv_rms) AND the
output numerator -- a composite path the sibling sims never exercised. This sim's
whole point is to measure that composite rel-L2, because the error partly cancels
(a coherent relative perturbation d in y scales the numerator by ~d and inv_rms by
~-d, so out = y*inv_rms is first-order insensitive to a common-mode scaling of y)
and we need the measured residual, not a hand-wave.

g placement: g is on the OUTPUT (free) axis N here, applied AFTER the norm, so it
does NOT commute into w (folding g[n] into w[k,n] would scale y before the norm and
break rms = sqrt(mean(y^2))). g stays a plain free-axis output multiply -- unlike
the sibling's per-K g which folded into the resident weight. So there is only ONE
placement to check here.

The split keeps each fp32 operand as two bf16 limbs and accumulates three bf16
products in fp32, dropping the negligible lo*lo term:
    x_hi = bf16(x),   x_lo = bf16(x - x_hi)             (round-to-nearest-even)
    w_hi = bf16(w),   w_lo = bf16(w - w_hi)
    prod ~= x_hi@w_hi + x_hi@w_lo + x_lo@w_hi

Idealized numpy RNE limb construction + exact fp32 accumulation is at least as
accurate as the hardware, so an offline result at/above the gate means the device
almost-certainly fails. It is a practical no-spend gate, not an impossibility proof.

CONTEXT (why the margin matters here): the on-device fp32 v1 already measures rel-L2
~1.5e-5 by analogy to the fp32 siblings (add_rmsnorm_matmul_v1 = 1.46e-5), only
~1.3x under the 2e-5 gate -- the trn2 PE array is bf16-native and its "fp32" matmul
is itself a multi-pass bf16 emulation. The device rel-L2 combines the fp32 floor
and the bf16 error in QUADRATURE (sibling: sqrt(1.46^2 + 4.4^2)e-6 = 1.528e-5), so
this sim's bf16-only number is the term added under the root, not the final gate
value. Gate carefully: if the bf16-only number here approaches ~1.3e-5 the device
quadrature could exceed 2e-5.
"""

from __future__ import annotations

import numpy as np

M, N, K = 4096, 2048, 2048
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
    (no draw), z ~ normal(0,1,(M,N)), g ~ normal(0,1,(N,)), all fp32, in that order,
    after a single np.random.seed(seed). eps is a python float so it does NOT advance
    the RNG; the draw order is x -> w -> z -> g. Note z is (M,N) and g is (N,) here
    (the mirror op: z/g live on the OUTPUT axes, not the contraction axis).
    """
    np.random.seed(seed)
    x = np.random.normal(loc=0.0, scale=1.0, size=(M, K)).astype(np.float32)
    w = np.random.normal(loc=0.0, scale=1.0, size=(K, N)).astype(np.float32)
    z = np.random.normal(loc=0.0, scale=1.0, size=(M, N)).astype(np.float32)
    g = np.random.normal(loc=0.0, scale=1.0, size=(N,)).astype(np.float32)
    return x, w, z, g


def reference_forward(x, w, z, g) -> np.ndarray:
    """The NKIBench numpy reference for matmul_add_rmsnorm (exact mirror)."""
    y = (np.matmul(x, w) + z).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(y), axis=-1, keepdims=True) + EPS).astype(np.float32)
    return (y * g / rms).astype(np.float32)


def _norm_and_scale(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Shared epilogue: out = y * g / sqrt(mean_n(y^2) + eps), per the reference.

    y is whatever matmul-output (+z) we produced; the norm is computed FROM y, so any
    bf16 error in y propagates into inv_rms here as well as into the numerator.
    """
    rms = np.sqrt(np.mean(np.square(y), axis=-1, keepdims=True) + EPS).astype(np.float32)
    return (y * g / rms).astype(np.float32)


def fp32_control(x, w, z, g) -> np.ndarray:
    """fp32 control: exact matmul + z, then the shared norm+scale epilogue.

    Algebraically identical to the reference; reproducing it to ~1e-7 confirms
    seed/draw-order/dtype/formula/eps all match before the bf16x2 number is trusted.
    """
    y = (np.matmul(x, w) + z).astype(np.float32)
    return _norm_and_scale(y, g)


def bf16x2_3prod(x, w, z, g) -> np.ndarray:
    """Idealized bf16x2 3-product split matmul, then +z, then the norm+scale epilogue.

    The bf16 error lives ONLY in the matmul; z, the norm reduction, and the output
    scale are all fp32. y (= approx matmul + z) then feeds both inv_rms and the
    numerator -- the composite path this sim exists to measure.
    """
    x_hi, x_lo = split_bf16x2(x)
    w_hi, w_lo = split_bf16x2(w)
    prod = (np.matmul(x_hi, w_hi) + np.matmul(x_hi, w_lo)
            + np.matmul(x_lo, w_hi)).astype(np.float32)
    y = (prod + z).astype(np.float32)
    return _norm_and_scale(y, g)


def bf16x2_4prod(x, w, z, g) -> np.ndarray:
    """Reference-only: full 4-product split (keeps lo*lo) to size the dropped term."""
    x_hi, x_lo = split_bf16x2(x)
    w_hi, w_lo = split_bf16x2(w)
    prod = (np.matmul(x_hi, w_hi) + np.matmul(x_hi, w_lo)
            + np.matmul(x_lo, w_hi) + np.matmul(x_lo, w_lo)).astype(np.float32)
    y = (prod + z).astype(np.float32)
    return _norm_and_scale(y, g)


def plain_bf16(x, w, z, g) -> np.ndarray:
    """Reference-only: plain single-limb bf16 matmul (the rejected route) for scale."""
    x_hi = to_bf16_rne(x)
    w_hi = to_bf16_rne(w)
    y = (np.matmul(x_hi, w_hi).astype(np.float32) + z).astype(np.float32)
    return _norm_and_scale(y, g)


def rel_l2(v_k: np.ndarray, v_r: np.ndarray) -> float:
    """NKIBench relative-L2 over the flattened output: ||v_k - v_r||_2 / ||v_r||_2."""
    num = np.linalg.norm((v_k - v_r).ravel().astype(np.float64))
    den = np.linalg.norm(v_r.ravel().astype(np.float64))
    return float(num / den)


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
    plain_rel = rel_l2(plain_bf16(x, w, z, g), ref)
    p3_rel = rel_l2(bf16x2_3prod(x, w, z, g), ref)
    p4_rel = rel_l2(bf16x2_4prod(x, w, z, g), ref)

    print(f"[seed {INPUT_SEED}] fp32 CONTROL vs reference   rel-L2 = {ctrl_rel:.3e}"
          f"   (validates seed/draw-order/dtype/eps; expect ~1e-7)")
    print(f"[seed {INPUT_SEED}] plain bf16 (rejected)       rel-L2 = {plain_rel:.3e}"
          f"   (scale check; expect ~1e-3)")
    print(f"[seed {INPUT_SEED}] bf16x2 3-product            rel-L2 = {p3_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 4-product (keeps lo*lo) rel-L2 = {p4_rel:.3e}"
          f"   (sizes the dropped lo*lo term)\n")

    extra_seeds = [0, 21, 63, 84, 123, 2024]
    p3_all = [p3_rel]
    for s in extra_seeds:
        xs, ws, zs, gs = draw_inputs(s)
        rs = reference_forward(xs, ws, zs, gs)
        p3 = rel_l2(bf16x2_3prod(xs, ws, zs, gs), rs)
        p3_all.append(p3)
        print(f"[seed {s:4d}] bf16x2 3-product = {p3:.3e}")

    worst_p3 = max(p3_all)

    print("\n" + "=" * 66)
    print(f"fp32 control rel-L2 (must be << gate, validates model): {ctrl_rel:.3e}")
    print(f"WORST bf16x2 3-product rel-L2 (all seeds):              {worst_p3:.3e}")
    print(f"on-device fp32 sibling v1 rel-L2 (reference datum):     1.46e-5")
    print(f"NKIBench gate:                                         {REL_TOL:.1e}")
    # Predict the device quadrature: sqrt(fp32_floor^2 + bf16^2). Use the sibling's
    # measured fp32 floor 1.46e-5 as the plug (this op's v1 floor is expected similar).
    floor = 1.46e-5
    quad = float(np.sqrt(floor ** 2 + worst_p3 ** 2))
    print(f"predicted device quadrature sqrt(1.46e-5^2 + bf16^2):  {quad:.3e}")
    verdict = "COMFORTABLY BELOW (<=1.5e-5)" if worst_p3 <= 1.5e-5 else (
        "MARGINAL/ABOVE (>1.5e-5) -- gate carefully")
    print(f"no-spend decision input: worst bf16={worst_p3:.3e}  ->  {verdict}")
    print(f"                         predicted device quad={quad:.3e} vs gate {REL_TOL:.1e}")
    print("=" * 66)


if __name__ == "__main__":
    main()
