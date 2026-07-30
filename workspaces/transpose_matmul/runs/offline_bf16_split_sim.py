#!/usr/bin/env python3
"""Offline numpy pre-check for a compensated bf16x2 split transpose_matmul.

Zero remote spend. Reproduces the EXACT input the remote gate scores (the adapter
seeds np.random.seed(seed) before drawing lhs, rhs in that order; every profiler
seed reuses that same draw), computes the fp32 reference the way the NKIBench numpy
reference does, then models an idealized bf16x2 compensated split-matmul and reports
the worst NKIBench-style relative-L2 against the fp32 reference.

This op is a PURE GEMM -- the simplest member of the bf16x2 family. There is no
residual add, no RMSNorm, no g scale (contrast the matmul_add_rmsnorm / rmsnorm_matmul
siblings, whose norm epilogues fed the bf16 error into inv_rms as well as the
numerator). Here the bf16 error enters ONLY the matmul and flows straight to the
output, so the rel-L2 is the plain GEMM split error -- no norm self-cancellation, no
composite path:

    out[m,n] = sum_k lhs[k,m] * rhs[k,n]        # == (lhs^T @ rhs)[m,n]

The reference does lhs_t = transpose(lhs) then lhs_t @ rhs; the transpose is an exact
axis permutation, so splitting lhs before or after the transpose is identical (bf16
rounding is element-wise). This sim splits the raw K-major operands, matching the
kernel, which builds its bf16 limbs directly from the K-on-partition loaded tiles.

The split keeps each fp32 operand as two bf16 limbs and accumulates three bf16
products in fp32, dropping the negligible lo*lo term:
    lhs_hi = bf16(lhs),  lhs_lo = bf16(lhs - lhs_hi)   (round-to-nearest-even)
    rhs_hi = bf16(rhs),  rhs_lo = bf16(rhs - rhs_hi)
    prod ~= lhs_hi^T@rhs_hi + lhs_hi^T@rhs_lo + lhs_lo^T@rhs_hi

Idealized numpy RNE limb construction + exact fp32 accumulation is at least as
accurate as the hardware, so an offline result at/above the gate means the device
almost-certainly fails. It is a practical no-spend gate, not an impossibility proof.

CONTEXT (why the margin is expected comfortable here): tmm_v1's on-device fp32 floor
is measured 3.99e-7 -- TINY, the pure-GEMM regime (swiglu 6.36e-7, rmsnorm_matmul
4.8e-7), NOT the add_rmsnorm family's ~1.46e-5 (which carries a RMSNorm square-reduce
feedback this op lacks). The device rel-L2 combines the fp32 floor and the bf16 error
in QUADRATURE (sqrt(floor^2 + bf16^2)); with a 3.99e-7 floor the bf16 term dominates
outright, so the predicted on-device rel-L2 ~= the offline bf16 number itself
(~4.5e-6, the pure-GEMM family value), ~4.5x under the 2e-5 gate.
"""

from __future__ import annotations

import numpy as np

# transpose_matmul (NKIBench case): lhs (K,M), rhs (K,N), out (M,N) = lhs^T @ rhs.
M, N, K = 4096, 10944, 2048
INPUT_SEED = 42          # adapter/nkibench_case.py DEFAULT_INPUT_SEED
REL_TOL = 2e-5           # adapter DEFAULT_REL_TOL (the NKIBench gate)
# tmm_v1's on-device fp32 floor (profile/tmm_v1_digest.md), used only to predict the
# device quadrature; the bf16 term dominates it here.
FP32_FLOOR = 3.99e-7


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
    bf16_as_u32 = rounded << np.uint32(16)
    return bf16_as_u32.view(np.float32)


def split_bf16x2(x: np.ndarray):
    """Two-limb bf16 split of an fp32 array: x ~= x_hi + x_lo, both bf16-valued fp32."""
    x = np.asarray(x, dtype=np.float32)
    x_hi = to_bf16_rne(x)
    residual = (x - x_hi).astype(np.float32)
    x_lo = to_bf16_rne(residual)
    return x_hi, x_lo


def draw_inputs(seed: int):
    """Draw (lhs, rhs) exactly as the adapter's seeded natural get_inputs() does.

    reference get_inputs(): lhs ~ normal(0,1,(K,M)), rhs ~ normal(0,1,(K,N)), both
    fp32, in that order, after a single np.random.seed(seed). Both operands are
    K-major (contraction axis first) -- the reference transposes lhs inside forward().
    """
    np.random.seed(seed)
    lhs = np.random.normal(loc=0.0, scale=1.0, size=(K, M)).astype(np.float32)
    rhs = np.random.normal(loc=0.0, scale=1.0, size=(K, N)).astype(np.float32)
    return lhs, rhs


def reference_forward(lhs, rhs) -> np.ndarray:
    """The NKIBench numpy reference for transpose_matmul: transpose(lhs) @ rhs."""
    lhs_t = np.transpose(lhs, axes=(1, 0))
    return np.matmul(lhs_t, rhs).astype(np.float32)


def fp32_control(lhs, rhs) -> np.ndarray:
    """fp32 control: lhs^T @ rhs, the same fp32 GEMM the reference computes.

    `lhs.T` and the reference's `np.transpose(lhs, (1,0))` are the identical 2-D
    operation on the same drawn operands, so the control is BIT-IDENTICAL to the
    reference and its rel-L2 is exactly 0.0. That exact-zero is the strongest form
    of the seed/draw-order/dtype/formula check: any deviation (a mis-scaled draw,
    wrong shape, wrong accumulation dtype, or a transpose applied to the wrong axis)
    would make this differ from the reference. It is the fp32 anchor for the bf16x2
    numbers, not an independent-precision cross-check.
    """
    return np.matmul(lhs.T, rhs).astype(np.float32)


def bf16x2_3prod(lhs, rhs) -> np.ndarray:
    """Idealized bf16x2 3-product split matmul (the kernel's exact accumulation).

    Splits both K-major operands, then accumulates hi@hi + hi@lo + lo@hi in fp32,
    dropping the negligible lo@lo cross term. Matches nc_matmul(stationary, moving)
    = stationary^T @ moving with the pinned order lhs_hi@rhs_hi, lhs_hi@rhs_lo,
    lhs_lo@rhs_hi.
    """
    lhs_hi, lhs_lo = split_bf16x2(lhs)
    rhs_hi, rhs_lo = split_bf16x2(rhs)
    prod = (np.matmul(lhs_hi.T, rhs_hi) + np.matmul(lhs_hi.T, rhs_lo)
            + np.matmul(lhs_lo.T, rhs_hi)).astype(np.float32)
    return prod


def bf16x2_4prod(lhs, rhs) -> np.ndarray:
    """Reference-only: full 4-product split (keeps lo@lo) to size the dropped term."""
    lhs_hi, lhs_lo = split_bf16x2(lhs)
    rhs_hi, rhs_lo = split_bf16x2(rhs)
    prod = (np.matmul(lhs_hi.T, rhs_hi) + np.matmul(lhs_hi.T, rhs_lo)
            + np.matmul(lhs_lo.T, rhs_hi) + np.matmul(lhs_lo.T, rhs_lo)).astype(np.float32)
    return prod


def plain_bf16(lhs, rhs) -> np.ndarray:
    """Reference-only: plain single-limb bf16 matmul (the rejected route) for scale."""
    lhs_hi = to_bf16_rne(lhs)
    rhs_hi = to_bf16_rne(rhs)
    return np.matmul(lhs_hi.T, rhs_hi).astype(np.float32)


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
    rel = rel_l2(a_hi + a_lo, a)
    assert rel < 3e-5, rel
    return rel


def main():
    self_test_rel = _unit_test_bf16()
    print(f"[self-test] bf16 RNE helper OK; bf16x2 reconstruction rel-L2 on N(0,1) = {self_test_rel:.3e}")
    print(f"[config] M={M} N={N} K={K}  input_seed={INPUT_SEED}  "
          f"gate rel_tol={REL_TOL:.1e}  fp32_floor={FP32_FLOOR:.2e}\n")

    lhs, rhs = draw_inputs(INPUT_SEED)
    ref = reference_forward(lhs, rhs)

    ctrl_rel = rel_l2(fp32_control(lhs, rhs), ref)
    plain_rel = rel_l2(plain_bf16(lhs, rhs), ref)
    p3_rel = rel_l2(bf16x2_3prod(lhs, rhs), ref)
    p4_rel = rel_l2(bf16x2_4prod(lhs, rhs), ref)

    print(f"[seed {INPUT_SEED}] fp32 CONTROL vs reference   rel-L2 = {ctrl_rel:.3e}"
          f"   (validates seed/draw-order/dtype/formula; expect 0.0 bit-exact)")
    print(f"[seed {INPUT_SEED}] plain bf16 (rejected)       rel-L2 = {plain_rel:.3e}"
          f"   (scale check; expect ~1e-3)")
    print(f"[seed {INPUT_SEED}] bf16x2 3-product            rel-L2 = {p3_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 4-product (keeps lo@lo) rel-L2 = {p4_rel:.3e}"
          f"   (sizes the dropped lo@lo term)\n")

    extra_seeds = [0, 21, 63, 84, 123, 2024]
    p3_all = [p3_rel]
    for s in extra_seeds:
        ls, rs_ = draw_inputs(s)
        rf = reference_forward(ls, rs_)
        p3 = rel_l2(bf16x2_3prod(ls, rs_), rf)
        p3_all.append(p3)
        print(f"[seed {s:4d}] bf16x2 3-product = {p3:.3e}")

    worst_p3 = max(p3_all)

    print("\n" + "=" * 66)
    print(f"fp32 control rel-L2 (0.0 bit-exact = seed/formula match): {ctrl_rel:.3e}")
    print(f"WORST bf16x2 3-product rel-L2 (all seeds):              {worst_p3:.3e}")
    print(f"on-device fp32 v1 floor (reference datum):             {FP32_FLOOR:.2e}")
    print(f"NKIBench gate:                                         {REL_TOL:.1e}")
    # Predict the device quadrature: sqrt(fp32_floor^2 + bf16^2). With the tiny 3.99e-7
    # pure-GEMM floor the bf16 term dominates, so quad ~= worst_p3.
    quad = float(np.sqrt(FP32_FLOOR ** 2 + worst_p3 ** 2))
    print(f"predicted device quadrature sqrt(floor^2 + bf16^2):    {quad:.3e}")
    # "Comfortably below": worst 3-product < 1.3e-5 (the point where the predicted device
    # quadrature could start to approach the 2e-5 gate). At/above that records a
    # precision-floor datum instead of authorizing the split kernel.
    verdict = "COMFORTABLY BELOW (<1.3e-5) -- authorizes the split kernel" if worst_p3 < 1.3e-5 else (
        "MARGINAL/ABOVE (>=1.3e-5) -- does NOT authorize; record precision-floor datum")
    print(f"no-spend decision input: worst bf16={worst_p3:.3e}  ->  {verdict}")
    print(f"                         predicted device quad={quad:.3e} vs gate {REL_TOL:.1e}")
    print("=" * 66)


if __name__ == "__main__":
    main()
