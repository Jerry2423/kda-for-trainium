#!/usr/bin/env python3
"""Offline numpy pre-check for a compensated bf16x2 split on the gqa_full SCORE matmul.

Zero remote spend. Reproduces the EXACT input the remote gate scores (the adapter
seeds np.random.seed(seed) then draws q, k, v in that order; every profiler seed
reuses that same draw), computes the fp32 reference exactly as the NKIBench numpy
reference does, then models an idealized bf16x2 3-product split applied to the SCORE
matmul ONLY (q @ k^T), leaving the 1/sqrt(D) scale, the full-row softmax, and the
context matmul @ v in fp32. It reports the worst NKIBench-style relative-L2 of the
whole-attention output against the fp32 reference.

WHY the whole path (not the GEMM quadrature floor): softmax EXPONENTIATES the score.
A bf16 perturbation dS on the score becomes exp(S+dS) ~ exp(S)(1+dS) -- the sibling
pure-GEMM rel-L2 (~4.5e-6 on the score itself) does NOT transfer to the attention
output, because the softmax + context matmul reshape it non-linearly. So this sim
runs the FULL attention with only the score matmul split, to see the perturbation as
it actually reaches the output. The score-split correctness must be MEASURED through
this whole path, not inferred from a pure-GEMM floor; this offline sim is the no-spend
authorize/deny gate before the remote 5-seed run.)

    score[m,n] = sum_d q_h[m,d] * k_h[n,d]        # == (q_h @ k_h^T)[m,n], contract D
    A          = softmax_over_n( score / sqrt(D) )
    O          = A @ v_h

The kernel would build the score from q_t (d-on-partition) @ k_t (d-on-partition),
i.e. the same D-contraction; splitting q_h/k_h on their D axis == splitting q_t/k_t
after the (exact identity) transpose. This sim splits the raw (N,D) operands, matching
what the kernel's bf16 limbs would carry.

Idealized numpy RNE limb construction + exact fp32 accumulation is at least as
accurate as the hardware, so an offline result at/above the gate means the device
almost-certainly fails -- a practical no-spend gate, not an impossibility proof; a
remote full-5-seed PASS is still required to promote.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# gqa_full (NKIBench case): q,k,v natural layout, B=1 N=4096 QH=16 KH=8 D=128, n_rep=2.
B, N, QH, KH, D = 1, 4096, 16, 8, 128
N_REP = QH // KH
INPUT_SEED = 42          # adapter/nkibench_case.py DEFAULT_INPUT_SEED
REL_TOL = 2e-5           # adapter DEFAULT_REL_TOL (the NKIBench gate)
# gqa_full_v1's on-device fp32 floor (candidates.jsonl), for the device-quadrature
# prediction only. Softmax exponentiation means the device rel-L2 is NOT simply
# sqrt(floor^2 + offline^2) as in a pure GEMM -- the offline whole-path number already
# includes the exponentiation, so we report both and lean on the offline number.
FP32_FLOOR = 2.874266e-6

# Fail-closed: this sim must run against the real NKIBench reference draw. It draws
# its own inputs with the documented seed/shape, but we assert the reference file the
# gate uses still matches this sim's forward (guards a silent reference drift).
# NKIBench sits alongside the repo at ../AccelOpt/NKIBench (verify.py convention);
# this file is workspaces/gqa_full/runs/, so the repo root is parents[3] and its
# parent parents[4]. $NKIBENCH_ROOT overrides, matching verify.py.
NKIBENCH_ROOT = Path(
    os.environ.get("NKIBENCH_ROOT",
                   Path(__file__).resolve().parents[4] / "AccelOpt" / "NKIBench"))
_REF_REL = "reference/gqa_full_B1_N4096_QH16_KH8_D128_numpy_2.py"


def to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """Round fp32 -> bfloat16 (round-to-nearest-even), returned as fp32 values.

    bf16 keeps the fp32 sign+8-bit exponent and truncates the 23-bit mantissa to 7
    explicit bits (drops the low 16 bits of the fp32 pattern); RNE adds a tie-to-even
    bias before truncating. Inputs here are O(1) normals (no Inf/NaN); an assert guards.
    """
    x = np.asarray(x, dtype=np.float32)
    assert np.all(np.isfinite(x)), "bf16 RNE helper assumes finite O(1) inputs"
    u = x.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = (u + bias) >> np.uint32(16)
    return (rounded << np.uint32(16)).view(np.float32)


def split_bf16x2(x: np.ndarray):
    """Two-limb bf16 split: x ~= x_hi + x_lo, both bf16-valued fp32."""
    x = np.asarray(x, dtype=np.float32)
    x_hi = to_bf16_rne(x)
    x_lo = to_bf16_rne((x - x_hi).astype(np.float32))
    return x_hi, x_lo


def draw_inputs(seed: int):
    """Draw (q, k, v) exactly as the adapter's seeded natural get_inputs() does:
    np.random.seed(seed) then q ~ normal(0,1,(B,N,QH,D)), k,v ~ (B,N,KH,D), fp32."""
    np.random.seed(seed)
    q = np.random.normal(0.0, 1.0, size=(B, N, QH, D)).astype(np.float32)
    k = np.random.normal(0.0, 1.0, size=(B, N, KH, D)).astype(np.float32)
    v = np.random.normal(0.0, 1.0, size=(B, N, KH, D)).astype(np.float32)
    return q, k, v


def _heads(q, k, v):
    """Yield per-query-head (q_h[N,D], k_h[N,D], v_h[N,D]) with the reference's kv-repeat
    (xk = repeat(k, n_rep, axis=head); kv head of query head qh is qh // n_rep)."""
    for qh in range(QH):
        kh = qh // N_REP
        yield q[0, :, qh, :], k[0, :, kh, :], v[0, :, kh, :]


def _softmax_context(score_unscaled: np.ndarray, v_h: np.ndarray) -> np.ndarray:
    """Scale by 1/sqrt(D), full-row softmax over the key axis, then @ v_h -- all fp32.
    This is IDENTICAL for the reference and the split; only score_unscaled differs."""
    scale = np.float32(1.0 / np.sqrt(D))
    s = (score_unscaled * scale).astype(np.float32)
    s = s - np.max(s, axis=-1, keepdims=True)
    e = np.exp(s).astype(np.float32)
    a = (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)
    return np.matmul(a, v_h).astype(np.float32)


def output_fp32(q, k, v) -> np.ndarray:
    """fp32 reference output: score = q_h @ k_h^T (fp32), softmax, @ v_h; stacked heads."""
    outs = [_softmax_context(np.matmul(qh, kh.T).astype(np.float32), vh)
            for qh, kh, vh in _heads(q, k, v)]
    return np.stack(outs, axis=0)          # (QH, N, D)


def output_score_split(q, k, v, n_prod: int = 3) -> np.ndarray:
    """Output with the SCORE matmul done as a bf16x2 split; softmax + @v stay fp32.

    n_prod=3: hi@hi + hi@lo + lo@hi (the kernel's 3-product form, dropping lo@lo).
    n_prod=4: also add lo@lo (sizes the dropped term). n_prod=1: plain bf16 (rejected)."""
    outs = []
    for qh, kh, vh in _heads(q, k, v):
        q_hi, q_lo = split_bf16x2(qh)
        k_hi, k_lo = split_bf16x2(kh)
        s = np.matmul(q_hi, k_hi.T).astype(np.float32)
        if n_prod >= 3:
            s = (s + np.matmul(q_hi, k_lo.T) + np.matmul(q_lo, k_hi.T)).astype(np.float32)
        if n_prod >= 4:
            s = (s + np.matmul(q_lo, k_lo.T)).astype(np.float32)
        outs.append(_softmax_context(s, vh))
    return np.stack(outs, axis=0)


def rel_l2(v_k: np.ndarray, v_r: np.ndarray) -> float:
    """NKIBench relative-L2 over the flattened output: ||v_k - v_r||_2 / ||v_r||_2."""
    num = np.linalg.norm((v_k - v_r).ravel().astype(np.float64))
    den = np.linalg.norm(v_r.ravel().astype(np.float64))
    return float(num / den)


def _check_reference_unchanged() -> None:
    """Fail-closed guard: the NKIBench reference this sim mirrors must still exist and
    define the same forward (score/sqrt(D) -> max-shift softmax -> @v). Raises (not
    warns) so the gate cannot silently pass against a drifted reference; survives -O."""
    ref = NKIBENCH_ROOT / _REF_REL
    if not ref.is_file():
        raise FileNotFoundError(f"NKIBench reference not found: {ref}")
    src = ref.read_text()
    for token in ("np.sqrt(D)", "np.max(attention", "exp_attention", "attention @ xv",
                  "np.repeat(k, n_rep", "np.repeat(v, n_rep"):
        if token not in src:
            raise RuntimeError(
                f"reference {ref} no longer contains {token!r}; this offline sim's "
                f"forward may have drifted from the gate -- re-derive before trusting.")


def _unit_test_bf16() -> float:
    assert to_bf16_rne(np.float32(1.0)) == np.float32(1.0)
    assert to_bf16_rne(np.float32(1.0 + 2.0 ** -8)) == np.float32(1.0)         # tie->even
    assert to_bf16_rne(np.float32(1.0 + 3.0 * 2.0 ** -9)) == np.float32(1.0 + 2.0 ** -7)
    rng = np.random.default_rng(0)
    a = rng.normal(size=10000).astype(np.float32)
    a_hi, a_lo = split_bf16x2(a)
    rel = rel_l2(a_hi + a_lo, a)
    assert rel < 3e-5, rel
    return rel


def main() -> int:
    _check_reference_unchanged()
    self_test_rel = _unit_test_bf16()
    print(f"[self-test] bf16 RNE OK; bf16x2 reconstruction rel-L2 on N(0,1) = {self_test_rel:.3e}")
    print(f"[config] B={B} N={N} QH={QH} KH={KH} D={D} n_rep={N_REP}  input_seed={INPUT_SEED}  "
          f"gate rel_tol={REL_TOL:.1e}  fp32_floor(v1)={FP32_FLOOR:.2e}")
    print("[note] SCORE matmul split ONLY; scale + softmax + context matmul stay fp32.\n")

    q, k, v = draw_inputs(INPUT_SEED)
    ref = output_fp32(q, k, v)

    ctrl_rel = rel_l2(output_fp32(q, k, v), ref)                 # bit-exact 0.0 (formula match)
    plain_rel = rel_l2(output_score_split(q, k, v, 1), ref)      # plain bf16 score (rejected route)
    p3_rel = rel_l2(output_score_split(q, k, v, 3), ref)         # kernel's 3-product
    p4_rel = rel_l2(output_score_split(q, k, v, 4), ref)         # +lo@lo (dropped-term size)

    print(f"[seed {INPUT_SEED}] fp32 CONTROL vs reference      rel-L2 = {ctrl_rel:.3e}   (expect 0.0)")
    print(f"[seed {INPUT_SEED}] plain bf16 score (rejected)    rel-L2 = {plain_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 3-product score         rel-L2 = {p3_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 4-product score (lo@lo) rel-L2 = {p4_rel:.3e}\n")

    p3_all = [p3_rel]
    for s in (0, 21, 63, 84, 123, 2024):
        qs, ks, vs = draw_inputs(s)
        rf = output_fp32(qs, ks, vs)
        p3 = rel_l2(output_score_split(qs, ks, vs, 3), rf)
        p3_all.append(p3)
        print(f"[seed {s:4d}] bf16x2 3-product score = {p3:.3e}")
    worst_p3 = max(p3_all)

    print("\n" + "=" * 70)
    print(f"fp32 control rel-L2 (0.0 = seed/formula match):        {ctrl_rel:.3e}")
    print(f"WORST bf16x2 3-product score rel-L2 (all seeds):       {worst_p3:.3e}")
    print(f"on-device fp32 v1 floor (reference datum):             {FP32_FLOOR:.2e}")
    print(f"NKIBench gate:                                         {REL_TOL:.1e}")
    quad = float(np.sqrt(FP32_FLOOR ** 2 + worst_p3 ** 2))
    print(f"loose device-quadrature estimate sqrt(floor^2+off^2):  {quad:.3e}  "
          f"(offline already includes softmax exponentiation)")
    verdict = ("COMFORTABLY BELOW (<1.3e-5) -- authorizes the score-split kernel"
               if worst_p3 < 1.3e-5 else
               "MARGINAL/ABOVE (>=1.3e-5) -- does NOT authorize; record precision-floor datum")
    print(f"no-spend decision: worst bf16={worst_p3:.3e}  ->  {verdict}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
