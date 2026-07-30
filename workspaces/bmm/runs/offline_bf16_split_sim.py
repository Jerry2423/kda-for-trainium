#!/usr/bin/env python3
"""Offline numpy pre-check for a compensated bf16x2 split of the bmm main matmul.

Zero remote spend. Reproduces the EXACT input the remote gate scores (the adapter
seeds np.random.seed(42) once, then draws lhs, rhs in that order -- see
adapter/nkibench_case.py DEFAULT_INPUT_SEED and NKIBench/seeds/bmm.yaml), computes
the fp32 reference the way the NKIBench numpy reference does (np.matmul(lhs, rhs),
batched over B), then models an idealized bf16x2 compensated split-matmul on the
single main matmul and reports the worst NKIBench-style relative-L2 vs the fp32
reference.

Why bmm needs this even though the split won on the matmul-family siblings: bmm's
contraction is SHORT (K=64), so there is far less within-dot-product error
self-averaging than the K=1024 rmsnorm/matmul siblings had (their ~4.5e-6 came
partly from sqrt(K)-style cancellation over a long sum). And bmm's output is a RAW
matmul with NO downstream averaging (unlike rmsnorm's /K). So the per-element
rounding error is closer to a single-pass matmul rounding -- likely safe, but it
MUST be shown to clear the tightened gate rather than assumed from the siblings.

The split keeps each fp32 operand as two bf16 limbs and accumulates three bf16
products in fp32 (drops the negligible lo@lo cross term):
    a_hi = bf16(a),  a_lo = bf16(a - a_hi)          (round-to-nearest-even)
    b_hi = bf16(b),  b_lo = bf16(b - b_hi)
    a @ b ~= a_hi@b_hi + a_hi@b_lo + a_lo@b_hi
This is the IDEALIZED case (numpy RNE limb construction + exact fp32 accumulation is
at least as accurate as the hardware), so an offline result at/above the gate means
the hardware almost-certainly fails. A practical no-spend GREEN-LIGHT, not an
impossibility proof: promotion still needs a remote full-5-seed on-device PASS.

Green-light gate for a bf16x2 split attempt: worst-over-seeds offline rel-L2 <= 7e-6
(a MAXIMUM, not a target). 7e-6 combined in quadrature with a ~1e-5 fp32 floor stays
comfortably under the 2e-5 on-device gate.

Usage (from workspaces/bmm/, plain numpy -- no profiler venv needed):
  python3 runs/offline_bf16_split_sim.py [--m-rows N]
"""

from __future__ import annotations

import argparse

import numpy as np

# bmm case 2 shapes (NKIBench summary.json / seeds/bmm.yaml).
B, M, K, N = 16, 4096, 64, 4096
INPUT_SEED = 42            # adapter/nkibench_case.py DEFAULT_INPUT_SEED
REL_TOL = 2e-5            # adapter DEFAULT_REL_TOL (the NKIBench on-device gate)
BF16X2_GREENLIGHT = 7e-6  # worst-seed offline max to green-light a remote bf16x2 attempt


def to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """Round fp32 -> bfloat16 (round-to-nearest-even), returned as fp32 values.

    bf16 keeps the fp32 sign+8-bit exponent and truncates the 23-bit mantissa to 7
    explicit bits (drops the low 16 bits of the fp32 pattern). RNE adds a tie-to-even
    bias before truncating: bias = 0x7FFF + lsb(bit16). Carry into the exponent is
    handled by the integer add. Inputs here are finite normals (asserted).
    """
    x = np.asarray(x, dtype=np.float32)
    assert np.all(np.isfinite(x)), "bf16 RNE helper assumes finite inputs"
    u = x.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = (u + bias) >> np.uint32(16)
    bf16_as_u32 = rounded.astype(np.uint32) << np.uint32(16)
    return bf16_as_u32.view(np.float32)


def split_bf16x2(x: np.ndarray):
    """Two-limb bf16 split: x ~= x_hi + x_lo, both bf16-valued fp32 (~16 mantissa bits)."""
    x = np.asarray(x, dtype=np.float32)
    x_hi = to_bf16_rne(x)
    residual = (x - x_hi).astype(np.float32)
    x_lo = to_bf16_rne(residual)
    return x_hi, x_lo


def mm_bf16x2_3prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Idealized bf16x2 3-product split matmul, fp32 accumulation (models fp32 PSUM).

    hi@hi + hi@lo + lo@hi, dropping the negligible lo@lo term (the intended kernel).
    """
    a_hi, a_lo = split_bf16x2(a)
    b_hi, b_lo = split_bf16x2(b)
    return (np.matmul(a_hi, b_hi) + np.matmul(a_hi, b_lo)
            + np.matmul(a_lo, b_hi)).astype(np.float32)


def mm_bf16x2_4prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """bf16x2 with the dropped lo@lo term added back (reference for how much it buys)."""
    a_hi, a_lo = split_bf16x2(a)
    b_hi, b_lo = split_bf16x2(b)
    return (np.matmul(a_hi, b_hi) + np.matmul(a_hi, b_lo)
            + np.matmul(a_lo, b_hi) + np.matmul(a_lo, b_lo)).astype(np.float32)


def mm_bf16_plain(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Plain single-limb bf16 matmul (the rejected route) for scale reference."""
    return np.matmul(to_bf16_rne(a), to_bf16_rne(b)).astype(np.float32)


def mm_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """fp32 control: reproduces the reference exactly (rel-L2 must be 0)."""
    return np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float32)


def draw_inputs(seed: int):
    """Draw (lhs, rhs) exactly as the seeded reference get_inputs() does.

    seeds/bmm.yaml: lhs (B,M,K) drawn first, then rhs (B,K,N), both N(0,1) fp32.
    """
    np.random.seed(seed)
    lhs = np.random.normal(loc=0, scale=1.0, size=(B, M, K)).astype(np.float32)
    rhs = np.random.normal(loc=0, scale=1.0, size=(B, K, N)).astype(np.float32)
    return lhs, rhs


# candidate matmul kernels (bmm has a SINGLE matmul per batch, so one fn each)
CANDIDATES = [
    ("fp32 control (== ref)   ", mm_fp32),
    ("bf16x2 3-product        ", mm_bf16x2_3prod),
    ("bf16x2 4-product        ", mm_bf16x2_4prod),
    ("PLAIN bf16 (reject)     ", mm_bf16_plain),
]


def rel_l2_batched(mm_fn, lhs: np.ndarray, rhs: np.ndarray, m_rows: int) -> float:
    """NKIBench relative-L2 over the full (B,M,N) output, accumulated per-batch.

    Accumulates ||cand - ref||_2^2 and ||ref||_2^2 in float64 across batches so the
    exact full-output rel-L2 = sqrt(num/den) is computed without holding the ~1 GB
    output at once. m_rows<M subsamples the M axis for speed: rows are iid, so the
    rel-L2 is a per-element statistic that converges long before M=4096 (it is the
    same population value with slightly more sampling noise). The K=64 contraction --
    the only axis that shapes the per-element rounding error -- is ALWAYS full.
    """
    num_sq = 0.0
    den_sq = 0.0
    for b in range(B):
        a = lhs[b, :m_rows, :]        # (m_rows, K)   K always full
        bb = rhs[b, :, :]             # (K, N)        N always full
        ref = mm_fp32(a, bb)
        cand = mm_fn(a, bb)
        diff = (cand - ref).astype(np.float64)
        num_sq += float(np.dot(diff.ravel(), diff.ravel()))
        r = ref.astype(np.float64)
        den_sq += float(np.dot(r.ravel(), r.ravel()))
    return float(np.sqrt(num_sq / den_sq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-rows", type=int, default=1024,
                    help="M rows to sample per batch (K always full=64; rows are iid "
                         "so rel-L2 converges well below M=4096). Default 1024.")
    args = ap.parse_args()
    m_rows = min(args.m_rows, M)

    print(f"[config] B={B} M={M} K={K} N={N}  (sampling m_rows={m_rows}/{M} per batch, "
          f"K + N always full)  input_seed={INPUT_SEED}")
    print(f"[config] on-device gate rel_tol={REL_TOL:.1e}   bf16x2 green-light <= {BF16X2_GREENLIGHT:.1e}\n")

    # Primary: the exact scored input (seed 42). Also NKIBench robustness seeds.
    seeds = [INPUT_SEED, 0, 21, 63, 84]
    worst = {name: 0.0 for name, _ in CANDIDATES}

    for s in seeds:
        lhs, rhs = draw_inputs(s)
        print(f"[seed {s:3d}]")
        for name, fn in CANDIDATES:
            r = rel_l2_batched(fn, lhs, rhs, m_rows)
            worst[name] = max(worst[name], r)
            print(f"           {name}  rel-L2 = {r:.3e}")
        print()

    print("=" * 72)
    print(f"WORST-over-seeds rel-L2 per candidate:")
    print(f"  (bf16x2 3-product green-light = <= {BF16X2_GREENLIGHT:.1e}; on-device NKIBench gate = {REL_TOL:.1e})")
    for name, _ in CANDIDATES:
        w = worst[name]
        if "3-product" in name:
            verdict = (f"PASS greenlight (<= {BF16X2_GREENLIGHT:.0e})" if w <= BF16X2_GREENLIGHT
                       else (f"over greenlight (> {BF16X2_GREENLIGHT:.0e}, < 2e-5)" if w < REL_TOL
                             else "FAIL on-device gate (>= 2e-5)"))
        else:
            verdict = ("(ref, expect ~0)" if "control" in name else
                       ("under on-device gate" if w < REL_TOL else "over on-device gate"))
        print(f"  {name}  worst = {w:.3e}   {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
