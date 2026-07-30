#!/usr/bin/env python3
"""Offline numpy pre-check for a compensated bf16x2 split of the fused SwiGLU op.

Zero remote spend. Reproduces the EXACT input the remote gate scores (the adapter
seeds np.random.seed(42) once, then draws x, w_up, w_down, w_gate in that order --
see adapter/nkibench_case.py DEFAULT_INPUT_SEED and NKIBench/seeds/swiglu.yaml),
computes the fp32 reference the way the NKIBench numpy reference does, then models
an idealized bf16x2 compensated split-matmul on EACH of the three GEMMs and reports
the worst NKIBench-style relative-L2 against the fp32 reference.

Why this matters for swiglu specifically: the siblings (rmsnorm_matmul,
add_rmsnorm_matmul) proved bf16x2 on a SINGLE GEMM lands ~4.5e-6 << the 2e-5 gate.
swiglu has THREE chained GEMMs (up, gate, down) plus a SiLU nonlinearity between
them, so bf16x2 rounding error can COMPOUND. This sim answers: does the compounded
error still clear 2e-5? And which GEMMs must stay fp32 if it doesn't?

The split keeps each fp32 operand as two bf16 limbs and accumulates three bf16
products in fp32 (drops the negligible lo@lo cross term):
    a_hi = bf16(a),  a_lo = bf16(a - a_hi)          (round-to-nearest-even)
    b_hi = bf16(b),  b_lo = bf16(b - b_hi)
    a @ b ~= a_hi@b_hi + a_hi@b_lo + a_lo@b_hi
This is the IDEALIZED case (numpy RNE limb construction + exact fp32 accumulation is
at least as accurate as the hardware), so an offline result at/above the gate means
the hardware almost-certainly fails. A practical no-spend gate, not an impossibility
proof.
"""

from __future__ import annotations

import numpy as np

M, N, K = 4096, 3072, 1024
INPUT_SEED = 42          # adapter/nkibench_case.py DEFAULT_INPUT_SEED
REL_TOL = 2e-5           # adapter DEFAULT_REL_TOL (the NKIBench gate)


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


def mm_bf16x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Idealized bf16x2 3-product split matmul, fp32 accumulation (models fp32 PSUM)."""
    a_hi, a_lo = split_bf16x2(a)
    b_hi, b_lo = split_bf16x2(b)
    return (np.matmul(a_hi, b_hi) + np.matmul(a_hi, b_lo)
            + np.matmul(a_lo, b_hi)).astype(np.float32)


def mm_bf16_plain(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Plain single-limb bf16 matmul (the rejected route) for scale reference."""
    return np.matmul(to_bf16_rne(a), to_bf16_rne(b)).astype(np.float32)


def draw_inputs(seed: int):
    """Draw (x, w_up, w_down, w_gate) exactly as the seeded reference get_inputs() does."""
    np.random.seed(seed)
    x = np.random.normal(loc=0, scale=1.0, size=(M, K)).astype(np.float32)
    w_up = np.random.normal(loc=0, scale=1.0, size=(K, N)).astype(np.float32)
    w_down = np.random.normal(loc=0, scale=1.0, size=(N, K)).astype(np.float32)
    w_gate = np.random.normal(loc=0, scale=1.0, size=(K, N)).astype(np.float32)
    return x, w_up, w_down, w_gate


def silu(g: np.ndarray) -> np.ndarray:
    """SiLU / swish: g * sigmoid(g) = g / (1 + exp(-g)); matches the reference exactly."""
    return (g / (1.0 + np.exp(-g))).astype(np.float32)


def reference_forward(x, w_up, w_down, w_gate) -> np.ndarray:
    """The NKIBench numpy reference for swiglu (all fp32)."""
    up = np.matmul(x, w_up)
    gate = np.matmul(x, w_gate)
    h = silu(gate) * up
    return np.matmul(h, w_down).astype(np.float32)


def rel_l2(v_k: np.ndarray, v_r: np.ndarray) -> float:
    """NKIBench relative-L2 over the flattened output: ||v_k - v_r||_2 / ||v_r||_2."""
    num = np.linalg.norm((v_k - v_r).ravel().astype(np.float64))
    den = np.linalg.norm(v_r.ravel().astype(np.float64))
    return float(num / den)


# --- candidate pipelines: which of the 3 GEMMs use bf16x2 vs stay fp32 ---

def all_bf16x2(x, w_up, w_down, w_gate) -> np.ndarray:
    """All three GEMMs bf16x2 (the aggressive target: max PE savings)."""
    up = mm_bf16x2(x, w_up)
    gate = mm_bf16x2(x, w_gate)
    h = silu(gate) * up
    return mm_bf16x2(h, w_down)


def updown_bf16x2_gate_fp32(x, w_up, w_down, w_gate) -> np.ndarray:
    """up+down bf16x2, gate fp32 (gate feeds the nonlinearity -> most error-sensitive)."""
    up = mm_bf16x2(x, w_up)
    gate = np.matmul(x, w_gate).astype(np.float32)
    h = silu(gate) * up
    return mm_bf16x2(h, w_down)


def upgate_bf16x2_down_fp32(x, w_up, w_down, w_gate) -> np.ndarray:
    """up+gate bf16x2, down fp32 (down is the last GEMM -> its input h is already lossy)."""
    up = mm_bf16x2(x, w_up)
    gate = mm_bf16x2(x, w_gate)
    h = silu(gate) * up
    return np.matmul(h, w_down).astype(np.float32)


def only_down_bf16x2(x, w_up, w_down, w_gate) -> np.ndarray:
    """Only the down GEMM bf16x2 (up+gate fp32); down alone is ~1/3 of PE MACs."""
    up = np.matmul(x, w_up).astype(np.float32)
    gate = np.matmul(x, w_gate).astype(np.float32)
    h = silu(gate) * up
    return mm_bf16x2(h, w_down)


def all_plain_bf16(x, w_up, w_down, w_gate) -> np.ndarray:
    """All three GEMMs plain bf16 (rejected route) for scale."""
    up = mm_bf16_plain(x, w_up)
    gate = mm_bf16_plain(x, w_gate)
    h = silu(gate) * up
    return mm_bf16_plain(h, w_down)


CANDIDATES = [
    ("all 3 bf16x2            ", all_bf16x2),
    ("up+down bf16x2, gate fp32", updown_bf16x2_gate_fp32),
    ("up+gate bf16x2, down fp32", upgate_bf16x2_down_fp32),
    ("only down bf16x2        ", only_down_bf16x2),
    ("all 3 PLAIN bf16 (reject)", all_plain_bf16),
]


def main():
    print(f"[config] M={M} N={N} K={K}  input_seed={INPUT_SEED}  gate rel_tol={REL_TOL:.1e}\n")

    # Primary: the exact scored input (seed 42). Also robustness seeds.
    seeds = [INPUT_SEED, 0, 21, 63, 84]
    worst = {name: 0.0 for name, _ in CANDIDATES}

    for s in seeds:
        x, w_up, w_down, w_gate = draw_inputs(s)
        ref = reference_forward(x, w_up, w_down, w_gate)
        # fp32 control: numpy fp32 reproduces itself (sanity that the ref path is stable)
        print(f"[seed {s:3d}] ||ref||={np.linalg.norm(ref.ravel()):.4e}")
        for name, fn in CANDIDATES:
            r = rel_l2(fn(x, w_up, w_down, w_gate), ref)
            worst[name] = max(worst[name], r)
            print(f"           {name}  rel-L2 = {r:.3e}")
        print()

    print("=" * 68)
    print(f"WORST-over-seeds rel-L2 per candidate (gate = {REL_TOL:.1e}):")
    for name, _ in CANDIDATES:
        w = worst[name]
        verdict = "PASS (<=1.5e-5, comfortable)" if w <= 1.5e-5 else (
            "MARGINAL (1.5e-5..2e-5)" if w < REL_TOL else "FAIL (>=2e-5)")
        print(f"  {name}  worst = {w:.3e}   {verdict}")
    print("=" * 68)


if __name__ == "__main__":
    main()
