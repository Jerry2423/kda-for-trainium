#!/usr/bin/env python3
"""Offline numpy pre-check for a compensated bf16x2 split of the matmul GEMM.

Zero remote spend. Reproduces the EXACT input the remote gate scores, computes the
fp32 reference the NKIBench way, then models an idealized bf16x2 compensated
split-matmul and reports the NKIBench-style relative-L2 against the fp32 reference.

IMPORTANT — what the remote gate actually scores: the NKIBench adapter
(adapter/nkibench_case.py) hard-codes np.random.seed(DEFAULT_INPUT_SEED=42) before
every natural draw, so ALL FIVE on-device profiler correctness seeds [0,21,42,63,84]
score the SAME seed-42 input tensors. The single gate-reproducing datum is therefore
the seed-42 draw. The extra draws [0,21,63,84] simulated below are an INPUT-DIVERSITY
robustness check on distinct inputs the remote gate does NOT run (per the
compensated-bf16x2 lesson, the offline sim's distinct draws are the real
input-diversity correctness margin, since the on-device 5-seed PASS reuses one
input); they are reported separately and never presented as the remote gate.

This op is a PURE dense GEMM out = lhs @ rhs (M=4096, K=5120, N=12288), the same
family member as transpose_matmul / the swiglu down-GEMM: no residual add, no
RMSNorm epilogue, so the bf16 error enters ONLY the matmul and flows straight to
the output. The rel-L2 is therefore the plain GEMM split error -- no norm
self-cancellation, no composite path.

    out[m,n] = sum_k lhs[m,k] * rhs[k,n]

The kernel transposes lhs (k onto the partition axis) before nc_matmul, but the
bf16 split is element-wise, so splitting the raw operands (as here) is identical
to splitting the transposed tiles the kernel builds.

The split keeps each fp32 operand as two bf16 limbs and accumulates three bf16
products in fp32, dropping the negligible lo*lo term:
    lhs_hi = bf16(lhs),  lhs_lo = bf16(lhs - lhs_hi)   (round-to-nearest-even)
    rhs_hi = bf16(rhs),  rhs_lo = bf16(rhs - rhs_hi)
    prod ~= lhs_hi@rhs_hi + lhs_hi@rhs_lo + lhs_lo@rhs_hi

Idealized numpy RNE limb construction + exact fp32 accumulation is at least as
accurate as the hardware, so an offline result at/above the gate means the device
almost-certainly fails. It is a practical no-spend gate, not an impossibility proof.

CONTEXT (why the margin is expected comfortable): matmul_v2_b4's on-device fp32
floor is measured worst rel-L2 = 4.207e-7 (layout check) -- TINY, the pure-GEMM
regime (transpose_matmul 3.99e-7, swiglu 6.36e-7), NOT the add_rmsnorm family's
~1.46e-5 (which carries a RMSNorm square-reduce feedback this op lacks). The device
rel-L2 combines the fp32 floor and the bf16 error in QUADRATURE
(sqrt(floor^2 + bf16^2)); with a sub-1e-6 floor the bf16 term dominates outright, so
the predicted on-device rel-L2 ~= the offline bf16 number itself (~4.5e-6, the
pure-GEMM family value), ~4.5x under the 2e-5 gate.

NOTE on K: the dropped lo@lo term is ~2^-16 relative PER product; summed over K
terms with random sign it grows like sqrt(K), but so does the full sum, so the
rel-L2 is ~K-independent. matmul's K=5120 (vs transpose_matmul's 2048) should give
essentially the same ~4.5e-6 -- this sim measures it to be sure.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

# matmul (NKIBench case): lhs (M,K), rhs (K,N), out (M,N) = lhs @ rhs.
M, N, K = 4096, 12288, 5120
INPUT_SEED = 42          # adapter/nkibench_case.py DEFAULT_INPUT_SEED
REL_TOL = 2e-5           # adapter DEFAULT_REL_TOL (the NKIBench gate)
_REFERENCE_REL = os.path.join(
    "reference", "matmul_M4096_N12288_K5120_numpy_2.py")


def _nkibench_root() -> str:
    """Resolve the NKIBench checkout, honoring the repo-supported $NKIBENCH_ROOT.

    Mirrors verify.py: os.environ["NKIBENCH_ROOT"] if set, else the default sibling
    layout <repo>/../AccelOpt/NKIBench. The repo root is this file's ../../../..
    (workspaces/matmul/runs -> repo root).
    """
    override = os.environ.get("NKIBENCH_ROOT")
    if override:
        return override
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    return os.path.join(os.path.dirname(repo_root), "AccelOpt", "NKIBench")


# The actual NKIBench reference module (never edited). The independent draw control
# loads THIS module's own get_inputs()/forward() to validate our draw model, instead
# of comparing the reference against itself (which would be tautologically 0.0).
_REFERENCE_PATH = os.path.join(_nkibench_root(), _REFERENCE_REL)
# matmul_v2_b4's on-device fp32 floor (candidates.jsonl layout_check_worst_relL2),
# used only to predict the device quadrature; the bf16 term dominates it here.
FP32_FLOOR = 4.207e-7


def to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """Round fp32 -> bfloat16 (round-to-nearest-even), returned as fp32 values.

    bf16 keeps the fp32 sign+8-bit exponent and truncates the 23-bit mantissa to 7
    explicit bits, i.e. it drops the low 16 bits of the fp32 bit pattern. RNE adds a
    tie-to-even rounding bias before truncating:
        lsb          = bit 16 (the least significant bit that survives)
        rounding_bias = 0x7FFF + lsb
        bf16_bits    = (uint32(x) + rounding_bias) >> 16
    Mantissa carry into the exponent is handled correctly by the integer add. Inputs
    here are O(1) normal values (no Inf/NaN), so the NaN/Inf edge cases do not arise.
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
    """Draw (lhs, rhs) exactly as the reference's seeded natural get_inputs() does.

    reference get_inputs(): lhs ~ normal(0,1,(M,K)), rhs ~ normal(0,1,(K,N)), both
    fp32, in that order, after a single np.random.seed(seed).
    """
    np.random.seed(seed)
    lhs = np.random.normal(loc=0.0, scale=1.0, size=(M, K)).astype(np.float32)
    rhs = np.random.normal(loc=0.0, scale=1.0, size=(K, N)).astype(np.float32)
    return lhs, rhs


def reference_forward(lhs, rhs) -> np.ndarray:
    """The NKIBench numpy reference for matmul: lhs @ rhs (fp32)."""
    return np.matmul(lhs, rhs).astype(np.float32)


def _load_reference_module():
    """Import the actual NKIBench reference module (read-only) for an independent draw.

    FAILS CLOSED: raises FileNotFoundError if the reference file is not reachable,
    rather than returning None. A missing reference must NOT let the offline gate
    emit an "authorizes" verdict on the unvalidated hard-coded draw model -- the
    whole point of the independent control is to validate against the real reference.
    Set $NKIBENCH_ROOT to point at the checkout if it is not at the default sibling
    path (mirrors verify.py's resolution).
    """
    path = os.path.normpath(_REFERENCE_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"NKIBench reference not found at {path}. The offline gate's independent "
            f"control cannot validate the draw model, so it will NOT authorize the "
            f"split. Set NKIBENCH_ROOT to the NKIBench checkout (verify.py resolves it "
            f"the same way) and re-run.")
    spec = importlib.util.spec_from_file_location("_nkibench_matmul_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def independent_reference_draw(seed: int):
    """Draw + forward using the REFERENCE module's OWN get_inputs()/forward().

    This is the independent control: it does NOT reuse this script's draw_inputs()
    or reference_forward(). The adapter seeds np.random.seed(DEFAULT_INPUT_SEED)
    before calling the reference's get_inputs(), so we mirror that exactly. Returns
    (lhs, rhs, ref_out) drawn/computed entirely by the reference's own code. Raises
    (fails closed) via _load_reference_module() if the reference cannot be loaded.
    """
    mod = _load_reference_module()
    np.random.seed(seed)                     # mirror adapter's pre-draw seeding
    lhs, rhs = mod.get_inputs()              # reference's OWN draw (order/dist/dtype)
    ref_out = np.asarray(mod.forward(lhs, rhs)).astype(np.float32)  # reference's OWN forward
    return lhs, rhs, ref_out


def bf16x2_3prod(lhs, rhs) -> np.ndarray:
    """Idealized bf16x2 3-product split matmul (the kernel's exact accumulation).

    Splits both operands, then accumulates hi@hi + hi@lo + lo@hi in fp32, dropping
    the negligible lo@lo cross term.
    """
    lhs_hi, lhs_lo = split_bf16x2(lhs)
    rhs_hi, rhs_lo = split_bf16x2(rhs)
    prod = (np.matmul(lhs_hi, rhs_hi) + np.matmul(lhs_hi, rhs_lo)
            + np.matmul(lhs_lo, rhs_hi)).astype(np.float32)
    return prod


def bf16x2_4prod(lhs, rhs) -> np.ndarray:
    """Reference-only: full 4-product split (keeps lo@lo) to size the dropped term."""
    lhs_hi, lhs_lo = split_bf16x2(lhs)
    rhs_hi, rhs_lo = split_bf16x2(rhs)
    prod = (np.matmul(lhs_hi, rhs_hi) + np.matmul(lhs_hi, rhs_lo)
            + np.matmul(lhs_lo, rhs_hi) + np.matmul(lhs_lo, rhs_lo)).astype(np.float32)
    return prod


def plain_bf16(lhs, rhs) -> np.ndarray:
    """Reference-only: plain single-limb bf16 matmul (the rejected route) for scale."""
    lhs_hi = to_bf16_rne(lhs)
    rhs_hi = to_bf16_rne(rhs)
    return np.matmul(lhs_hi, rhs_hi).astype(np.float32)


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

    # INDEPENDENT draw control: load the actual NKIBench reference module and draw +
    # forward with ITS OWN code (seeded like the adapter), then compare that against
    # this script's draw_inputs()/reference_forward(). This is NOT tautological --
    # if our seed / distribution / draw order / dtype / formula is wrong, the two
    # independently-produced references DIVERGE (non-zero), and we fail loudly. A
    # self-comparison (ref vs ref) would be 0.0 even with a wrong input model.
    # FAILS CLOSED: independent_reference_draw() raises if the reference is
    # unreachable, so the gate can never authorize on an unvalidated draw model.
    _lhs_ref, _rhs_ref, ref_indep = independent_reference_draw(INPUT_SEED)
    input_match_rel = rel_l2(lhs, _lhs_ref)              # our draw vs reference's draw
    input_match_rel = max(input_match_rel, rel_l2(rhs, _rhs_ref))
    ctrl_rel = rel_l2(ref, ref_indep)                    # our forward vs reference's forward
    # Explicit raise (NOT assert): assert is stripped under `python -O` /
    # PYTHONOPTIMIZE, which would silently disable the only check that the offline
    # numbers were computed on the SAME input the NKIBench gate scores. Fail closed
    # unconditionally.
    if not (ctrl_rel < 1e-6 and input_match_rel < 1e-6):
        raise RuntimeError(
            f"draw model MISMATCH vs reference module: inputs rel-L2={input_match_rel:.3e}, "
            f"outputs rel-L2={ctrl_rel:.3e} -- offline gate would score the WRONG input")
    ctrl_note = ("INDEPENDENT vs reference module get_inputs()/forward(); "
                 f"inputs match {input_match_rel:.1e}; expect ~0 iff draw model correct")

    plain_rel = rel_l2(plain_bf16(lhs, rhs), ref)
    p3_rel = rel_l2(bf16x2_3prod(lhs, rhs), ref)
    p4_rel = rel_l2(bf16x2_4prod(lhs, rhs), ref)

    print(f"[seed {INPUT_SEED}] fp32 CONTROL vs reference   rel-L2 = {ctrl_rel:.3e}"
          f"   ({ctrl_note})")
    print(f"[seed {INPUT_SEED}] plain bf16 (rejected)       rel-L2 = {plain_rel:.3e}"
          f"   (scale check; expect ~1e-3)")
    print(f"[seed {INPUT_SEED}] bf16x2 3-product            rel-L2 = {p3_rel:.3e}")
    print(f"[seed {INPUT_SEED}] bf16x2 4-product (keeps lo@lo) rel-L2 = {p4_rel:.3e}"
          f"   (sizes the dropped lo@lo term)\n")

    # The remote gate scores ONLY the seed-42 draw (the adapter pins seed 42 for
    # every profiler seed). These extra draws are an INPUT-DIVERSITY robustness
    # check on inputs the remote gate does NOT run -- reported separately below,
    # never folded into the gate-reproducing seed-42 datum.
    diversity_seeds = [0, 21, 63, 84]
    p3_diversity = []
    print("[input-diversity check] distinct draws NOT scored by the remote gate "
          "(the adapter pins seed 42 for all profiler seeds):")
    for s in diversity_seeds:
        ls, rs_ = draw_inputs(s)
        rf = reference_forward(ls, rs_)
        p3 = rel_l2(bf16x2_3prod(ls, rs_), rf)
        p3_diversity.append(p3)
        print(f"    [seed {s:4d}] bf16x2 3-product = {p3:.3e}")

    # gate datum = seed 42 (the only input the remote profiler actually scores);
    # diversity worst = the largest across the distinct-input robustness draws.
    gate_p3 = p3_rel
    worst_diversity_p3 = max(p3_diversity)
    worst_overall_p3 = max(gate_p3, worst_diversity_p3)

    print("\n" + "=" * 66)
    print(f"fp32 control rel-L2 (INDEPENDENT ref module; ~0 iff draw model correct): {ctrl_rel:.3e}")
    print(f"GATE datum: seed-42 bf16x2 3-product rel-L2:           {gate_p3:.3e}"
          f"   (the ONLY input the remote profiler scores)")
    print(f"input-diversity worst (seeds 0/21/63/84, NOT gated):   {worst_diversity_p3:.3e}")
    print(f"on-device fp32 b4 floor (reference datum):             {FP32_FLOOR:.2e}")
    print(f"NKIBench gate:                                         {REL_TOL:.1e}")
    # Predict the device rel-L2 from the seed-42 gate datum (what the gate scores);
    # the diversity draws only confirm the bf16 term is input-robust, not tuned.
    quad = float(np.sqrt(FP32_FLOOR ** 2 + gate_p3 ** 2))
    print(f"predicted device quadrature sqrt(floor^2 + bf16^2):    {quad:.3e}")
    verdict = "COMFORTABLY BELOW (<1.3e-5) -- authorizes the split kernel" if worst_overall_p3 < 1.3e-5 else (
        "MARGINAL/ABOVE (>=1.3e-5) -- does NOT authorize; record precision-floor datum")
    print(f"no-spend decision input: gate(seed42)={gate_p3:.3e}, "
          f"diversity worst={worst_diversity_p3:.3e}  ->  {verdict}")
    print(f"                         predicted device quad={quad:.3e} vs gate {REL_TOL:.1e}")
    print("=" * 66)


if __name__ == "__main__":
    main()
