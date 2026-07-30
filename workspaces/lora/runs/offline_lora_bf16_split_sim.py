#!/usr/bin/env python3
"""Offline numpy pre-check for a base-GEMM-only compensated bf16x2 split of lora.

Zero remote spend. Reproduces the EXACT input the remote gate scores, computes the
fp32 COMPOSITE reference the NKIBench way, then models an idealized bf16x2 compensated
split applied to ONLY the base GEMM x@w (keeping the low-rank path fp32) and reports the
composite NKIBench-style relative-L2 against the fp32 reference.

IMPORTANT — what the remote gate actually scores: the NKIBench adapter
(adapter/nkibench_case.py) hard-codes np.random.seed(DEFAULT_INPUT_SEED=42) before every
natural draw, so ALL FIVE on-device profiler correctness seeds [0,21,42,63,84] score the
SAME seed-42 input tensors. The single gate-reproducing datum is therefore the seed-42
composite draw. The extra draws [0,21,63,84] simulated below are an INPUT-DIVERSITY
robustness check on distinct inputs the remote gate does NOT run (per the compensated
bf16x2 lesson, the offline sim's distinct draws are the real input-diversity correctness
margin, since the on-device 5-seed PASS reuses one input); they are reported separately
and never presented as the remote gate.

lora is a COMPOSITE op:  out = x@w + (x@a)@b  (M=4096, K=5120, N=12288, R=128).
Unlike the pure-GEMM siblings (matmul / transpose_matmul), the output is the SUM of a
base GEMM and a low-rank term. The KEY lora-specific fact is that the low-rank term
DOMINATES the output magnitude: x@a has variance ~K, then @b sums over R=128, so the
low-rank output has variance ~R*K and swamps the base's ~K by ~sqrt(R) = 11x. Measured
below: the base is ~8.8% of the composite L2, the low-rank ~99.6% (dilution ~11.4x). So
splitting ONLY the base to bf16x2 puts the split's rounding error into the small 8.8%
component; in the composite rel-L2 that base-only error is DILUTED ~11x. This is why the
composite rel-L2 (route [A], ~3.9e-7) is far below the base-only split error in isolation
(route [B], ~4.45e-6, the pure-GEMM family value the matmul sibling measures directly).

The base split keeps each fp32 operand as two bf16 limbs and accumulates three bf16
products in fp32, dropping the negligible lo@lo cross term:
    x_hi = bf16(x),  x_lo = bf16(x - x_hi)   (round-to-nearest-even)
    w_hi = bf16(w),  w_lo = bf16(w - w_hi)
    x@w ~= x_hi@w_hi + x_hi@w_lo + x_lo@w_hi     (drop x_lo@w_lo)
The kernel transposes x (k onto the partition axis) before nc_matmul, but the bf16 split
is element-wise, so splitting the raw operand (as here) is identical to splitting the
transposed tiles the kernel builds. The down-projection x@a and the up-projection
(x@a)@b stay fp32.

Idealized numpy RNE limb construction + exact fp32 accumulation is at least as accurate
as the hardware, so an offline result at/above the gate means the device almost-certainly
fails. It is a practical no-spend gate, not an impossibility proof — a remote full-5-seed
PASS is still required to promote.

DEVICE PREDICTION (quadrature): the on-device rel-L2 is NOT the offline composite bf16
number — it is that bf16 error added IN QUADRATURE with the pre-existing on-device fp32
floor (the bf16-native PE emulates "fp32" with multi-pass rounding the idealized numpy
sim cannot model):  rel_L2_ondevice ~= sqrt(fp32_floor^2 + composite_bf16^2). Here the
lora fp32 floor is 4.874e-7 (lora_v1 / lora_v2_mblk4 measured, byte-identical across the
fp32 kernels) and the composite bf16 term is ~3.9e-7 — so the FP32 FLOOR DOMINATES and
the predicted device rel-L2 is ~sqrt(4.874^2 + 3.9^2)e-7 ~= 6.2e-7, still ~32x under the
gate. This INVERTS the sibling matmul case (tiny sub-1e-6 floor, bf16 term dominated);
here the offline 3.9e-7 is an idealized-composite datum, NOT a promise the device beats
its own fp32 floor — the device lands slightly ABOVE the offline number, at the quadrature.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

# lora (NKIBench case): x (M,K), w (K,N), a (K,R), b (R,N); out = x@w + (x@a)@b.
M, N, K, R = 4096, 12288, 5120, 128
INPUT_SEED = 42          # adapter/nkibench_case.py DEFAULT_INPUT_SEED
REL_TOL = 2e-5           # adapter DEFAULT_REL_TOL (the NKIBench gate)
AUTHORIZE_BELOW = 1.3e-5  # no-spend gate: composite must be comfortably below this
# Fold model-consistency band (weight-fold route only): the model predicts the folded route
# lands at the UNDILUTED pure-GEMM value ~4.45e-6. A folded worst in [FOLD_MODEL_CONSISTENCY,
# AUTHORIZE_BELOW) is UNDER the absolute cap yet ABOVE what the model predicts -> the numeric
# model is wrong -> HALT and investigate, do NOT spend even though it is under the cap.
FOLD_MODEL_CONSISTENCY = 8e-6
_REFERENCE_REL = os.path.join(
    "reference", "lora_M4096_N12288_K5120_R128_numpy_1.py")

# lora's on-device fp32 floor (lora_v1 / lora_v2_mblk4 candidates.jsonl worst_rel_l2),
# used only to predict the device quadrature. Here the floor DOMINATES the diluted
# composite bf16 term, so the device rel-L2 sits at ~sqrt(floor^2 + bf16^2), just above
# the offline composite number.
FP32_FLOOR = 4.874185370915276e-07


def _nkibench_root() -> str:
    """Resolve the NKIBench checkout, honoring the repo-supported $NKIBENCH_ROOT.

    Mirrors verify.py: os.environ["NKIBENCH_ROOT"] if set, else the default sibling
    layout <repo>/../AccelOpt/NKIBench. The repo root is this file's ../../../..
    (workspaces/lora/runs -> repo root).
    """
    override = os.environ.get("NKIBENCH_ROOT")
    if override:
        return override
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    return os.path.join(os.path.dirname(repo_root), "AccelOpt", "NKIBench")


# The actual NKIBench reference module (never edited). The independent draw control loads
# THIS module's own get_inputs()/forward() to validate our draw model, instead of
# comparing the reference against itself (which would be tautologically 0.0).
_REFERENCE_PATH = os.path.join(_nkibench_root(), _REFERENCE_REL)


def to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """Round fp32 -> bfloat16 (round-to-nearest-even), returned as fp32 values.

    bf16 keeps the fp32 sign+8-bit exponent and truncates the 23-bit mantissa to 7
    explicit bits, i.e. it drops the low 16 bits of the fp32 bit pattern. RNE adds a
    tie-to-even rounding bias before truncating:
        lsb          = bit 16 (the least significant bit that survives)
        rounding_bias = 0x7FFF + lsb
        bf16_bits    = (uint32(x) + rounding_bias) >> 16
    Mantissa carry into the exponent is handled correctly by the integer add. Inputs here
    are O(1) normal values (no Inf/NaN), so the NaN/Inf edge cases do not arise.
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


def mm_bf16x2_3prod(lhs, rhs) -> np.ndarray:
    """Idealized bf16x2 3-product split matmul (the kernel's exact accumulation).

    Splits both operands, accumulates hi@hi + hi@lo + lo@hi in fp32, drops lo@lo.
    """
    lhs_hi, lhs_lo = split_bf16x2(lhs)
    rhs_hi, rhs_lo = split_bf16x2(rhs)
    return (np.matmul(lhs_hi, rhs_hi) + np.matmul(lhs_hi, rhs_lo)
            + np.matmul(lhs_lo, rhs_hi)).astype(np.float32)


def mm_bf16x2_4prod(lhs, rhs) -> np.ndarray:
    """Reference-only: full 4-product split (keeps lo@lo) to size the dropped term."""
    lhs_hi, lhs_lo = split_bf16x2(lhs)
    rhs_hi, rhs_lo = split_bf16x2(rhs)
    return (np.matmul(lhs_hi, rhs_hi) + np.matmul(lhs_hi, rhs_lo)
            + np.matmul(lhs_lo, rhs_hi) + np.matmul(lhs_lo, rhs_lo)).astype(np.float32)


def mm_plain_bf16(lhs, rhs) -> np.ndarray:
    """Reference-only: plain single-limb bf16 matmul (the rejected route) for scale."""
    return np.matmul(to_bf16_rne(lhs), to_bf16_rne(rhs)).astype(np.float32)


def draw_inputs(seed: int):
    """Draw (x, w, a, b) exactly as the reference's seeded natural get_inputs() does.

    reference get_inputs(): x ~ normal(0,1,(M,K)), w ~ normal(0,1,(K,N)),
    a ~ normal(0,1,(K,R)), b ~ normal(0,1,(R,N)), all fp32, in that order, after a
    single np.random.seed(seed).
    """
    np.random.seed(seed)
    x = np.random.normal(loc=0.0, scale=1.0, size=(M, K)).astype(np.float32)
    w = np.random.normal(loc=0.0, scale=1.0, size=(K, N)).astype(np.float32)
    a = np.random.normal(loc=0.0, scale=1.0, size=(K, R)).astype(np.float32)
    b = np.random.normal(loc=0.0, scale=1.0, size=(R, N)).astype(np.float32)
    return x, w, a, b


def reference_forward(x, w, a, b) -> np.ndarray:
    """The NKIBench numpy reference for lora: x@w + (x@a)@b (fp32)."""
    y1 = np.matmul(x, w)
    y2 = np.matmul(np.matmul(x, a), b)
    return (y1 + y2).astype(np.float32)


# --- candidate composite pipelines: which parts are bf16x2 vs stay fp32 ---

def composite_base_bf16x2(x, w, a, b) -> np.ndarray:
    """[A] THE PLAN: base x@w bf16x2 3-product; low-rank (x@a)@b stays fp32.

    This is exactly what lora_v3_bf16_split.py computes on device: only the base GEMM is
    split; the down-projection x@a and the up-projection (x@a)@b are fp32.
    """
    base = mm_bf16x2_3prod(x, w)
    low_rank = np.matmul(np.matmul(x, a), b).astype(np.float32)
    return (base + low_rank).astype(np.float32)


def composite_base_and_up_bf16x2(x, w, a, b) -> np.ndarray:
    """[B'] out-of-scope reference: split base AND the up-projection (x@a)@b, down fp32.

    No PE upside (low-rank is only 3.4% of MACs) and needless risk — reported only to
    show the composite stays well under the gate even then.
    """
    base = mm_bf16x2_3prod(x, w)
    xa = np.matmul(x, a).astype(np.float32)          # down-projection fp32
    up = mm_bf16x2_3prod(xa, b)                       # up-projection bf16x2
    return (base + up).astype(np.float32)


def composite_plain_bf16(x, w, a, b) -> np.ndarray:
    """[C] rejected scale check: plain single-limb bf16 on ALL matmuls."""
    base = mm_plain_bf16(x, w)
    xa = mm_plain_bf16(x, a)
    up = mm_plain_bf16(xa, b)
    return (base + up).astype(np.float32)


def w_prime_fp32(w, a, b) -> np.ndarray:
    """The weight-fold: w' = w + a@b in fp32 (the intentional HBM materialization).

    The kernel materializes w' = w + a@b once as an fp32 (K,N) tensor, then runs a pure
    x@w' base GEMM. This mirrors the on-device prologue exactly: a@b is accumulated in fp32
    and added to the fp32 w, and w' STAYS fp32 (it is split into bf16 limbs only at the
    main-GEMM consumption point, never stored as bf16).
    """
    ab = np.matmul(a, b).astype(np.float32)      # (K,R)@(R,N) = (K,N), the low-rank weight
    return (w + ab).astype(np.float32)


def composite_fold_bf16x2(x, w, a, b) -> np.ndarray:
    """[F] THE FOLD (lora_v4_fold): out = x@w' via ONE bf16x2 3-product GEMM, w' = fp32(w+a@b).

    Unlike the base-only split [A] (which keeps the low-rank term fp32 and dilutes the base
    split error ~11.4x), the fold routes the ENTIRE output -- including the 99.6%-dominant
    low-rank part -- through the single bf16x2 GEMM x@w'. So its rel-L2 is the UNDILUTED
    pure-GEMM family value ~4.45e-6 (route [B]), NOT the diluted ~3.93e-7 of [A]. w' is fp32
    (built by w_prime_fp32), split into bf16 limbs inside mm_bf16x2_3prod exactly as the
    kernel splits it at the consumption point.
    """
    wp = w_prime_fp32(w, a, b)
    return mm_bf16x2_3prod(x, wp).astype(np.float32)


def fold_fp32_control(x, w, a, b) -> np.ndarray:
    """[F-fp32] fp32 reassociation control: x@(w + a@b) computed entirely in fp32.

    Scored against the fp32 reference x@w + (x@a)@b, this isolates the fp32 REASSOCIATION
    error of the fold (folding a@b into the weights then multiplying, vs the reference's
    separate base + low-rank matmuls) from the bf16 rounding. Expected ~fp32-floor level
    (~1e-7): if fold_fp32_control << composite_fold_bf16x2, the folded-route error is
    bf16-dominated (matches the pure-GEMM sibling), not a reassociation artifact.
    """
    wp = w_prime_fp32(w, a, b)
    return np.matmul(x, wp).astype(np.float32)


def rel_l2(v_k: np.ndarray, v_r: np.ndarray) -> float:
    """NKIBench relative-L2 over the flattened output: ||v_k - v_r||_2 / ||v_r||_2."""
    num = np.linalg.norm((v_k - v_r).ravel().astype(np.float64))
    den = np.linalg.norm(v_r.ravel().astype(np.float64))
    return float(num / den)


def _load_reference_module():
    """Import the actual NKIBench reference module (read-only) for an independent draw.

    FAILS CLOSED: raises FileNotFoundError if the reference file is not reachable, rather
    than returning None. A missing reference must NOT let the offline gate emit an
    "authorizes" verdict on the unvalidated hard-coded draw model — the whole point of the
    independent control is to validate against the real reference. Set $NKIBENCH_ROOT to
    point at the checkout if it is not at the default sibling path (mirrors verify.py).
    """
    path = os.path.normpath(_REFERENCE_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"NKIBench reference not found at {path}. The offline gate's independent "
            f"control cannot validate the draw model, so it will NOT authorize the "
            f"split. Set NKIBENCH_ROOT to the NKIBench checkout (verify.py resolves it "
            f"the same way) and re-run.")
    spec = importlib.util.spec_from_file_location("_nkibench_lora_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def independent_reference_draw(seed: int):
    """Draw + forward using the REFERENCE module's OWN get_inputs()/forward().

    This is the independent control: it does NOT reuse this script's draw_inputs() or
    reference_forward(). The adapter seeds np.random.seed(DEFAULT_INPUT_SEED) before
    calling the reference's get_inputs(), so we mirror that exactly. Returns
    (x, w, a, b, ref_out) drawn/computed entirely by the reference's own code. Raises
    (fails closed) via _load_reference_module() if the reference cannot be loaded.
    """
    mod = _load_reference_module()
    np.random.seed(seed)                     # mirror adapter's pre-draw seeding
    x, w, a, b = mod.get_inputs()            # reference's OWN draw (order/dist/dtype)
    ref_out = np.asarray(mod.forward(x, w, a, b)).astype(np.float32)  # reference's OWN forward
    return x, w, a, b, ref_out


def main():
    print(f"[config] M={M} N={N} K={K} R={R}  input_seed={INPUT_SEED}  "
          f"gate rel_tol={REL_TOL:.1e}  authorize_below={AUTHORIZE_BELOW:.1e}  "
          f"fp32_floor={FP32_FLOOR:.3e}\n")

    x, w, a, b = draw_inputs(INPUT_SEED)
    ref = reference_forward(x, w, a, b)

    # INDEPENDENT draw control: load the actual NKIBench reference module and draw +
    # forward with ITS OWN code (seeded like the adapter), then compare against this
    # script's draw_inputs()/reference_forward(). NOT tautological — if our seed /
    # distribution / draw order / dtype / formula is wrong, the two independently-produced
    # references DIVERGE (non-zero) and we fail loudly. A self-comparison (ref vs ref)
    # would be 0.0 even with a wrong input model. FAILS CLOSED: independent_reference_draw
    # raises if the reference is unreachable, so the gate can never authorize on an
    # unvalidated draw model.
    _x, _w, _a, _b, ref_indep = independent_reference_draw(INPUT_SEED)
    input_match_rel = max(rel_l2(x, _x), rel_l2(w, _w),
                          rel_l2(a, _a), rel_l2(b, _b))   # our draws vs reference's draws
    ctrl_rel = rel_l2(ref, ref_indep)                     # our forward vs reference's forward
    # Explicit raise (NOT assert): assert is stripped under `python -O` / PYTHONOPTIMIZE,
    # which would silently disable the only check that the offline numbers were computed
    # on the SAME input the NKIBench gate scores. Fail closed unconditionally.
    if not (ctrl_rel < 1e-6 and input_match_rel < 1e-6):
        raise RuntimeError(
            f"draw model MISMATCH vs reference module: inputs rel-L2={input_match_rel:.3e}, "
            f"outputs rel-L2={ctrl_rel:.3e} -- offline gate would score the WRONG input")
    ctrl_note = ("INDEPENDENT vs reference module get_inputs()/forward(); "
                 f"inputs match {input_match_rel:.1e}; expect ~0 iff draw model correct")

    # Composite magnitude decomposition: WHY the base-only split is diluted ~11x. The
    # low-rank term dominates the output L2, so a base-only rounding error is a small
    # fraction of the composite norm.
    base_fp32 = np.matmul(x, w).astype(np.float32)
    low_rank_fp32 = np.matmul(np.matmul(x, a), b).astype(np.float32)
    base_n = np.linalg.norm(base_fp32.ravel().astype(np.float64))
    lr_n = np.linalg.norm(low_rank_fp32.ravel().astype(np.float64))
    comp_n = np.linalg.norm(ref.ravel().astype(np.float64))
    dilution = comp_n / base_n     # composite / base ~ 11x

    # Route [B]: the base-only split error IN ISOLATION (against the fp32 base), i.e. the
    # pure-GEMM family value the matmul sibling measures directly. This is what gets
    # diluted ~dilution-fold in the composite.
    base_only_rel = rel_l2(mm_bf16x2_3prod(x, w), base_fp32)

    a_rel = rel_l2(composite_base_bf16x2(x, w, a, b), ref)          # [A] the plan
    b_rel = rel_l2(composite_base_and_up_bf16x2(x, w, a, b), ref)   # [B'] base+up (oos)
    c_rel = rel_l2(composite_plain_bf16(x, w, a, b), ref)          # [C] plain bf16 reject
    a4_rel = rel_l2((mm_bf16x2_4prod(x, w) + low_rank_fp32).astype(np.float32), ref)

    print(f"[seed {INPUT_SEED}] fp32 CONTROL vs reference       rel-L2 = {ctrl_rel:.3e}"
          f"   ({ctrl_note})")
    print(f"[seed {INPUT_SEED}] composite ||out||={comp_n:.4e}: base is {base_n/comp_n*100:.2f}%"
          f" of L2, low-rank is {lr_n/comp_n*100:.2f}%  (dilution {dilution:.2f}x)")
    print(f"[seed {INPUT_SEED}] [B] base-only split IN ISOLATION rel-L2 = {base_only_rel:.3e}"
          f"   (pure-GEMM family value; diluted {dilution:.1f}x in the composite)")
    print(f"[seed {INPUT_SEED}] [A] composite base-bf16x2, LR fp32 rel-L2 = {a_rel:.3e}"
          f"   (THE PLAN; ~ base_only/{dilution:.1f})")
    print(f"[seed {INPUT_SEED}] [A] 4-product base (keeps lo@lo)  rel-L2 = {a4_rel:.3e}"
          f"   (sizes the dropped lo@lo term)")
    print(f"[seed {INPUT_SEED}] [B'] base+up-proj bf16x2 (oos)    rel-L2 = {b_rel:.3e}"
          f"   (still << gate, but no PE upside)")
    print(f"[seed {INPUT_SEED}] [C] plain bf16 everywhere (reject) rel-L2 = {c_rel:.3e}"
          f"   (scale check; expect ~2.3e-3)\n")

    # The remote gate scores ONLY the seed-42 draw (the adapter pins seed 42 for every
    # profiler seed). These extra draws are an INPUT-DIVERSITY robustness check on inputs
    # the remote gate does NOT run — reported separately, never folded into the gate datum.
    diversity_seeds = [0, 21, 63, 84]
    # Draw each distinct-input seed and its fp32 reference ONCE; the base-only [A] and the
    # weight-fold [F] diversity checks below both score against these same draws (the fold
    # loop reuses them rather than re-drawing + re-running the full composite reference).
    diversity_draws = [(s, *draw_inputs(s)) for s in diversity_seeds]
    diversity_refs = {s: reference_forward(xs, ws, as_, bs)
                      for s, xs, ws, as_, bs in diversity_draws}
    a_diversity = []
    print("[input-diversity check] distinct draws NOT scored by the remote gate "
          "(the adapter pins seed 42 for all profiler seeds):")
    for s, xs, ws, as_, bs in diversity_draws:
        ad = rel_l2(composite_base_bf16x2(xs, ws, as_, bs), diversity_refs[s])
        a_diversity.append(ad)
        print(f"    [seed {s:4d}] [A] composite base-bf16x2 = {ad:.3e}")

    # gate datum = seed 42 (the only input the remote profiler actually scores);
    # diversity worst = the largest across the distinct-input robustness draws.
    gate_a = a_rel
    worst_diversity_a = max(a_diversity)
    worst_overall_a = max(gate_a, worst_diversity_a)

    print("\n" + "=" * 70)
    print(f"fp32 control rel-L2 (INDEPENDENT ref module; ~0 iff draw model correct): {ctrl_rel:.3e}")
    print(f"GATE datum: seed-42 composite base-bf16x2 rel-L2:      {gate_a:.3e}"
          f"   (the ONLY input the remote profiler scores)")
    print(f"input-diversity worst (seeds 0/21/63/84, NOT gated):   {worst_diversity_a:.3e}")
    print(f"base-only split in isolation (route [B], NOT gated):   {base_only_rel:.3e}"
          f"   (pure-GEMM value; composite dilutes it {dilution:.1f}x)")
    print(f"on-device fp32 floor (lora_v1 / lora_v2_mblk4 datum):  {FP32_FLOOR:.3e}")
    print(f"NKIBench gate:                                         {REL_TOL:.1e}")
    # Predict the device rel-L2 from the seed-42 gate datum. UNLIKE the pure-GEMM
    # siblings, here the fp32 floor DOMINATES the diluted composite bf16 term, so the
    # device lands slightly ABOVE the offline number, at the quadrature.
    quad = float(np.sqrt(FP32_FLOOR ** 2 + gate_a ** 2))
    print(f"predicted device quadrature sqrt(floor^2 + composite_bf16^2): {quad:.3e}"
          f"   (fp32 floor DOMINATES; device ~ this, not the offline {gate_a:.2e})")
    authorized = worst_overall_a < AUTHORIZE_BELOW
    verdict = (f"COMFORTABLY BELOW (<{AUTHORIZE_BELOW:.1e}) -- authorizes the base-only "
               "split kernel (lora_v3_bf16_split)") if authorized else (
        f"MARGINAL/ABOVE (>={AUTHORIZE_BELOW:.1e}) -- does NOT authorize; record "
        "precision-floor datum instead of spending a remote run")
    print(f"no-spend decision input: gate(seed42)={gate_a:.3e}, "
          f"diversity worst={worst_diversity_a:.3e}  ->  {verdict}")
    print(f"                         predicted device quad={quad:.3e} vs gate {REL_TOL:.1e}")
    print("=" * 70)

    # ==================================================================================
    # WEIGHT-FOLD GATE (route [F], lora_v4_fold): out = x@w' via ONE bf16x2 GEMM,
    # w' = fp32(w + a@b). This is a SEPARATE, STRICTER gate from the base-only [A] gate
    # above, because the fold LOSES the ~11.4x dilution: the whole output (incl. the
    # 99.6%-dominant low-rank term) flows through one bf16x2 GEMM, so the fold rel-L2 is
    # the UNDILUTED pure-GEMM value ~4.45e-6 (NOT the diluted ~3.9e-7 of [A]).
    # ==================================================================================
    print("\n" + "#" * 70)
    print("WEIGHT-FOLD GATE (route [F]: out = x@w' bf16x2, w' = fp32(w + a@b))")
    print("#" * 70)

    # Seed-42 gate datum: the folded bf16x2 route + its fp32 reassociation control.
    # Materialize w' = w + a@b ONCE and score both routes against it (the bf16x2 fold and
    # the all-fp32 reassociation control both consume the same w'), rather than rebuilding
    # the (K,N) fold inside each route helper.
    wp = w_prime_fp32(w, a, b)
    fold_gate = rel_l2(mm_bf16x2_3prod(x, wp).astype(np.float32), ref)  # [F]
    fold_fp32 = rel_l2(np.matmul(x, wp).astype(np.float32), ref)        # [F-fp32] reassoc.
    print(f"[seed {INPUT_SEED}] [F] composite FOLD x@w' bf16x2     rel-L2 = {fold_gate:.3e}"
          f"   (UNDILUTED pure-GEMM value ~ base_only {base_only_rel:.2e})")
    print(f"[seed {INPUT_SEED}] [F-fp32] fp32 reassociation control rel-L2 = {fold_fp32:.3e}"
          f"   (x@(w+a@b) all-fp32 vs ref; expect ~fp32-floor ~1e-7 -> fold err is bf16-dominated)")

    # Fold input-diversity draws (NOT gated by the remote profiler; same caveat as [A]).
    # Reuse the same distinct-input draws + fp32 references the base-only check drew above.
    fold_diversity = []
    print("[input-diversity check] folded route on distinct draws (NOT scored by the gate):")
    for s, xs, ws, as_, bs in diversity_draws:
        fd = rel_l2(composite_fold_bf16x2(xs, ws, as_, bs), diversity_refs[s])
        fold_diversity.append(fd)
        print(f"    [seed {s:4d}] [F] composite fold bf16x2 = {fd:.3e}")

    fold_worst_diversity = max(fold_diversity)
    fold_worst_overall = max(fold_gate, fold_worst_diversity)
    fold_quad = float(np.sqrt(FP32_FLOOR ** 2 + fold_gate ** 2))

    print("\n" + "=" * 70)
    print(f"FOLD GATE datum: seed-42 fold bf16x2 rel-L2:           {fold_gate:.3e}"
          f"   (the ONLY input the remote profiler scores)")
    print(f"fold input-diversity worst (seeds 0/21/63/84, NOT gated): {fold_worst_diversity:.3e}")
    print(f"fold worst (seed42 gate ORed with diversity, conservative): {fold_worst_overall:.3e}")
    print(f"fp32 reassociation control (route [F-fp32]):          {fold_fp32:.3e}")
    print(f"on-device fp32 floor (lora_v1 / lora_v2_mblk4 datum): {FP32_FLOOR:.3e}")
    print(f"predicted device quadrature sqrt(floor^2 + fold^2):   {fold_quad:.3e}"
          f"   (bf16 DOMINATES here, unlike [A]; device ~ this)")
    print(f"model-consistency band: < {FOLD_MODEL_CONSISTENCY:.1e}   "
          f"absolute cap: < {AUTHORIZE_BELOW:.1e}   NKIBench gate: {REL_TOL:.1e}")

    # Two-tier fold authorization (absolute cap + model-consistency band):
    #   worst >= AUTHORIZE_BELOW (1.3e-5)          -> does NOT authorize (absolute fail-closed)
    #   worst in [FOLD_MODEL_CONSISTENCY, cap)     -> HALT: under the cap but the model
    #                                                 predicts ~4.45e-6, so it is WRONG; do
    #                                                 NOT spend even though it is under the cap
    #   worst < FOLD_MODEL_CONSISTENCY (8e-6)      -> AUTHORIZES the fold remote run
    if fold_worst_overall >= AUTHORIZE_BELOW:
        fold_verdict = (f"ABOVE absolute cap (>={AUTHORIZE_BELOW:.1e}) -- does NOT authorize "
                        "the fold; record numeric no-spend reject, finalize D2")
    elif fold_worst_overall >= FOLD_MODEL_CONSISTENCY:
        fold_verdict = (f"MODEL-INCONSISTENT ([{FOLD_MODEL_CONSISTENCY:.1e}, "
                        f"{AUTHORIZE_BELOW:.1e})) -- under the cap but ABOVE the ~4.45e-6 the "
                        "model predicts -> HALT and investigate, do NOT spend")
    else:
        fold_verdict = (f"COMFORTABLY BELOW (<{FOLD_MODEL_CONSISTENCY:.1e}) -- authorizes the "
                        "weight-fold remote run (lora_v4_fold)")
    print(f"fold no-spend decision: worst={fold_worst_overall:.3e}  ->  {fold_verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
