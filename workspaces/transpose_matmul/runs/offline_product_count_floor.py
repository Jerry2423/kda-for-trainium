#!/usr/bin/env python3
"""Offline product-count floor table for transpose_matmul (near-optimality evidence).

Zero remote spend. Reproduces the plan's section-3 table proving that 3 bf16 products
is the NUMERIC FLOOR for the 2e-5 relative-L2 gate: every scheme with fewer products
(or fp16 in place of bf16) fails the gate, so PE-active -- which scales linearly with
the product count -- cannot be lowered, and the 1.385x arithmetic ceiling is hard.

It reuses the validated helpers in offline_bf16_split_sim.py (to_bf16_rne, split,
draw_inputs, reference_forward, rel_l2) so the fp32 control is bit-exact (0.0) and the
bf16 3-product number matches that sim. It ADDS the two rows that sim omits:
  * 2-product bf16 (split ONE operand, drop the other's lo)  -- one operand left at
    7-bit mantissa; its rounding dominates.
  * fp16 variants (plain / 2-product / 3-product)            -- fp16's 10-bit mantissa
    does not rescue a 2-product scheme, and fp16's 5-bit exponent is a correctness
    risk for zero speed gain (fp16 limbs cost the same PE cycles as bf16).

Run (numpy required):  python3 runs/offline_product_count_floor.py
"""

from __future__ import annotations

import numpy as np

# Reuse the validated pure-GEMM sim helpers (same directory).
from offline_bf16_split_sim import (
    K, M, N, REL_TOL,
    to_bf16_rne, split_bf16x2, draw_inputs, reference_forward, rel_l2,
)


def to_fp16_rne(x: np.ndarray) -> np.ndarray:
    """fp32 -> fp16 (round-to-nearest-even) via numpy's IEEE cast, back to fp32."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def split_fp16x2(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    x_hi = to_fp16_rne(x)
    x_lo = to_fp16_rne((x - x_hi).astype(np.float32))
    return x_hi, x_lo


def gemm(a_t_operand, b_operand) -> np.ndarray:
    """lhs^T @ rhs on already-limbed K-major operands (a is lhs (K,M), b is rhs (K,N))."""
    return np.matmul(a_t_operand.T, b_operand).astype(np.float32)


def scheme_rel(lhs, rhs, ref, kind: str) -> float:
    """rel-L2 of a named limb scheme vs the fp32 reference."""
    if kind == "bf16_1prod":                       # plain bf16, 1 product
        return rel_l2(gemm(to_bf16_rne(lhs), to_bf16_rne(rhs)), ref)
    if kind == "bf16_2prod_lhs":                   # split lhs only (drop rhs_lo)
        lhi, llo = split_bf16x2(lhs)
        rhi = to_bf16_rne(rhs)
        return rel_l2((gemm(lhi, rhi) + gemm(llo, rhi)).astype(np.float32), ref)
    if kind == "bf16_2prod_rhs":                   # split rhs only (drop lhs_lo)
        lhi = to_bf16_rne(lhs)
        rhi, rlo = split_bf16x2(rhs)
        return rel_l2((gemm(lhi, rhi) + gemm(lhi, rlo)).astype(np.float32), ref)
    if kind == "bf16_3prod":                       # current kernel: hi@hi+hi@lo+lo@hi
        lhi, llo = split_bf16x2(lhs)
        rhi, rlo = split_bf16x2(rhs)
        return rel_l2((gemm(lhi, rhi) + gemm(lhi, rlo) + gemm(llo, rhi)).astype(np.float32), ref)
    if kind == "fp16_1prod":
        return rel_l2(gemm(to_fp16_rne(lhs), to_fp16_rne(rhs)), ref)
    if kind == "fp16_2prod_lhs":
        lhi, llo = split_fp16x2(lhs)
        rhi = to_fp16_rne(rhs)
        return rel_l2((gemm(lhi, rhi) + gemm(llo, rhi)).astype(np.float32), ref)
    if kind == "fp16_2prod_rhs":                   # split rhs only (drop lhs_lo)
        lhi = to_fp16_rne(lhs)
        rhi, rlo = split_fp16x2(rhs)
        return rel_l2((gemm(lhi, rhi) + gemm(lhi, rlo)).astype(np.float32), ref)
    if kind == "fp16_3prod":
        lhi, llo = split_fp16x2(lhs)
        rhi, rlo = split_fp16x2(rhs)
        return rel_l2((gemm(lhi, rhi) + gemm(lhi, rlo) + gemm(llo, rhi)).astype(np.float32), ref)
    raise ValueError(kind)


SCHEMES = [
    ("plain bf16",                 1, "bf16_1prod"),
    ("bf16 split lhs only",        2, "bf16_2prod_lhs"),
    ("bf16 split rhs only",        2, "bf16_2prod_rhs"),
    ("bf16 3-product (current)",   3, "bf16_3prod"),
    ("plain fp16",                 1, "fp16_1prod"),
    ("fp16 split lhs only",        2, "fp16_2prod_lhs"),
    ("fp16 split rhs only",        2, "fp16_2prod_rhs"),
    ("fp16 3-product",             3, "fp16_3prod"),
]


def main() -> int:
    seeds = [42, 0, 84]                    # the plan's stated offline seed set for §3
    print(f"[config] M={M} N={N} K={K}  gate rel_tol={REL_TOL:.1e}  seeds={seeds}")
    print(f"[config] PE-active scales LINEARLY with product count "
          f"(3 products => 3.502 ms bf16 floor; 2 => 2.335 ms)\n")

    # fp32 control on the primary seed must be bit-exact 0.0 (seed/formula match).
    lhs0, rhs0 = draw_inputs(seeds[0])
    ref0 = reference_forward(lhs0, rhs0)
    ctrl = rel_l2(gemm(lhs0.astype(np.float32), rhs0), ref0)
    print(f"[control] fp32 lhs^T@rhs vs reference rel-L2 = {ctrl:.3e}  "
          f"(expect 0.0 bit-exact)\n")

    print(f"{'scheme':<28}{'products':>9}{'worst rel-L2':>16}{'vs 2e-5 gate':>16}")
    print("-" * 69)
    worst = {}
    for name, prod, kind in SCHEMES:
        vals = []
        for s in seeds:
            ls, rs_ = draw_inputs(s)
            rf = reference_forward(ls, rs_)
            vals.append(scheme_rel(ls, rs_, rf, kind))
        w = max(vals)
        worst[kind] = w
        ratio = w / REL_TOL
        verdict = f"PASS ({REL_TOL/w:.1f}x under)" if w < REL_TOL else f"FAIL x{ratio:.0f}"
        print(f"{name:<28}{prod:>9}{w:>16.3e}{verdict:>16}")

    print("-" * 69)
    print("\nReading: every <3-product scheme leaves one operand at 7-bit (bf16) or")
    print("10-bit (fp16) mantissa; that un-refined operand's rounding dominates and")
    print("FAILS the 2e-5 gate by 10-120x. fp16 does NOT rescue a 2-product scheme")
    print("(still ~10x over) and its 5-bit exponent is a correctness risk for ZERO")
    print("speed gain (fp16 limbs cost the same PE cycles as bf16). => 3 bf16 products")
    print("is the numeric floor; PE-active cannot drop; 1.385x is a HARD ceiling.")
    # Floor is confirmed only if the 3-product bf16 scheme clears the gate AND EVERY
    # 2-product scheme (both bf16 splits and both fp16 splits) fails it -- derive the
    # check from the SCHEMES table so no computed 2-product variant is silently skipped.
    two_prod = [kind for _, prod, kind in SCHEMES if prod == 2]
    two_prod_worst = max(worst[k] for k in two_prod)
    all_two_prod_fail = all(worst[k] >= REL_TOL for k in two_prod)
    ok = worst["bf16_3prod"] < REL_TOL and all_two_prod_fail
    print(f"\nfloor confirmed: 3-product bf16 PASS ({worst['bf16_3prod']:.3e}) AND "
          f"ALL {len(two_prod)} 2-product schemes FAIL "
          f"(worst 2-product {two_prod_worst:.3e} >= gate {REL_TOL:.1e})  ->  {ok}")
    if not all_two_prod_fail:
        passers = [k for k in two_prod if worst[k] < REL_TOL]
        print(f"  WARNING: a 2-product scheme cleared the gate: {passers} -- "
              f"the 3-product floor claim would NOT hold.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
