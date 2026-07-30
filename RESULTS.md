# KDA on NKIBench — Results

Optimization results for the 14 NKIBench operators driven through the KDA
three-phase loop (research → correct baseline → profile-driven opt → shape
specialization) and scored on the **remote Trainium profiler (trn2, single-core,
`--disable-dge --logical-nc-config=1`)**.

- **Correctness gate:** NKIBench relative-L2 across seeds `[0, 21, 42, 63, 84]`,
  `rel_tol = 2e-5` (`3e-5` for `mamba`). Every promoted kernel passes the full
  5-seed gate.
- **Speedup:** `baseline_latency / candidate_latency` (p50 on-device), both measured
  on the same remote/trn2 backend so the ratio is apples-to-apples. Absolute
  latencies differ from the trn1 paper baselines; the ratio is the comparable
  quantity.
- **Aggregate:** geomean of per-op speedups.

Numbers below are regenerated from the committed evidence
(`workspaces/<op>/benchmark.csv`, `candidates.jsonl`) via `python3 scripts/summary.py`.

## Headline

| | |
|---|---|
| Ops with a passing kernel | **14 / 14** |
| **Geomean speedup** | **1.986×** |
| Range | 1.026× … 4.879× |

## Per-operator results

Ranked by speedup. Latency is p50 on-device (ms); "speedup" is the best passing
screen (what `summary.py` reports); the deployed/promoted kernel is in the last
column. See the two footnotes where the best screen ≠ the promoted kernel.

| # | op | speedup | baseline (ms) | promoted (ms) | winning lever | promoted kernel |
|---|----|--------:|--------------:|--------------:|---------------|-----------------|
| 1 | matmul_add_rmsnorm | **4.879×** | 3.7685 | 0.7724 | compensated bf16x2 3-product split (breaks fp32 floor) | `matmul_add_rmsnorm_v2_bf16_split.py` ¹ |
| 2 | add_rmsnorm_matmul | **4.632×** | 1.8593 | 0.4014 | bf16x2 3-product split; on-device err = fp32⊕bf16 in quadrature | `add_rmsnorm_matmul_v3_bf16_split.py` |
| 3 | gqa_full | **3.614×** | 15.5795 | 4.3109 | drop softmax-max (serialization barrier) → exp-from-PSUM | `gqa_full_v7a_exp_from_psum.py` |
| 4 | silu | **3.478×** | 1.0224 | 0.2940 | DMA-bound; finer free-axis `affine_range` tiling (go finer, not wider) | `silu_v3_s7.py` |
| 5 | adamw | **2.384×** | 1.3050 | 0.5474 | mask-free `(128,ITERS,CH)` reshape-view + exhaustive CH sweep → wide-edge DMA lobe | `adamw_v2_ch1824.py` |
| 6 | bmm_softmax | **1.946×** | 7.2901 | 3.7462 | two-phase transpose-all + max-negate fold, M_SUB=16 | `bmm_softmax_v4.py` |
| 7 | mamba | **1.756×** | 1.2583 | 0.7166 | seq-tiling + shared b/c broadcast + static unroll | `mamba_v4_s4unroll.py` |
| 8 | rope_single_freq_apply | **1.641×** ² | 1.1418 | 0.6960 | layout-B 128-partition pack + 0/±1 permutation matmul on idle PE | `rope_v3_layoutB_pe.py` |
| 9 | rmsnorm_matmul | **1.363×** | 0.5026 | 0.3688 | offline-gated bf16x2 split breaks the fp32 floor | `rmsnorm_matmul_v4_bf16_split.py` |
| 10 | transpose_matmul | **1.334×** | 4.8496 | 3.6354 | bf16x2 split + M_BLK=16 (read-floor safe) | `tmm_v3_mblk16_bf16_split.py` |
| 11 | lora | **1.297×** | 14.6645 | 11.3065 | fp32 M-block + base-GEMM-only bf16x2 split | `lora_v3_bf16_split.py` |
| 12 | matmul | **1.274×** | 13.5814 | 10.6604 | compensated bf16x2 3-product split | `matmul_v3_bf16_split.py` |
| 13 | bmm | **1.253×** | 2.5500 | 2.0352 | two-phase transpose/matmul + whole-batch M-block | `bmm_v2.py` |
| 14 | swiglu | **1.026×** | 2.0743 | 2.0217 | down-GEMM-only bf16x2 split + M-block B=4 | `swiglu_v3_mblock.py` |

¹ `matmul_add_rmsnorm`: the deployed final winner is `v2_bf16_split` at **4.879×**
(0.7724 ms); `v1` (fp32, 3.920×) is retained as the fp32 fallback. (`summary.py`'s
`promoted` field tags `v1` from phase 1; the true final is `v2_bf16_split`.)

² `rope_single_freq_apply`: `summary.py`'s best *screen* is `rope_v3_layoutB_pe_w256`
at 1.662×, but that finer-W point sat **inside the noise band** and was deliberately
**not promoted**. The deployable kernel is `rope_v3_layoutB_pe` at **1.641×**
(0.6960 ms). The table reports the promoted number.

## Reproducing

```bash
cd /path/to/kda-trainium
source scripts/setup.sh                              # backend env + auth (see scripts/)
python3 scripts/summary.py                           # regenerate this table
# score any promoted kernel (full 5-seed gate):
PY=python3
cd workspaces/<op> && $PY ../../verify.py --op <op> --candidate runs/<promoted>.py
```
