# rmsnorm_matmul (M4096 N2048 K1024, fp32) — Shape-Specialization Closure (Phase 3, AC-6)

`out = rmsnorm(x) @ w`, RMSNorm over the K axis. Fixed shape M=4096, N=2048, K=1024, fp32.
Scored single-core on the remote profiler (`--disable-dge --logical-nc-config=1`), correctness
by NKIBench relative-L2 < 2e-5 on seeds `[0,21,42,63,84]`.

Phase 3 = regime/shape specialization: specialize only where the tensor's structure admits a
lever *and* a measured win justifies the complexity. The finding below is that this fixed shape
offers **no specialization surface** — every classic phase-3 lever is either vacuous (the shape
has no ragged structure to exploit) or already pinned at its hardware constraint. The absence of a
lever is recorded here as a first-class result, not a gap.

## Per-lever closure

| Lever | Applies? | Shape / constraint reason |
|---|---|---|
| **Edge / partial tiles** | **No — vacuous** | Every dim divides its tile size evenly: M=4096=32·128, K=1024=8·128, N=2048=4·512. There is no ragged tile, no remainder loop, and no masked/partial tile anywhere in the kernel. Edge-tile specialization requires an edge to specialize; there is none. |
| **Tile-size / partition-free regime** | **No — layout forced** | `nc_matmul(stationary, moving) = stationary.T @ moving` requires the contraction dim (k_in) on the PARTITION axis of both SBUF operands, and produces an output tile `[m_in(par), n(free)]`. So m_in is forced onto the stationary/partition axis and n onto the moving/free axis. You cannot swap m↔n without transposing the entire N=2048-wide result back — a cost far larger than any tiling gain. There is no free choice of partition/free split to specialize. |
| **N-chunk (moving-free) width** | **No — already maximal** | The kernel uses N_CHUNK=512 = `psum_fmax` = exactly one fp32 PSUM bank in the free dim — the documented per-matmul moving-free maximum (knowledgebase precedent `6288aaad`: "tile budget is psum_fmax, not pmax"). Larger is impossible (exceeds a PSUM bank); smaller wastes the systolic array's streaming width. Already at the constraint. |
| **M-blocking** (the sibling `matmul` task's phase-2 win) | **No — vacuous** | That win removed *redundant w HBM reloads* by reusing a loaded w-tile across B output-row tiles. Here w is only 8 MB (64 KB/partition, budget 192 KB) and is loaded **fully resident once**, then reused across all 32 M-tiles; each x tile is loaded exactly once. DMA is only ~19–20% busy and HBMrd=25MB is already near the minimal (one read of x + one read of w). There is no redundant HBM traffic to block for — the reuse M-blocking would create already exists. |
| **LNC2 / multi-core sharding** | **No — out of contract** | Scored single-core (`--logical-nc-config=1`). LNC2 sharding is not a lever on this harness; using it would change the scoring contract, not optimize within it. |

## Why nothing else is left inside the fp32 path

The time budget is dominated by the PE, and the PE is at its fp32 rate ceiling:

```
v1 control (fresh, same-session, full-5-seed): latency=0.4714 ms
  MFU=46%  PE=97%  Vec=15%  Scl=11%  DMA=19%  HBMrd=25MB  HBMwr=34MB
```

- **MFU=46% is the fp32 ceiling, not inefficiency.** The bf16-peak floor for this GEMM is
  `2·M·N·K / (128·128·2.4e9·2) = 218.5 µs`; `218.5 / 471.4 = 46.3%` = the measured MFU. MFU is
  literally measured-latency-vs-bf16-peak, so a *correct fp32* GEMM is capped near ~46% by
  construction. The ~2.16× gap to the bf16 floor is the fp32 PE-rate penalty (the array is
  bf16-native; a same-kernel bf16 swap ran ~3.23× faster — calibration node
  `rmsnorm_matmul_probe_bf16_calib_D4`).
- **RMSNorm and data movement are hidden.** Vec 15% / Scl 11% / DMA 19% all sit well under the
  PE-bound matmul. The post-scale eviction fold (`v2_postscale`) moved the per-row scale off the
  full-width input onto the narrow output tiles with no latency change, confirming the norm is not
  on the critical path.
- **The only non-floor PE work — the x-sub-tile transpose — is fully hidden.** Phase-2 probes
  removed it three different ways: `nc_transpose(engine=vector)` (available but regresses +2.08%,
  Vec 7→90% co-bottleneck), `dma_transpose` (fp32-ineligible, compiler exit 70), and
  `load_transpose2d` (removed the transpose from the PE entirely, full-5-seed PASS, PE stayed 97%,
  latency within noise). Even *fully removing* it moved nothing measurable → it was never on the
  wall-clock critical path.
- **The main-matmul stationary fill is also already hidden (Phase-3 measurement).** The
  stationary-activation reuse reorder (`runs/rmsnorm_matmul_v3_stationary_reuse.py`) cut stationary
  fills 4× (1024→256) yet landed at 0.4723 ms (+0.19% vs control, within noise), PE unchanged at
  97%. So the stationary fills were already pipelined behind the moving stream — no exposed
  main-matmul fill lever exists either.

## Conclusion

Every shape-specialization lever is closed by the shape or by a hardware constraint, and every
within-fp32 micro-lever (transpose placement, stationary-fill amortization) is measured to be
already hidden under the PE-bound matmul. The **only** remaining lever that can move latency is
breaking the fp32 rate ceiling itself, which requires lower-precision arithmetic and therefore must
clear the 2e-5 relative-L2 gate — handled separately by the offline-gated compensated bf16×2 split
(`runs/offline_bf16_split_sim.py`). Within the fp32 contract, v1 (1.066×) is at the floor.
