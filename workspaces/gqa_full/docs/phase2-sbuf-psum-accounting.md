# gqa_full Phase 2 — SBUF / PSUM live-set accounting (AC-6)

Written BEFORE running any candidate (AC-6 requires the table before the run, not
only before promotion). Per-partition budget on trn2 = **192 KB/partition**
(24 MB SBUF / 128 partitions); PSUM = **8 banks**, each bank `[128, 512]` fp32
(2 KB/partition). fp32 = 4 B/element; a `[par_dim(128), F]` tile costs `F·4` B/part.

Fixed shape: B=1, N=4096, QH=16, KH=8, n_rep=2, D=128, T=N/128=32.

## Shared per-head operands (allocated in the `kh` loop; reused across grp∈{0,1}, t_q∈0..31)

| tile | shape | B/part |
|------|-------|--------|
| `identity_local` | [128,128] | 512 |
| `k_t` | [128,4096] | 16384 |
| `v_sb` | [128,4096] | 16384 |
| **shared subtotal** | | **33280 B ≈ 32.5 KB** |

## Per query-tile softmax working buffers (inside grp, t_q)

| tile | shape | B/part |
|------|-------|--------|
| `q_sb` | [128,128] | 512 |
| `q_t` | [128,128] | 512 |
| `score` | [128,4096] | 16384 |
| `exp_t` | [128,4096] | 16384 |
| `attn` (v1 / D1-only only) | [128,4096] | 16384 |
| `neg_max`,`row_sum`,`recip` | [128,1]×3 | 12 |
| `o_sb` | [128,128] | 512 |

## The A-transpose buffer — the only thing D1a changes

- **v1 (interleaved):** one **rotating** `a_t` `[128,128]` = **512 B/part** (transpose
  subtile j, copy, matmul, discard, next j).
- **D1a (tile-local two-phase):** one **resident** `a_t_bank` `[128,4096]` holding all
  32 transposed subtiles at column offset `128·j` = **16384 B/part** (fill in phase A,
  consume in phase B). Delta vs v1 = **+15872 B ≈ +15.5 KB/part**.

## Totals for the three required configurations

### v1 (baseline, for reference)
shared 32.5 KB + tile(q_sb 512 + q_t 512 + score 16384 + exp_t 16384 + attn 16384 +
a_t 512 + vec 12 + o_sb 512) = **32.5 + 50.4 = ~82.9 KB/part** (43% of 192 KB).

### D1-only (M_SUB=1, tile-local two-phase; attn STILL present)
shared 32.5 KB + tile(q_sb 512 + q_t 512 + score 16384 + exp_t 16384 + attn 16384 +
**a_t_bank 16384** + vec 12 + o_sb 512) = **32.5 + 65.9 = ~98.4 KB/part** (51%).
→ +15.5 KB vs v1 (the resident a_t bank). **No spill** (98 KB « 192 KB).

### D2-only (scale-fold + defer-normalize on v1; attn FREED, no a_t bank)
Defer-normalize removes the 4096-wide `attn = exp·recip` buffer entirely (the context
matmul consumes `exp_t` directly, then O is scaled 128-wide). scale-fold removes an
*op*, not a buffer.
shared 32.5 KB + tile(q_sb 512 + q_t 512 + score 16384 + exp_t 16384 + a_t 512 (rotating)
+ vec 12 + o_sb 512) = **32.5 + 34.0 = ~66.5 KB/part** (35%). → −16.4 KB vs v1.

### D1+D2 (tile-local two-phase + scale-fold + defer-normalize)
shared 32.5 KB + tile(q_sb 512 + q_t 512 + score 16384 + exp_t 16384 +
**a_t_bank 16384** (attn FREED) + vec 12 + o_sb 512) = **32.5 + 49.5 = ~82.0 KB/part** (43%).
**KEY:** D2's freed `attn` (−16 KB) almost exactly pays for D1a's resident `a_t_bank`
(+16 KB) → **D1+D2 footprint ≈ v1's footprint**. No spill.

## PSUM banks (all configurations)

| bank | when | count |
|------|------|-------|
| score matmul `acc` [128,512] | score build loop (dead before context) | 1 (rotating over 8 chunks) |
| `q_t_ps` [128,128] | q transpose | 1 (transient) |
| D1a phase-A `a_t_ps` [128,128] | transpose-all | 1 declared/iter (compiler pipelines ~2–3) |
| `o_psum` [128,128] | phase-B accumulator, loop-carried over 32 j | 1 (held across the stream) |

Peak live in the context region = `o_psum` (1) + phase-A `a_t_ps` pipeline (~2–3) ≤ **~4 banks ≤ 8**. ✓
(v1 peak was the same ~2: `o_psum` + rotating `a_t_ps`. D1a does not raise the bank count,
only lengthens the phase-A transpose pipeline before phase B drains `o_psum`.)

## D1b M_SUB sweep — live set under the D2c base (buffer set now fixed)

The D2c base (two-phase + scale-fold + defer-normalize) freed the `attn` buffer.
Batching `M_SUB` query tiles holds `M_SUB` resident `a_t_bank` slots simultaneously
(one `[128, M_SUB·N]` tile = **M_SUB·16 KB/part**) plus `M_SUB` reciprocals (tiny).
The score/exp_t/q working tiles are declared INSIDE the phase-1 `mm` loop, so they
ROTATE (one tile's ~33 KB live at a time, not M_SUB copies). Peak SBUF:

`32.5 KB (shared k_t+v_sb+identity) + M_SUB·16 KB (a_t_bank) + ~33 KB (rotating score
16 + exp_t 16 + q 1) + ~0.5 KB (o_sb) + M_SUB·tiny (recip)`

| M_SUB | a_t_bank | peak SBUF/part | vs 192 KB | verdict |
|-------|----------|----------------|-----------|---------|
| 1 | 16 KB | ~82 KB | 43% | fits (== D2c) |
| 2 | 32 KB | ~98 KB | 51% | fits |
| 4 | 64 KB | ~130 KB | 68% | fits |
| **8** | **128 KB** | **~194 KB** | **>100%** | **spill probe (expected over budget)** |

So the realistic no-spill sweep is `{1,2,4}` with `8` as the confirming spill/
regression upper probe — the `a_t_bank` (16 KB/part/tile) is the dominant scaling term.

**PSUM (all M_SUB):** phase-1 declares `q_t_ps` + `acc` (score) + `a_t_ps` transiently
(rotating, dead before phase 2); phase 2 declares one `o_psum [128,128]` per `mm`
inside the `mm` loop, so it ROTATES (≤ ~2 banks the compiler pipelines). Peak live
≤ ~3–4 banks ≤ **8** at every M_SUB (M_SUB does NOT hold M_SUB o_psum banks live
simultaneously — each drains and stores before the next mm). ✓
