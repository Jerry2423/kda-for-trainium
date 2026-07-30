"""Generate per-operator, per-phase NKI optimization prompts for the pilot ops.

Keeps the GPU KDA prompt *structure* (Kernel Information / Reference computation /
Acceptance / Workflow Requirements / Phase Goal) and swaps in the Trainium/NKI
payload. Run from the prompts/ dir:  python _gen_prompts.py
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

# Per-op payload. `values`/`shapes` are grounded in the NKIBench reference files.
# `case`, `reference`, `baseline` mirror AccelOpt/NKIBench/summary.json exactly.
# `rel_tol` defaults to 2e-5; mamba relaxes to 3e-5 (its recurrence amplifies fp
# error). All ops are fp32, 5 seeds [0,21,42,63,84].
OPS = {
    # ---- pilot ops (baselines already cached) --------------------------------
    "silu": {
        "case": "2",
        "op_type": "elementwise activation",
        "shapes": "x: (4096, 7168) float32  ->  tiled (128, 32, 7168)",
        "reference": "silu_M4096_N7168_numpy_0.py",
        "baseline": "silu_M4096_N7168_0.py",
        "spec": "SiLU / swish: `y = x / (1 + exp(-x))`, elementwise over a (4096, 7168) fp32 tensor.",
        "bottleneck_hint": "memory / vector-engine bound: the arithmetic is trivial, so DMA "
                           "bandwidth and vector/scalar engine throughput dominate. AccelOpt "
                           "found ~1.67x here, so there is real headroom.",
        "directions": "- fuse the exp / add / reciprocal / multiply chain to cut intermediate SBUF traffic\n"
                       "- larger free-dim tiles to amortize DMA and improve engine pipelining\n"
                       "- overlap load/compute/store (double-buffering) across the tile loop\n"
                       "- pick the cheapest instruction sequence for sigmoid (activation vs tensor_scalar+reciprocal)",
    },
    "matmul": {
        "case": "3",
        "op_type": "matmul (GEMM)",
        "shapes": "lhs: (4096, 5120), rhs: (5120, 12288) float32  ->  tiled "
                  "lhs (32,128,40,128), rhs (40,128,12288); out (32,128,12288)",
        "reference": "matmul_M4096_N12288_K5120_numpy_2.py",
        "baseline": "matmul_M4096_N12288_K5120_0.py",
        "spec": "Dense matmul `out = lhs @ rhs`, M=4096, K=5120, N=12288, fp32.",
        "bottleneck_hint": "compute (PE-array / TensorEngine) bound at this size; the goal is high "
                           "MFU: keep the PE array fed, minimize weight reloads, and tile K/N to "
                           "fit PSUM banks (free <= 512).",
        "directions": "- stationary/moving operand choice and reload minimization (fast weight load)\n"
                       "- K-dim accumulation tiling that keeps PSUM banks full (free <= 512)\n"
                       "- N/M tiling for temporal locality of the moving operand\n"
                       "- overlap the rhs/lhs DMA loads with matmul via double-buffering\n"
                       "- reduce transpose overhead in the tiled layout",
    },
    "rmsnorm_matmul": {
        "case": "2",
        "op_type": "fused RMSNorm + matmul",
        "shapes": "x: (4096, 1024), w: (1024, 2048) float32  ->  out (4096, 2048)",
        "reference": "rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py",
        "baseline": "rmsnorm_matmul_M4096_N2048_K1024_0.py",
        "spec": "RMSNorm over K then matmul: `normalized = x / sqrt(mean(x^2, axis=1))`, "
                "then `out = normalized @ w`. M=4096, K=1024, N=2048, fp32.",
        "bottleneck_hint": "mixed: a memory-bound reduction (RMSNorm over K) feeding a compute-bound "
                           "matmul. The main win is fusing the norm into the matmul's input staging "
                           "so normalized rows are consumed from SBUF without a round-trip to HBM.",
        "directions": "- fuse the RMSNorm reduction + normalization into the matmul input load\n"
                       "- keep normalized activations in SBUF (avoid HBM round-trip between norm and matmul)\n"
                       "- tile the K reduction to overlap with matmul accumulation\n"
                       "- balance vector-engine (norm) vs PE-array (matmul) pressure",
    },
    # ---- remaining 11 ops ----------------------------------------------------
    "swiglu": {
        "case": "2",
        "op_type": "fused SwiGLU MLP (3 matmuls + gating)",
        "shapes": "x: (4096, 1024); w_up,w_gate: (1024, 3072); w_down: (3072, 1024) "
                  "float32  ->  out (4096, 1024). M=4096, K=1024, N=3072.",
        "reference": "swiglu_M4096_N3072_K1024_numpy_2.py",
        "baseline": "swiglu_M4096_N3072_K1024_0.py",
        "spec": "SwiGLU feed-forward: `up = x@w_up`, `gate = x@w_gate`, "
                "`h = (gate * sigmoid(gate)) * up`, `out = h @ w_down`. fp32.",
        "bottleneck_hint": "compute-bound (three GEMMs) with an elementwise SiLU-gate between them. "
                           "Win = fuse the gate/up activation into the h@w_down staging and reuse x "
                           "across the up/gate matmuls without reloading.",
        "directions": "- share the single x load across the up and gate matmuls (no double reload)\n"
                       "- fuse the SiLU gate + elementwise multiply into the down-matmul input staging\n"
                       "- keep the (M,N) intermediate in SBUF between the gate and down matmuls\n"
                       "- tile K/N to keep PSUM banks full (free <= 512) and overlap weight DMA",
    },
    "matmul_add_rmsnorm": {
        "case": "1",
        "op_type": "fused matmul + residual add + RMSNorm",
        "shapes": "x: (4096, 2048), w: (2048, 2048), z: (4096, 2048), g: (2048,) "
                  "float32  ->  out (4096, 2048). M=4096, K=2048, N=2048.",
        "reference": "matmul_add_rmsnorm_M4096_N2048_K2048_numpy_1.py",
        "baseline": "matmul_add_rmsnorm_M4096_N2048_K2048_0.py",
        "spec": "`y = x@w + z`; `rms = sqrt(mean(y^2, axis=-1) + eps)`; `out = y * g / rms`. "
                "matmul feeds a residual add then a row-wise RMSNorm over N. fp32, eps=1e-5.",
        "bottleneck_hint": "compute-bound matmul feeding a memory-bound row reduction. Win = keep the "
                           "matmul output tile in SBUF/PSUM and do the add+RMSNorm in place, avoiding "
                           "an HBM round-trip of the (M,N) intermediate.",
        "directions": "- fuse residual add + RMSNorm onto the matmul output tile before it leaves SBUF\n"
                       "- compute the row reduction over N incrementally as N-tiles are produced\n"
                       "- avoid materializing the full (M,N) intermediate in HBM\n"
                       "- balance PE-array (matmul) vs vector-engine (reduction) pressure",
    },
    "add_rmsnorm_matmul": {
        "case": "2",
        "op_type": "fused residual add + RMSNorm + matmul",
        "shapes": "x,z: (4096, 1024), w: (1024, 2048), g: (1024,) float32  ->  "
                  "out (4096, 2048). M=4096, K=1024, N=2048.",
        "reference": "add_rmsnorm_matmul_M4096_N2048_K1024_numpy_1.py",
        "baseline": "add_rmsnorm_matmul_M4096_N2048_K1024_0.py",
        "spec": "`y = x + z`; RMSNorm over K: `y = y / sqrt(mean(y^2, axis=-1) + eps) * g`; "
                "`out = y @ w`. Add + norm over K feed a matmul. fp32, eps=1e-5.",
        "bottleneck_hint": "memory-bound add+reduction over K feeding a compute-bound matmul. Win = "
                           "fuse the add+RMSNorm into the matmul input staging so normalized rows are "
                           "consumed from SBUF without an HBM round-trip.",
        "directions": "- fuse residual add + RMSNorm into the matmul input load (rows stay in SBUF)\n"
                       "- tile the K reduction to overlap with matmul accumulation\n"
                       "- avoid a separate HBM write/read of the normalized activations\n"
                       "- balance vector-engine (norm) vs PE-array (matmul) pressure",
    },
    "gqa_full": {
        "case": "0",
        "op_type": "grouped-query attention (full softmax attention)",
        "shapes": "q: (1,4096,16,128), k,v: (1,4096,8,128) float32  ->  out. "
                  "B=1, N=4096, QH=16, KH=8 (n_rep=2), D=128.",
        "reference": "gqa_full_B1_N4096_QH16_KH8_D128_numpy_2.py",
        "baseline": "gqa_full_B1_N4096_QH16_KH8_D128_0.py",
        "spec": "GQA: repeat KV heads to n_rep=2, `attn = softmax(q@k^T / sqrt(D))`, `out = attn@v`. "
                "Full (non-causal) attention over N=4096. fp32.",
        "bottleneck_hint": "two matmuls (scores, context) bridged by a row softmax over N=4096; the "
                           "N*N scores matrix is large. Win = flash-attention-style tiling so the "
                           "scores tile never fully materializes and softmax is computed online.",
        "directions": "- flash-attention tiling: stream K/V tiles, keep running max + sum for online softmax\n"
                       "- avoid materializing the full (N,N) scores matrix in HBM\n"
                       "- exploit GQA head sharing (n_rep=2): reuse each KV head across 2 Q heads\n"
                       "- overlap the score matmul, softmax, and context matmul across tiles",
    },
    "rope_single_freq_apply": {
        "case": "1",
        "op_type": "rotary position embedding (elementwise)",
        "shapes": "x: (128, 262144), freqs_cos/sin: (64, 262144) float32. "
                  "D=128 (half=64), B*H*N=262144.",
        "reference": "rope_single_freq_apply_B1_H64_N4096_D128_numpy_1.py",
        "baseline": "rope_single_freq_apply_B1_H64_N4096_D128_0.py",
        "spec": "RoPE: split x into halves x0,x1 (over D); "
                "`out0 = x0*cos - x1*sin`, `out1 = x0*sin + x1*cos`; concat over D. Elementwise. fp32.",
        "bottleneck_hint": "memory / vector-engine bound: pure elementwise multiply-add over a large "
                           "tensor. DMA bandwidth and vector throughput dominate; arithmetic is minimal.",
        "directions": "- fuse the four multiplies + two add/sub into minimal vector-engine passes\n"
                       "- large free-dim tiles to amortize DMA over the 262144 columns\n"
                       "- overlap load/compute/store (double-buffering) across tiles\n"
                       "- lay out the D-halves so cos/sin are read once and reused for both outputs",
    },
    "bmm": {
        "case": "2",
        "op_type": "batched matmul",
        "shapes": "lhs: (16,4096,64), rhs: (16,64,4096) float32  ->  out (16,4096,4096). "
                  "B=16, M=4096, K=64, N=4096.",
        "reference": "bmm_B16_M4096_K64_N4096_numpy_1.py",
        "baseline": "bmm_B16_M4096_K64_N4096_0.py",
        "spec": "Batched matmul `out[b] = lhs[b] @ rhs[b]` for b in 0..15. K=64 is small; "
                "M=N=4096 are large. fp32.",
        "bottleneck_hint": "small contraction (K=64) over 16 independent large (M,N) matmuls. Low "
                           "arithmetic intensity per element => partly DMA-bound; keeping the PE array "
                           "busy across the batch and reusing loads matters.",
        "directions": "- schedule the 16 batch matmuls to keep the PE array continuously fed\n"
                       "- tile M/N to fill PSUM banks (free <= 512) despite the small K=64\n"
                       "- overlap per-batch lhs/rhs DMA with compute (double-buffering)\n"
                       "- minimize transpose/layout overhead for the K=64 contraction",
    },
    "bmm_softmax": {
        "case": "2",
        "op_type": "batched matmul + softmax",
        "shapes": "lhs: (16,4096,64), rhs: (16,64,4096) float32  ->  out (16,4096,4096). "
                  "B=16, M=4096, K=64, N=4096.",
        "reference": "bmm_softmax_B16_K64_M4096_N4096_numpy_1.py",
        "baseline": "bmm_softmax_B16_K64_M4096_N4096_0.py",
        "spec": "Batched matmul then softmax over axis 2 (the N=4096 dim): "
                "`x = lhs@rhs`, `out = softmax(x, axis=2)`. fp32.",
        "bottleneck_hint": "batched matmul (small K=64) feeding a row softmax over N=4096. Win = fuse "
                           "the softmax onto the matmul output tiles with online max/sum so the large "
                           "(B,M,N) scores never fully round-trip through HBM.",
        "directions": "- fuse the softmax (online max + sum) onto the matmul output tiles\n"
                       "- avoid materializing the full (B,M,N) scores in HBM\n"
                       "- keep the PE array fed across the 16-way batch\n"
                       "- overlap matmul, max-reduce, exp, and normalize across tiles",
    },
    "transpose_matmul": {
        "case": "2",
        "op_type": "matmul with transposed lhs",
        "shapes": "lhs: (2048, 4096) [stored K-major], rhs: (2048, 10944) float32  ->  "
                  "out (4096, 10944). M=4096, K=2048, N=10944.",
        "reference": "transpose_matmul_M4096_K2048_N10944_numpy_1.py",
        "baseline": "transpose_matmul_M4096_K2048_N10944_0.py",
        "spec": "`out = lhs^T @ rhs` where lhs is (K,M) and transposed to (M,K) before the matmul. "
                "M=4096, K=2048, N=10944, fp32.",
        "bottleneck_hint": "compute-bound GEMM whose lhs arrives transposed. The PE array already "
                           "contracts over the partition dim, so the (K,M) layout may be consumed "
                           "directly — avoiding an explicit transpose is the main lever.",
        "directions": "- consume the (K,M) lhs directly as the stationary operand (avoid an explicit transpose)\n"
                       "- K-accumulation tiling that keeps PSUM banks full (free <= 512)\n"
                       "- N-tiling over the wide N=10944 for locality and DMA amortization\n"
                       "- overlap lhs/rhs DMA with matmul via double-buffering",
    },
    "lora": {
        "case": "2",
        "op_type": "LoRA-augmented matmul (low-rank residual)",
        "shapes": "x: (4096,5120), w: (5120,12288), a: (5120,128), b: (128,12288) "
                  "float32  ->  out (4096,12288). M=4096, K=5120, N=12288, R=128.",
        "reference": "lora_M4096_N12288_K5120_R128_numpy_1.py",
        "baseline": "lora_M4096_N12288_K5120_R128_0.py",
        "spec": "`out = x@w + (x@a)@b`. A large base matmul plus a low-rank (R=128) update: "
                "`x@a` is (M,R), then `@b` is (M,N). fp32.",
        "bottleneck_hint": "dominated by the large base GEMM x@w; the low-rank path (x@a@b) is cheap "
                           "but adds a second output accumulation. Win = fuse the low-rank result into "
                           "the base matmul's output accumulation without an extra HBM round-trip.",
        "directions": "- accumulate the low-rank path (x@a)@b into the same output tile as x@w\n"
                       "- reuse the single x load across both the base and low-rank matmuls\n"
                       "- keep the tiny (M,R=128) intermediate in SBUF\n"
                       "- tile K/N for the base GEMM (free <= 512) and overlap weight DMA",
    },
    "adamw": {
        "case": "2",
        "op_type": "AdamW optimizer step (elementwise)",
        "shapes": "theta,g,m,v: (10944, 2048) float32  ->  new_theta (10944, 2048). "
                  "M=10944, N=2048.",
        "reference": "adamw_M10944_N2048_numpy_1.py",
        "baseline": "adamw_M10944_N2048_0.py",
        "spec": "AdamW update: `m' = 0.9m + 0.1g`, `v' = 0.999v + 0.001g^2`, "
                "`theta' = theta - 1e-5*theta - 0.01*m'/(sqrt(v'*1000) + 1e-8)`. Elementwise over 4 tensors. fp32.",
        "bottleneck_hint": "memory-bound: reads 4 large tensors, does a modest elementwise chain, writes 1. "
                           "DMA bandwidth dominates; the win is fusing the whole update into one pass so "
                           "each element is read once and written once.",
        "directions": "- fuse the entire m/v/theta update into a single vector-engine pass per tile\n"
                       "- large free-dim tiles to amortize the 4-input / 1-output DMA\n"
                       "- overlap load/compute/store (double-buffering) across tiles\n"
                       "- pick the cheapest instruction sequence for the sqrt/reciprocal chain",
    },
    "mamba": {
        "case": "2",
        "op_type": "Mamba selective-scan (SSM recurrence)",
        "shapes": "delta,u: (256,7168), a: (256,16), b,c: (16,7168) float32  ->  "
                  "out (256,7168). C=256, M=7168 (sequence), S=16 (state).",
        "reference": "mamba_M7168_C256_S16_numpy_1.py",
        "baseline": "mamba_M7168_C256_S16_0.py",
        "spec": "Selective-scan SSM: `deltaA = exp(delta*a)`, `deltaB_u = delta*b*u`; sequential scan "
                "over M: `state_i = deltaA_i*state_{i-1} + deltaB_u_i`; `out = sum_S(c * state)`. fp32.",
        "bottleneck_hint": "a sequential recurrence over the M=7168 axis with a small S=16 state — the "
                           "scan dependency is the challenge. Win = a parallel/associative-scan or "
                           "chunked-scan formulation that exposes parallelism the naive loop hides. "
                           "AccelOpt's optimized sample reaches ~1.6x here.",
        "directions": "- reformulate the sequential scan as a chunked / associative (parallel) scan\n"
                       "- keep the small S=16 state in SBUF across the scan\n"
                       "- precompute deltaA / deltaB_u tiles and overlap with the scan\n"
                       "- balance the elementwise (exp, mul) vs the reduction (sum over S) work",
        "rel_tol": "3e-5",
    },
}

PHASE_GOALS = {
    1: ("Research + first correct NKI kernel",
        "Produce the first CORRECT NKI kernel for this operator. Study the numpy "
        "reference and the tiled input/output layout, use the NKI docs and API skills "
        "to get the language right, and prioritize a clean, correct baseline over "
        "speed. It must pass the relative-L2 gate across all five seeds. You may start "
        "from the NKIBench baseline kernel's structure, but understand every tile."),
    2: ("Profile-driven optimization",
        "Start from the best correct kernel. Use the profiling skills to identify the "
        "real bottleneck, enumerate optimization directions, rank them by expected "
        "benefit vs risk, and explore each for AT MOST five iterations. For each "
        "direction collect before/after latency (verify.py) and profiling evidence to "
        "justify keep / revise / reject. Never regress correctness."),
    3: ("Regime / shape specialization",
        "Analyze where time goes across the tensor's structure and specialize only "
        "where the measured win justifies the added complexity (e.g. tile-size regimes, "
        "partition/free splits, edge tiles). Evaluate the final candidate on the full "
        "correctness gate and report speedup vs baseline."),
}

TEMPLATE = """\
# {op} — Phase {phase} Prompt ({phase_title})

Develop an NKI (Neuron Kernel Interface) kernel that minimizes on-device latency
while preserving numerical correctness. The target hardware is **AWS Trainium
(trn2)**; implement in **NKI Python** with a single `@nki.jit def kernel(...)`
entry point whose signature matches the baseline's tiled inputs.

## Kernel Information

- Operator: `{op}`  (NKIBench case `{case}`)
- Operation type: {op_type}
- Shapes / dtype: {shapes}
- Numpy reference: `../AccelOpt/NKIBench/reference/{reference}`
- Baseline kernel: `../AccelOpt/NKIBench/kernels/{baseline}`

## Reference computation

{spec}

The reference `forward(...)` runs in natural (untiled) layout. The kernel runs in
TILED layout (partition dim <= 128); `transform_to_nki_inputs` / `transform_nki_outputs`
in the reference file bridge the two. The harness handles this reconciliation — your
kernel only needs to consume the tiled inputs and produce the tiled output shape.

## Acceptance (NKIBench contract)

- Correct <=> every seed in `[0, 21, 42, 63, 84]` passes relative-L2
  `||v_k - v_r||_2 < {rel_tol} * ||v_r||_2`, fp32.
- Score = speedup over the baseline kernel = `baseline_latency / candidate_latency`
  (p50 on-device time), single-core, compiled with `--disable-dge --logical-nc-config=1`.

Run the KDA loop from this task's workspace directory: `workspaces/{op}/`.
All evidence paths below are relative to it; `verify.py` lives at the repo root.

Validate / score a candidate (from `workspaces/{op}/`):

```bash
python3 \\
    ../../verify.py --op {op} --candidate runs/<your_kernel>.py --fast
```

(Drop `--fast` for the full 5-seed / higher-iter measurement before promoting.)

Likely bottleneck: {bottleneck_hint}

## Tools

Performance data comes from the REMOTE profiler (see CLAUDE.md), NOT the local
local build-based profiling skills — read the per-engine / MFU / HBM digest that
`verify.py` prints and the profiler's `summary_metrics`.

- `kernel-cost-analysis` — theoretical per-engine cost and bottleneck engine (the floor to compare measured metrics against).
- `kernel-optimization-kb` — real optimization precedents from git history.
- `kernel-accuracy-debugging` — when the L2 gate fails.
- `nki-concept-docs` / `nki-api-reference` — NKI concepts and API signatures.

## Workflow requirements

All evidence lives under this task's `workspaces/{op}/`:

- Record every performance-related change in `benchmark.csv`.
- Record every candidate in `candidates.jsonl` with parent links as a DAG.
- Keep profiling evidence for each major direction under `profile/`.
- Put candidate kernels in `runs/`; never edit the NKIBench baseline or reference.

{phase_directions}## Phase {phase} goal

{phase_goal}

## How to run this phase

**Automated (recommended):** from the repo root, `bash scripts/run-op.sh {op}`
drives all three phases headlessly — for each phase it makes the agent write the
draft, runs `/humanize:gen-plan` and `/humanize:start-rlcr-loop` for you, and
commits between them. To run only this phase: `--from-phase {phase} --to-phase {phase}`.

**Manual:** in an interactive Claude Code session started inside `workspaces/{op}/`,
first write the implementation-plan draft to `docs/draft-phase{phase}.md`, then run
`/humanize:gen-plan --input docs/draft-phase{phase}.md --output docs/plan-phase{phase}.md`,
then `/humanize:start-rlcr-loop docs/plan-phase{phase}.md --skip-quiz`.
(Per-phase filenames matter: gen-plan errors if the output file already exists.)
"""


def main() -> None:
    for op, cfg in OPS.items():
        (HERE / op).mkdir(exist_ok=True)
        for phase in (1, 2, 3):
            title, goal = PHASE_GOALS[phase]
            directions = ""
            if phase == 2:
                directions = "## Candidate optimization directions\n\n" + cfg["directions"] + "\n\n"
            text = TEMPLATE.format(
                op=op, phase=phase, phase_title=title, case=cfg["case"],
                op_type=cfg["op_type"], shapes=cfg["shapes"],
                reference=cfg["reference"], baseline=cfg["baseline"],
                spec=cfg["spec"], bottleneck_hint=cfg["bottleneck_hint"],
                phase_directions=directions, phase_goal=goal,
                rel_tol=cfg.get("rel_tol", "2e-5"),
            )
            (HERE / op / f"phase{phase}.md").write_text(text)
            print(f"wrote {op}/phase{phase}.md")


if __name__ == "__main__":
    main()
