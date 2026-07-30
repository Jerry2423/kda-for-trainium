"""NKIBench -> profiler adapter.

An NKIBench case is split across two files:

  * reference file  (NKIBench/reference/<op>_..._numpy_N.py):
        get_inputs()               -> natural-layout numpy inputs
        forward(*inputs)           -> numpy golden output (natural layout)
        transform_to_nki_inputs(x) -> natural -> tiled inputs the kernel expects
        transform_nki_outputs(k,r) -> tiled kernel output -> natural (for comparison)

  * kernel file     (NKIBench/kernels/<op>_..._0.py):
        @nki.jit def kernel(...)   -> runs in *tiled* space

A profiler backend that understands this contract server-side — given a kernel
module with a `get_inputs()` that yields tiled inputs, and a reference
(`initial_code`) that defines `get_numpy_inputs()` + `forward()` +
`transform_nki_outputs()` — computes the golden in natural space, reshapes the
kernel output back via `transform_nki_outputs`, and compares with relative-L2.

So this adapter does NOT reimplement evaluation. It just:

  1. reads a case from NKIBench/summary.json,
  2. assembles a single self-contained *kernel module* whose `get_inputs()`
     returns tiled inputs (compose reference get_inputs -> transform_to_nki_inputs),
  3. exposes the reference file as-is as the profiler `initial_code`,
  4. packs everything (+ multi-seed + relative-L2 tolerance) into the kwargs a
     profiler backend's profile() call needs (see verify.py for the contract).

The candidate kernel under test can be swapped for any file with a compatible
`kernel(...)` signature; only the kernel source changes between the baseline and
an agent-generated candidate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# NKIBench's correctness contract (AccelOpt/accelopt/kernel_wrapper.py).
NKIBENCH_SEEDS: List[int] = [0, 21, 42, 63, 84]

# Fixed seed used inside the assembled module so the kernel-input path and the
# reference-input path draw IDENTICAL natural data (the profiler does not thread
# its own seed into user get_inputs()). See assembled_module_src for why.
# NOTE: this makes the profiler's multi-seed runs use the same inputs each time
# (correctness gate still valid; it just doesn't add per-seed input variation).
DEFAULT_INPUT_SEED: int = 42
DEFAULT_REL_TOL: float = 2e-5
# mamba uses a looser bound in AccelOpt (eval_numpy.py).
REL_TOL_OVERRIDES: Dict[str, float] = {"mamba": 3e-5}

# AccelOpt compiles single-core with these flags; match them so speedups are
# comparable to the paper's methodology (kernel_wrapper.py:87).
NKIBENCH_COMPILER_FLAGS: List[str] = ["--disable-dge", "--logical-nc-config=1"]

# Standard NKI imports guaranteed at the top of every assembled module so that
# top-level get_inputs/transform_* work even when a candidate kernel omits them.
_NKI_IMPORT_PREAMBLE = "\n".join([
    "import numpy as np",
    "import neuronxcc.nki as nki",
    "import neuronxcc.nki.language as nl",
    "import neuronxcc.nki.isa as nisa",
    "import neuronxcc.nki.typing as nt",
    "from neuronxcc.nki import trace",
    "from neuronxcc.nki.language import par_dim",
])


@dataclass
class NKIBenchCase:
    """One resolved NKIBench operator/case."""

    op: str
    case_id: str
    values: Dict[str, Any]
    reference_path: Path
    baseline_kernel_path: Path
    rel_tol: float

    # The candidate kernel source to evaluate. Defaults to the baseline kernel;
    # set to an agent-generated kernel to score a candidate.
    candidate_kernel_path: Optional[Path] = None

    _reference_src: str = field(default="", repr=False)

    @property
    def name(self) -> str:
        return f"{self.op}[case={self.case_id}]"

    @property
    def kernel_path(self) -> Path:
        return self.candidate_kernel_path or self.baseline_kernel_path

    def reference_src(self) -> str:
        if not self._reference_src:
            self._reference_src = self.reference_path.read_text()
        return self._reference_src

    def kernel_src(self) -> str:
        return self.kernel_path.read_text()

    def assembled_module_src(self) -> str:
        """Combine reference helpers + candidate kernel into one profiler module.

        The kernel file already imports nki and defines `kernel`. We prepend the
        reference file's I/O helpers (get_inputs / forward / transform_*), then
        override `get_inputs` so the profiler's compile+benchmark path receives
        *tiled* inputs, exactly like examples/matmul_kernel.py.
        """
        ref = _strip_module_dunder(self.reference_src())
        kern = self.kernel_src()
        fwd_params = _forward_param_names(ref)

        # The reference defines natural-space get_inputs(); NKIBench names it
        # get_inputs and also ships get_numpy_inputs in some ops. The profiler
        # uses get_numpy_inputs() for the *reference* path, and get_inputs() for
        # the *kernel* (tiled) path. Guarantee both exist with the right meaning.
        parts = [
            "# --- assembled by kda-trainium adapter ---",
            # Guarantee the standard NKI imports exist at module top level. NKIBench
            # baseline kernels carry these, but agent-generated / standalone
            # candidates (e.g. AccelOpt samples) often put imports inside the kernel
            # fn or omit them, which breaks the top-level get_inputs/transform_* we
            # add below. Redundant imports are harmless.
            _NKI_IMPORT_PREAMBLE,
            "# --- NKIBench reference helpers ---",
            ref,
            "",
            "# Capture the natural-layout get_inputs BEFORE we override it below.",
            "_nkibench_orig_get_inputs = get_inputs",
            "",
            "# The remote profiler does NOT seed user get_inputs(), and it calls the",
            "# kernel-input path and the reference-input path SEPARATELY. NKIBench's",
            "# get_inputs draws fresh np.random data each call, so two unseeded draws",
            "# would give the kernel and the reference DIFFERENT inputs -> spurious",
            "# correctness failure. Seed a fixed value before every draw so both paths",
            "# see identical natural data (mirrors the profiler's own example kernel,",
            "# whose get_numpy_inputs() calls np.random.seed(42)).",
            f"_NKIBENCH_SEED = {DEFAULT_INPUT_SEED}",
            "def _nkibench_natural_get_inputs():",
            "    np.random.seed(_NKIBENCH_SEED)",
            "    return _nkibench_orig_get_inputs()",
            "",
            "# get_numpy_inputs(): natural-layout inputs for the reference forward().",
            "# Returns a DICT keyed by forward()'s own parameter names so the profiler",
            "# calls forward(**inputs) correctly. The kernel's params (v1, v2, ...)",
            "# differ from forward's (lhs, rhs / x / ...), so a positional list would",
            "# be mis-named against the kernel signature and then filtered away.",
            f"_nkibench_forward_params = {fwd_params!r}",
            "def get_numpy_inputs():",
            "    _vals = _nkibench_natural_get_inputs()",
            "    return {name: _vals[i] for i, name in enumerate(_nkibench_forward_params)}",
            "",
            "# --- kernel under test (tiled space) ---",
            kern,
            "",
            "# Override get_inputs() to yield TILED inputs for the kernel path,",
            "# derived from the SAME seeded natural draw the reference uses.",
            "def get_inputs():",
            "    return transform_to_nki_inputs(_nkibench_natural_get_inputs())",
            "",
        ]
        return "\n".join(parts)

    def profile_kwargs(self, *, num_cores: int = 1) -> Dict[str, Any]:
        """Kwargs describing this case for the profiler backend (see verify.py).

        Returns a plain dict so verify.py can map it onto whatever request shape
        its profiler backend expects, without this module depending on any
        particular profiler client.
        """
        return {
            "src_code": self.assembled_module_src(),
            "initial_code": self.reference_src(),  # reference for correctness
            "kernel_fn": "kernel",
            "inputs_fn": "get_inputs",
            "forward_fn": "forward",
            "reference_fn": "forward",
            "reference_type": "numpy",
            "reference_inputs_fn": "get_numpy_inputs",
            "multi_seed_seeds": NKIBENCH_SEEDS,
            "tolerance": {
                # NKIBench's metric is relative-L2; the profiler always computes
                # every mode and reports l2_norm_passed / mean_relative_l2
                # separately, so we only supply the tolerances here. rtol carries
                # NKIBench's rel_tol; atol=0 keeps it purely relative.
                "rtol": self.rel_tol,
                "atol": 0.0,
                "auto_adjust_for_dtype": False,
            },
            "compiler_flags": list(NKIBENCH_COMPILER_FLAGS),
            "num_cores": num_cores,
        }


def _forward_param_names(reference_src: str) -> List[str]:
    """Extract forward()'s positional parameter names from reference source.

    Used so get_numpy_inputs() can return a dict keyed by these names, which the
    profiler passes straight into forward(**inputs). Matmul-style references use
    (lhs, rhs); others use (x) or (input_tensor, weight_matrix), etc.
    """
    m = re.search(r"def\s+forward\s*\(([^)]*)\)", reference_src)
    if not m:
        raise ValueError("reference file defines no forward(...) function")
    params: List[str] = []
    for raw in m.group(1).split(","):
        name = raw.split(":")[0].split("=")[0].strip().lstrip("*")
        if name and name != "self":
            params.append(name)
    if not params:
        raise ValueError("forward() has no positional parameters")
    return params


def _strip_module_dunder(src: str) -> str:
    """Drop stray `if __name__ == '__main__'` blocks; keep top-level defs."""
    # References are simple; nothing to strip today, but guard against a
    # trailing main guard sneaking into the assembled module namespace.
    return re.sub(
        r"\nif\s+__name__\s*==\s*['\"]__main__['\"]\s*:.*\Z",
        "\n",
        src,
        flags=re.DOTALL,
    )


def load_summary(nkibench_root: Path) -> Dict[str, Any]:
    return json.loads((nkibench_root / "summary.json").read_text())


def resolve_case(
    nkibench_root: Path,
    op: str,
    case_id: Optional[str] = None,
    candidate_kernel_path: Optional[Path] = None,
) -> NKIBenchCase:
    """Resolve one operator/case from summary.json into an NKIBenchCase."""
    summary = load_summary(nkibench_root)
    if op not in summary:
        raise KeyError(f"operator {op!r} not in summary.json; have {sorted(summary)}")
    cases = summary[op]["cases"]
    if case_id is None:
        if len(cases) != 1:
            raise ValueError(f"{op} has cases {sorted(cases)}; pass case_id explicitly")
        case_id = next(iter(cases))
    if case_id not in cases:
        raise KeyError(f"{op} has no case {case_id!r}; have {sorted(cases)}")

    impl = cases[case_id]["impls"][0]
    return NKIBenchCase(
        op=op,
        case_id=case_id,
        values=cases[case_id]["values"],
        reference_path=(nkibench_root / impl["task"]).resolve(),
        baseline_kernel_path=(nkibench_root / impl["kernel"]).resolve(),
        rel_tol=REL_TOL_OVERRIDES.get(op, DEFAULT_REL_TOL),
        candidate_kernel_path=(
            candidate_kernel_path.resolve() if candidate_kernel_path else None
        ),
    )


# Pilot operators for milestone 1 (span compute / memory / fused bottlenecks).
PILOT_OPS: List[str] = ["matmul", "silu", "rmsnorm_matmul"]
