# Python Test Guide

These tests cover the Python API layer for Purlin.

## Prerequisites

- Install the package in an environment with PyTorch and CUDA support.
- Install pytest:

```bash
uv pip install pytest
```

The distributed tests require at least two CUDA devices and must be launched with
`torchrun`.

## CPU-Safe Plumbing Tests

Run these from the repository root:

```bash
python -m pytest tests/test_python_plumbing.py
```

These tests do not initialize CUDA or `torch.distributed`. They use a fake bound
module to verify that the public Python wrappers call the expected C++ binding
methods with the expected pointer, dtype, context, and stream arguments.

## Distributed CUDA Tests

Run these from the repository root on a machine with at least two GPUs:

```bash
PURLIN_CACHE_DIR=/tmp/purlin_jit_test torchrun --standalone --nproc-per-node=2 -m pytest -q tests/test_python_collectives_distributed.py
```

The first run JIT-compiles the generated pybind/CUDA extension, which can take
several minutes. Reusing the same `PURLIN_CACHE_DIR` avoids rebuilding when the
generated source has not changed.

The distributed tests cover:

- `all_gather`
- `all_gather_v`
- `all_reduce`
- `all_to_all`
- `all_to_all_v`
- `reduce_scatter`
- `reduce_scatter_v`

Reduction tests use deterministic rank-ordered expected values instead of NCCL
as the oracle, because NCCL reduction order is not bitwise identical to Purlin's
strict rank order.

## Running Everything

The distributed file skips when it is not launched under `torchrun`, so a normal
pytest run only exercises the CPU-safe tests and reports the distributed tests as
skipped:

```bash
python -m pytest tests
```
