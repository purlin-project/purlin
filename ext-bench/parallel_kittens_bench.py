#!/usr/bin/env python3
"""CUDA Graph microbenchmarks for ParallelKittens collectives.

ParallelKittens is distributed as CUDA source rather than as an installable
Python package.  Point this script at a ThunderKittens checkout and launch it
with one process per local GPU, for example:

    git clone https://github.com/HazyResearch/ThunderKittens.git
    THUNDERKITTENS_ROOT=$PWD/ThunderKittens \
      torchrun --standalone --nproc-per-node=8 \
      ext-bench/parallel_kittens_bench.py \
      --collective all_reduce --min-bytes 32K --max-bytes 128M \
      --graph-launches 8

The extension is compiled for SM90a and cached by PyTorch.  The benchmark uses
ordinary PyTorch tensors at its API boundary.  Every timed invocation includes
the copy/layout conversion into TKParallelTensor, the ParallelKittens
collective, and the copy/layout conversion back to the ordinary output tensor.

The byte-size argument has the same meaning as the csrc benchmarks: local bytes
for all-reduce/all-gather/reduce-scatter, and bytes per peer for all-to-all.
The current kernels impose these BF16 byte alignments:

    all_reduce:     world_size * 1024
    all_gather:     32768
    reduce_scatter: 512
    all_to_all:     4096
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist


COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
ALIASES = {
    "ar": "all_reduce",
    "ag": "all_gather",
    "rs": "reduce_scatter",
    "a2a": "all_to_all",
    **{collective: collective for collective in COLLECTIVES},
}

HEADER = (
    "collective",
    "world",
    "totalBytes",
    "datatype",
    "lat(us)",
    "bw(GB/s)",
    "error(%)",
    "GPUName",
    "warmup",
    "runs",
    "graph_launches",
)


@dataclass
class Workload:
    """A conventional-buffer adapter around one ParallelKittens collective."""

    pack_input: Callable[[], None]
    launch_collective: Callable[[], None]
    unpack_output: Callable[[], None]
    normal_output: torch.Tensor
    expected: torch.Tensor | None
    special_tensors: tuple[object, ...]
    total_bytes: int
    logical_bytes: int

    def run(self) -> None:
        # This ordering is the interoperability contract being benchmarked.
        self.pack_input()
        self.launch_collective()
        self.unpack_output()


def write_csv_row(row: Sequence[object]) -> None:
    fields = []
    for value in row:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="").writerow((value,))
        fields.append(buffer.getvalue())
    sys.stdout.write(", ".join(fields) + "\n")


def parse_size(text: str) -> int:
    suffix = text[-1].upper()
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(suffix, 1)
    number = text[:-1] if multiplier != 1 else text
    return int(float(number) * multiplier)


def size_sweep(min_bytes: int, max_bytes: int):
    size = min_bytes
    while True:
        yield size
        if size == max_bytes:
            return
        size *= 2


def thunderkittens_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.resolve()

    for name in ("THUNDERKITTENS_ROOT", "PARALLEL_KITTENS_ROOT"):
        if value := os.environ.get(name):
            return Path(value).resolve()

    repository = Path(__file__).resolve().parents[1]
    candidates = (
        repository / "third_party" / "ThunderKittens",
        repository.parent / "ThunderKittens",
        Path.cwd() / "ThunderKittens",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "ThunderKittens checkout not found; pass --thunderkittens-root or set "
        "THUNDERKITTENS_ROOT"
    )


def parallel_kittens_source(root: Path, collective: str, world: int) -> str:
    source_path = root / "kernels" / "parallel" / collective / f"{collective}.cu"
    source = source_path.read_text()

    # The upstream kernels are templates with NUM_DEVICES fixed to 8.  Build a
    # module specialized to the local torchrun world without changing the kernel.
    source, replacements = re.subn(
        r"static constexpr int NUM_DEVICES = 8;",
        f"static constexpr int NUM_DEVICES = {world};",
        source,
    )
    if replacements != 2:
        raise RuntimeError(
            f"unexpected ParallelKittens source layout in {source_path}: "
            f"found {replacements} NUM_DEVICES declarations"
        )

    source, replacements = re.subn(
        r"PYBIND11_MODULE\(_C,\s*m\)",
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)",
        source,
    )
    if replacements != 1:
        raise RuntimeError(f"could not find the pybind module in {source_path}")
    return source


def load_parallel_kittens(
    root: Path,
    collective: str,
    world: int,
    verbose: bool,
) -> ModuleType:
    from torch.utils.cpp_extension import load_inline

    # SM90a is required for Hopper multimem and multicast instructions.  Setting
    # this explicitly also permits building on a host whose visible GPUs are not
    # Hopper, as is useful for --build-only validation.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    source = parallel_kittens_source(root, collective, world)
    name = f"parallel_kittens_{collective}_sm90_w{world}"

    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=source,
        functions=None,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-forward-unknown-to-host-compiler",
            "-Xcompiler=-Wno-psabi",
            "-Xcompiler=-fno-strict-aliasing",
            "-diag-suppress=3189",
            "-DNDEBUG",
            "-DKITTENS_SM90",
            "-lineinfo",
            "-ftemplate-backtrace-limit=0",
        ],
        extra_ldflags=["-lcuda", "-lrt", "-lpthread", "-ldl"],
        extra_include_paths=[str(root / "include"), str(root / "prototype")],
        with_cuda=True,
        verbose=verbose,
        no_implicit_headers=True,
    )


def tk_tensor(
    module: ModuleType,
    shape: Sequence[int],
    dtype: torch.dtype,
    rank: int,
    world: int,
    multicast: bool,
):
    return module.TKParallelTensor(tuple(shape), dtype, rank, world, multicast)


def make_barrier(module: ModuleType, rank: int, world: int):
    barrier = tk_tensor(module, (1, 1), torch.int32, rank, world, True)
    barrier.data_.zero_()
    return barrier


def make_all_reduce(
    size: int,
    module: ModuleType,
    rank: int,
    world: int,
    device: torch.device,
) -> Workload:
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    elements = size // element_size
    alignment = world * 512
    if elements % alignment:
        raise ValueError(
            f"all_reduce size must be a multiple of {alignment * element_size} bytes"
        )

    normal_input = torch.full(
        (elements,), rank + 1, dtype=torch.bfloat16, device=device
    )
    normal_output = torch.empty_like(normal_input)
    expected = torch.full_like(normal_output, world * (world + 1) // 2)
    tensor = tk_tensor(module, (elements,), torch.bfloat16, rank, world, True)
    barrier = make_barrier(module, rank, world)

    return Workload(
        pack_input=lambda: tensor.data_.copy_(normal_input),
        launch_collective=lambda: module.tk_all_reduce(tensor, barrier),
        unpack_output=lambda: normal_output.copy_(tensor.data_),
        normal_output=normal_output,
        expected=expected,
        special_tensors=(tensor, barrier),
        total_bytes=size * world,
        logical_bytes=size,
    )


def make_all_gather(
    size: int,
    module: ModuleType,
    rank: int,
    world: int,
    device: torch.device,
) -> Workload:
    rows = 128
    tile_columns = 128
    elements = size // 2
    if elements % (rows * tile_columns):
        raise ValueError("all_gather size must be a multiple of 32768 bytes")
    columns = elements // rows

    normal_input = torch.full(
        (rows, columns), rank + 1, dtype=torch.bfloat16, device=device
    )
    normal_output = torch.empty(
        (world, rows, columns), dtype=torch.bfloat16, device=device
    )
    expected = torch.empty_like(normal_output)
    for source in range(world):
        expected[source].fill_(source + 1)

    input_tensor = tk_tensor(
        module, (rows, columns), torch.bfloat16, rank, world, False
    )
    output_tensor = tk_tensor(
        module, (rows, columns * world), torch.bfloat16, rank, world, True
    )
    barrier = make_barrier(module, rank, world)

    def unpack_output() -> None:
        gathered = output_tensor.data_.view(rows, world, columns).permute(1, 0, 2)
        normal_output.copy_(gathered)

    return Workload(
        pack_input=lambda: input_tensor.data_.copy_(normal_input),
        launch_collective=lambda: module.tk_all_gather(
            output_tensor, input_tensor, barrier
        ),
        unpack_output=unpack_output,
        normal_output=normal_output,
        expected=expected,
        special_tensors=(input_tensor, output_tensor, barrier),
        total_bytes=size * world,
        logical_bytes=size * world,
    )


def make_reduce_scatter(
    size: int,
    module: ModuleType,
    rank: int,
    world: int,
    device: torch.device,
) -> Workload:
    elements = size // 2
    if elements % 256:
        raise ValueError("reduce_scatter size must be a multiple of 512 bytes")

    normal_input = torch.full(
        (world, elements), rank + 1, dtype=torch.bfloat16, device=device
    )
    normal_output = torch.empty(elements, dtype=torch.bfloat16, device=device)
    expected = torch.full_like(normal_output, world * (world + 1) // 2)
    input_tensor = tk_tensor(
        module, (1, elements * world), torch.bfloat16, rank, world, True
    )
    output_tensor = tk_tensor(module, (1, elements), torch.bfloat16, rank, world, False)
    barrier = make_barrier(module, rank, world)

    return Workload(
        pack_input=lambda: input_tensor.data_.copy_(normal_input.view(1, -1)),
        launch_collective=lambda: module.tk_reduce_scatter(
            output_tensor, input_tensor, barrier
        ),
        unpack_output=lambda: normal_output.copy_(output_tensor.data_.view(-1)),
        normal_output=normal_output,
        expected=expected,
        special_tensors=(input_tensor, output_tensor, barrier),
        total_bytes=size * world,
        logical_bytes=size * world,
    )


def make_all_to_all(
    size: int,
    module: ModuleType,
    rank: int,
    world: int,
    device: torch.device,
) -> Workload:
    rows_per_peer = 16
    columns = 128
    elements_per_depth = rows_per_peer * columns
    peer_elements = size // 2
    if peer_elements % elements_per_depth:
        raise ValueError("all_to_all size must be a multiple of 4096 bytes")
    depth = peer_elements // elements_per_depth

    normal_input = torch.empty(
        (world, depth, rows_per_peer, columns),
        dtype=torch.bfloat16,
        device=device,
    )
    normal_output = torch.empty_like(normal_input)
    expected = torch.empty_like(normal_output)
    for destination in range(world):
        normal_input[destination].fill_(rank * world + destination)
    for source in range(world):
        expected[source].fill_(source * world + rank)

    input_tensor = tk_tensor(
        module,
        (1, depth, rows_per_peer * world, columns),
        torch.bfloat16,
        rank,
        world,
        False,
    )
    output_tensor = tk_tensor(
        module,
        (1, depth * world, rows_per_peer, columns),
        torch.bfloat16,
        rank,
        world,
        False,
    )
    barrier = make_barrier(module, rank, world)

    def pack_input() -> None:
        parallel_layout = input_tensor.data_.view(depth, world, rows_per_peer, columns)
        parallel_layout.copy_(normal_input.permute(1, 0, 2, 3))

    return Workload(
        pack_input=pack_input,
        launch_collective=lambda: module.tk_all_to_all(
            output_tensor, input_tensor, barrier, 2, 1
        ),
        unpack_output=lambda: normal_output.copy_(
            output_tensor.data_.view(world, depth, rows_per_peer, columns)
        ),
        normal_output=normal_output,
        expected=expected,
        special_tensors=(input_tensor, output_tensor, barrier),
        total_bytes=size * world,
        logical_bytes=size * world,
    )


WORKLOAD_FACTORIES = {
    "all_reduce": make_all_reduce,
    "all_gather": make_all_gather,
    "reduce_scatter": make_reduce_scatter,
    "all_to_all": make_all_to_all,
}


def rank_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def correctness_error(workload: Workload, device: torch.device) -> float:
    workload.run()
    assert workload.expected is not None
    mismatches = torch.count_nonzero(workload.normal_output != workload.expected).item()
    checked = workload.normal_output.numel()
    local_error = 100.0 * mismatches / checked if checked else 0.0
    return rank_max(local_error, device)


def measure_graph(
    workload: Workload,
    device: torch.device,
    stream: torch.cuda.Stream,
    runs: int,
    graph_launches: int,
) -> float:
    dist.barrier()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(runs):
            workload.run()

    graph.replay()
    stream.synchronize()
    dist.barrier()
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(graph_launches):
        graph.replay()
    stop.record(stream)
    stop.synchronize()

    milliseconds = start.elapsed_time(stop) / (runs * graph_launches)
    maximum = rank_max(milliseconds, device)
    graph.reset()
    return maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collective",
        required=True,
        choices=sorted(ALIASES),
        help="one of all_reduce, all_gather, reduce_scatter, or all_to_all",
    )
    parser.add_argument("--min-bytes", type=parse_size, default=32 * 1024)
    parser.add_argument("--max-bytes", type=parse_size, default=128 * 1024**2)
    parser.add_argument(
        "--graph-launches",
        type=int,
        default=8,
        help="number of graph replays to time",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=128,
        help="adapter invocations captured in each graph (default: 128)",
    )
    parser.add_argument(
        "--thunderkittens-root",
        type=Path,
        help="path to a ThunderKittens checkout (or set THUNDERKITTENS_ROOT)",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="compile/import the SM90 extension without initializing GPUs",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="template world size for --build-only (default: WORLD_SIZE or 8)",
    )
    parser.add_argument(
        "--verbose-build", action="store_true", help="show extension build commands"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collective = ALIASES[args.collective]
    root = thunderkittens_root(args.thunderkittens_root)
    world = args.world_size or int(os.environ.get("WORLD_SIZE", 8))
    module = load_parallel_kittens(root, collective, world, args.verbose_build)

    if args.build_only:
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"built {module.__name__} for SM90a and world_size={world}")
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    if local_world != world or rank != local_rank:
        raise RuntimeError("ParallelKittens supports single-node torchrun jobs only")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    try:
        stream = torch.cuda.Stream(device=device)
        header_written = False

        with torch.cuda.stream(stream):
            for size in size_sweep(args.min_bytes, args.max_bytes):
                workload = WORKLOAD_FACTORIES[collective](
                    size, module, rank, world, device
                )
                dist.barrier()
                error = correctness_error(workload, device)

                # Correctness is outside the timed region; release its potentially
                # large reference before graph capture.
                workload.expected = None
                gc.collect()

                milliseconds = measure_graph(
                    workload,
                    device,
                    stream,
                    args.runs,
                    args.graph_launches,
                )

                if rank == 0:
                    bandwidth = workload.logical_bytes / 1e9 / (milliseconds * 1e-3)
                    # Keep build/allocation logs above the CSV rather than allowing
                    # them to separate the header from its first result row.
                    if not header_written:
                        write_csv_row(HEADER)
                        header_written = True
                    write_csv_row(
                        (
                            collective,
                            world,
                            workload.total_bytes,
                            "bfloat16",
                            f"{milliseconds * 1000.0:.4f}",
                            f"{bandwidth:.4f}",
                            f"{error:.9g}",
                            torch.cuda.get_device_name(device),
                            args.runs,
                            args.runs,
                            args.graph_launches,
                        )
                    )
                    sys.stdout.flush()

                stream.synchronize()
                dist.barrier()
                del workload
                gc.collect()
                dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
