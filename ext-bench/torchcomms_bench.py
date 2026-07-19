#!/usr/bin/env python3
"""CUDA Graph microbenchmarks for the torchcomms NCCLX backend.

Launch with torchrun, for example:

    torchrun --standalone --nproc-per-node=2 tests/torchcomms_bench.py \
        --collective all_reduce --min-bytes 128 --max-bytes 1M \
        --graph-launches 8

The payload definitions and CSV columns mirror the benchmarks in csrc/tests.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torchcomms


COLLECTIVES = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "all_gather_v",
    "reduce_scatter_v",
    "all_to_all_v",
)

ALIASES = {
    "ar": "all_reduce",
    "ag": "all_gather",
    "rs": "reduce_scatter",
    "a2a": "all_to_all",
    "agv": "all_gather_v",
    "rsv": "reduce_scatter_v",
    "a2av": "all_to_all_v",
    **{collective: collective for collective in COLLECTIVES},
}


@dataclass
class Workload:
    operation: Callable[[], None]
    actual: Sequence[torch.Tensor]
    expected: Sequence[torch.Tensor]
    prepare_benchmark: Callable[[], None]
    total_bytes: int
    logical_bytes: int
    datatype: str


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


def all_gather_sizes(size: int, world: int) -> list[int]:
    sizes = [size] * world
    if size >= 128 * world:
        for rank in range(1, world):
            sizes[rank] -= 128
    return sizes


def reduce_scatter_sizes(size: int, world: int) -> list[int]:
    sizes = [size] * world
    if size >= 128 * world:
        for rank in range(1, world):
            sizes[0] -= 128
            sizes[rank] += 128
    return sizes


def all_to_all_splits_for_source(total: int, source: int, world: int) -> list[int]:
    peer_base = (total // world) // 32 * 32
    splits = [peer_base] * world
    splits[-1] += total - peer_base * world
    if peer_base >= 64 * world:
        for destination in range(world - 1):
            delta = -128 if (source + destination) % 2 == 0 else 128
            splits[destination] += delta
            splits[-1] -= delta
    return splits


def all_to_all_receive_splits(total: int, rank: int, world: int) -> list[int]:
    return [
        all_to_all_splits_for_source(total, source, world)[rank]
        for source in range(world)
    ]


def make_workload(
    collective: str,
    size: int,
    comm: torchcomms.TorchComm,
    device: torch.device,
) -> Workload:
    rank = comm.get_rank()
    world = comm.get_size()

    if collective == "all_reduce":
        dtype = torch.bfloat16
        elements = size // torch.empty((), dtype=dtype).element_size()
        tensor = torch.full((elements,), rank + 1, dtype=dtype, device=device)
        expected = torch.full_like(tensor, world * (world + 1) // 2)

        def operation() -> None:
            comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)

        return Workload(
            operation,
            [tensor],
            [expected],
            tensor.zero_,
            size * world,
            size,
            "bfloat16",
        )

    if collective == "all_gather":
        input_tensor = torch.full((size,), rank % 256, dtype=torch.uint8, device=device)
        output = torch.empty(size * world, dtype=torch.uint8, device=device)
        expected = torch.cat(
            [
                torch.full((size,), source % 256, dtype=torch.uint8, device=device)
                for source in range(world)
            ]
        )

        def operation() -> None:
            comm.all_gather_single(output, input_tensor, False)

        return Workload(
            operation,
            [output],
            [expected],
            lambda: None,
            size * world,
            size * world,
            "uint8",
        )

    if collective == "reduce_scatter":
        dtype = torch.bfloat16
        element_size = torch.empty((), dtype=dtype).element_size()
        local_elements = size // element_size
        input_tensor = torch.full(
            (local_elements * world,), rank + 1, dtype=dtype, device=device
        )
        output = torch.empty(local_elements, dtype=dtype, device=device)
        expected = torch.full_like(output, world * (world + 1) // 2)

        def operation() -> None:
            comm.reduce_scatter_single(
                output, input_tensor, torchcomms.ReduceOp.SUM, False
            )

        total = size * world
        return Workload(
            operation,
            [output],
            [expected],
            lambda: None,
            total,
            total,
            "bfloat16",
        )

    if collective == "all_to_all":
        total = size * world
        input_tensor = torch.cat(
            [
                torch.full(
                    (size,),
                    (rank * world + destination) % 256,
                    dtype=torch.uint8,
                    device=device,
                )
                for destination in range(world)
            ]
        )
        output = torch.empty(total, dtype=torch.uint8, device=device)
        expected = torch.cat(
            [
                torch.full(
                    (size,),
                    (source * world + rank) % 256,
                    dtype=torch.uint8,
                    device=device,
                )
                for source in range(world)
            ]
        )

        def operation() -> None:
            comm.all_to_all_single(output, input_tensor, False)

        return Workload(
            operation,
            [output],
            [expected],
            lambda: None,
            total,
            total,
            "uint8",
        )

    if collective == "all_gather_v":
        sizes = all_gather_sizes(size, world)
        input_tensor = torch.full(
            (sizes[rank],), rank % 256, dtype=torch.uint8, device=device
        )
        outputs = [
            torch.empty(peer_size, dtype=torch.uint8, device=device)
            for peer_size in sizes
        ]
        expected = [
            torch.full((peer_size,), source % 256, dtype=torch.uint8, device=device)
            for source, peer_size in enumerate(sizes)
        ]

        def operation() -> None:
            comm.all_gather_v(outputs, input_tensor, False)

        total = sum(sizes)
        return Workload(
            operation,
            outputs,
            expected,
            lambda: None,
            total,
            total,
            "uint8",
        )

    if collective == "reduce_scatter_v":
        dtype = torch.bfloat16
        element_size = torch.empty((), dtype=dtype).element_size()
        sizes = reduce_scatter_sizes(size, world)
        element_counts = [peer_size // element_size for peer_size in sizes]
        inputs = [
            torch.full((count,), rank + 1, dtype=dtype, device=device)
            for count in element_counts
        ]
        output = torch.empty(element_counts[rank], dtype=dtype, device=device)
        expected = torch.full_like(output, world * (world + 1) // 2)

        def operation() -> None:
            comm.reduce_scatter_v(output, inputs, torchcomms.ReduceOp.SUM, False)

        total = sum(sizes)
        return Workload(
            operation,
            [output],
            [expected],
            lambda: None,
            total,
            total,
            "bfloat16",
        )

    send_splits = all_to_all_splits_for_source(size, rank, world)
    receive_splits = all_to_all_receive_splits(size, rank, world)
    input_tensor = torch.cat(
        [
            torch.full((count,), rank % 256, dtype=torch.uint8, device=device)
            for count in send_splits
        ]
    )
    output = torch.empty(sum(receive_splits), dtype=torch.uint8, device=device)
    expected = torch.cat(
        [
            torch.full((count,), source % 256, dtype=torch.uint8, device=device)
            for source, count in enumerate(receive_splits)
        ]
    )

    def operation() -> None:
        comm.all_to_all_v_single(
            output, input_tensor, receive_splits, send_splits, False
        )

    return Workload(
        operation,
        [output],
        [expected],
        lambda: None,
        sum(send_splits),
        sum(receive_splits),
        "uint8",
    )


def rank_max(value: float, comm: torchcomms.TorchComm, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    comm.all_reduce(tensor, torchcomms.ReduceOp.MAX, False)
    return tensor.item()


def correctness_error(
    workload: Workload,
    comm: torchcomms.TorchComm,
    device: torch.device,
) -> float:
    workload.operation()
    mismatches = sum(
        torch.count_nonzero(actual != expected).item()
        for actual, expected in zip(workload.actual, workload.expected)
    )
    checked = sum(tensor.numel() for tensor in workload.actual)
    local_error = 100.0 * mismatches / checked if checked else 0.0
    return rank_max(local_error, comm, device)


def measure_graph(
    workload: Workload,
    comm: torchcomms.TorchComm,
    device: torch.device,
    stream: torch.cuda.Stream,
    runs: int,
    graph_launches: int,
) -> float:
    workload.prepare_benchmark()
    comm.barrier(False)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(runs):
            workload.operation()

    graph.replay()
    stream.synchronize()
    comm.barrier(False)
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(graph_launches):
        graph.replay()
    stop.record(stream)
    stop.synchronize()

    milliseconds = start.elapsed_time(stop) / (runs * graph_launches)
    maximum = rank_max(milliseconds, comm, device)
    graph.reset()
    return maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collective",
        required=True,
        choices=(*sorted(ALIASES), "all"),
        help="collective to benchmark; short csrc names such as ar and agv work too",
    )
    parser.add_argument("--min-bytes", type=parse_size, default=128)
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
        help="collective invocations captured in each graph (default: 128)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    comm = torchcomms.new_comm("ncclx", device, name="torchcomms_bench")
    try:
        rank = comm.get_rank()
        world = comm.get_size()
        stream = torch.cuda.Stream(device=device)
        collectives = (
            COLLECTIVES if args.collective == "all" else (ALIASES[args.collective],)
        )

        header_written = False

        with torch.cuda.stream(stream):
            for collective in collectives:
                for size in size_sweep(args.min_bytes, args.max_bytes):
                    workload = make_workload(collective, size, comm, device)
                    error = correctness_error(workload, comm, device)
                    milliseconds = measure_graph(
                        workload,
                        comm,
                        device,
                        stream,
                        args.runs,
                        args.graph_launches,
                    )
                    if rank == 0:
                        bandwidth = workload.logical_bytes / 1e9 / (milliseconds * 1e-3)
                        if not header_written:
                            write_csv_row(
                                (
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
                            )
                            header_written = True
                        write_csv_row(
                            (
                                collective,
                                world,
                                workload.total_bytes,
                                workload.datatype,
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
    finally:
        comm.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
