import os

import pytest
import torch
import torch.distributed as dist

import purlin


pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ or "WORLD_SIZE" not in os.environ,
    reason="run with torchrun to exercise distributed CUDA collectives",
)


def _round_down(value, alignment):
    return value - (value % alignment)


def _make_splits_for_source(base_bytes, src, world):
    peer_base = _round_down(base_bytes // world, 32)
    splits = [peer_base for _ in range(world)]
    splits[-1] += base_bytes - peer_base * world
    if peer_base >= 64 * world:
        for dst in range(world - 1):
            delta = -128 if (src + dst) % 2 == 0 else 128
            splits[dst] += delta
            splits[-1] -= delta
    return splits


def _make_in_splits(base_bytes, rank, world):
    return _make_splits_for_source(base_bytes, rank, world)


def _make_out_splits(base_bytes, rank, world):
    return [_make_splits_for_source(base_bytes, peer, world)[rank] for peer in range(world)]


def _offsets(sizes):
    offsets = []
    current = 0
    for size in sizes:
        offsets.append(current)
        current += size
    return offsets


def _filled_segments(rank, sizes, dtype, device):
    elems_per_segment = [size // torch.empty((), dtype=dtype).element_size() for size in sizes]
    return torch.cat(
        [
            torch.full((elems,), rank * 100 + peer, dtype=dtype, device=device)
            for peer, elems in enumerate(elems_per_segment)
        ]
    )


@pytest.fixture(scope="session")
def runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            device_id=device,
        )

    stream = torch.cuda.current_stream(device)
    major, minor = torch.cuda.get_device_capability(device)
    arch = major * 10 + minor
    handle = purlin.initialize(dist.group.WORLD, device, arch, stream.cuda_stream)

    yield {
        "device": device,
        "stream": stream,
        "stream_ptr": stream.cuda_stream,
        "handle": handle,
        "rank": dist.get_rank(),
        "world": dist.get_world_size(),
    }

    purlin.finalize(handle, stream.cuda_stream)
    stream.synchronize()
    dist.destroy_process_group()


def test_all_gather(runtime):
    rank = runtime["rank"]
    world = runtime["world"]
    device = runtime["device"]
    stream = runtime["stream"]

    local_elems = 64
    src = torch.arange(local_elems, dtype=torch.float32, device=device) + rank * 1000
    dst = torch.empty(world * local_elems, dtype=torch.float32, device=device)

    purlin.all_gather(src, dst, runtime["handle"], runtime["stream_ptr"])
    stream.synchronize()

    expected = torch.cat(
        [
            torch.arange(local_elems, dtype=torch.float32, device=device) + peer * 1000
            for peer in range(world)
        ]
    )
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


def test_all_gather_v(runtime):
    rank = runtime["rank"]
    world = runtime["world"]
    device = runtime["device"]
    stream = runtime["stream"]

    dtype = torch.float32
    elem_size = torch.empty((), dtype=dtype).element_size()
    sizes = [256 * (peer + 1) for peer in range(world)]
    local_elems = sizes[rank] // elem_size
    total_elems = sum(sizes) // elem_size
    src = torch.arange(local_elems, dtype=dtype, device=device) + rank * 1000
    dst = torch.empty(total_elems, dtype=dtype, device=device)

    purlin.all_gather_v(src, dst, sizes, runtime["handle"], runtime["stream_ptr"])
    stream.synchronize()

    expected = torch.cat(
        [
            torch.arange(sizes[peer] // elem_size, dtype=dtype, device=device) + peer * 1000
            for peer in range(world)
        ]
    )
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction_mode", list(purlin.ReductionMode))
def test_all_reduce(runtime, dtype, reduction_mode):
    rank = runtime["rank"]
    world = runtime["world"]
    device = runtime["device"]
    stream = runtime["stream"]

    src = torch.full((128,), rank + 1, dtype=dtype, device=device)
    dst = torch.empty_like(src)

    purlin.all_reduce(src, dst, runtime["handle"], runtime["stream_ptr"],
                          reduction_mode=reduction_mode)
    stream.synchronize()

    expected_value = sum(range(1, world + 1))
    expected = torch.full_like(dst, expected_value)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


def test_all_to_all(runtime):
    rank = runtime["rank"]
    world = runtime["world"]
    device = runtime["device"]
    stream = runtime["stream"]

    dtype = torch.float32
    local_elems = 64
    src = torch.cat(
        [
            torch.full((local_elems,), rank * 100 + peer, dtype=dtype, device=device)
            for peer in range(world)
        ]
    )
    dst = torch.empty_like(src)

    purlin.all_to_all(src, dst, runtime["handle"], runtime["stream_ptr"])
    stream.synchronize()

    expected = torch.cat(
        [
            torch.full((local_elems,), peer * 100 + rank, dtype=dtype, device=device)
            for peer in range(world)
        ]
    )
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


def test_all_to_all_v(runtime):
    rank = runtime["rank"]
    world = runtime["world"]
    device = runtime["device"]
    stream = runtime["stream"]

    dtype = torch.float32
    elem_size = torch.empty((), dtype=dtype).element_size()
    base_bytes = 2048
    in_splits = _make_in_splits(base_bytes, rank, world)
    out_splits = _make_out_splits(base_bytes, rank, world)
    src = _filled_segments(rank, in_splits, dtype, device)
    dst = torch.empty(sum(out_splits) // elem_size, dtype=dtype, device=device)

    purlin.all_to_all_v(src, dst, in_splits + out_splits, runtime["handle"], runtime["stream_ptr"])
    stream.synchronize()

    expected = torch.cat(
        [
            torch.full(
                (out_splits[peer] // elem_size,),
                peer * 100 + rank,
                dtype=dtype,
                device=device,
            )
            for peer in range(world)
        ]
    )
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduction_mode", list(purlin.ReductionMode))
def test_reduce_scatter(runtime, reduction_mode):
    rank = runtime["rank"]
    world = runtime["world"]
    device = runtime["device"]
    stream = runtime["stream"]

    dtype = torch.float32
    local_elems = 64
    src = torch.cat(
        [
            torch.full((local_elems,), rank * 100 + peer, dtype=dtype, device=device)
            for peer in range(world)
        ]
    )
    dst = torch.empty(local_elems, dtype=dtype, device=device)

    purlin.reduce_scatter(src, dst, runtime["handle"], runtime["stream_ptr"],
                          reduction_mode=reduction_mode)
    stream.synchronize()

    expected_value = 100 * sum(range(world)) + world * rank
    expected = torch.full_like(dst, expected_value)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduction_mode", list(purlin.ReductionMode))
def test_reduce_scatter_v(runtime, reduction_mode):
    rank = runtime["rank"]
    world = runtime["world"]
    device = runtime["device"]
    stream = runtime["stream"]

    dtype = torch.float32
    elem_size = torch.empty((), dtype=dtype).element_size()
    sizes = [256 * (peer + 1) for peer in range(world)]
    src = _filled_segments(rank, sizes, dtype, device)
    dst = torch.empty(sizes[rank] // elem_size, dtype=dtype, device=device)

    purlin.reduce_scatter_v(src, dst, sizes, runtime["handle"], runtime["stream_ptr"],
                          reduction_mode=reduction_mode)
    stream.synchronize()

    expected_value = 100 * sum(range(world)) + world * rank
    expected = torch.full_like(dst, expected_value)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
