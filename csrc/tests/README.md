# Collective performance benchmarks

This directory contains the Purlin benchmark suite. The small files under
`tests/` describe only collective-specific setup, invocation, and validation.
Tests and examples share the reusable facilities under
`support/purlin/benchmark/`:

- command-line parsing, size sweeps, stream timing, and CUDA Graph timing;
- deterministic seeded random data and reference generation;
- stream-ordered RAII device buffers, runtime ownership, and error checking;
- variable-count layouts; and
- consistent CSV reporting.

## Targets

| Target | Operation | Correctness reference |
| --- | --- | --- |
| `testAG` | AllGather | local replay of peer streams |
| `testAR` | AllReduce | ordered reduction kernel + MatX |
| `testRS` | ReduceScatter | ordered reduction kernel + MatX |
| `testA2A` | AllToAll | local replay of pair streams |
| `testAGV` | AllGatherV | local replay of peer streams |
| `testA2AV` | AllToAllV | local replay of pair streams |
| `testRSV` | ReduceScatterV | ordered reduction kernel + MatX |
| `testAGVSKEW` | AllGatherV, skewed sizes | local replay of peer streams |
| `testA2AVSKEW` | AllToAllV, skewed splits | local replay of pair streams |
| `testRSVSKEW` | ReduceScatterV, skewed shards | ordered reduction kernel + MatX |

## Build and run

Configure and build the suite in Release mode:

```sh
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target \
  testAG testAGV testA2A testA2AV testAR testRS testRSV \
  testAGVSKEW testA2AVSKEW testRSVSKEW
```

Each executable accepts the same positional arguments:

```text
<program> [minBytes] [maxBytes] [graphLaunches] [runs] [warmup] [seed]
```

Sizes may use `K`, `M`, or `G` suffixes and the endpoints must be powers of two.
The optional `seed` pins the data seed; `0` (the default) draws a random one.
Every run prints the resolved data seed on stderr, so any run can be replayed
bit-for-bit by passing that seed back.

Purlin targets use NVSHMEM's MPI bootstrap. For example, a short two-GPU
stream-mode run is:

```sh
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  mpirun -n 2 ./cmake-build-release/testAR 128 1M 0 20 5
```

To use CUDA Graphs, set `graphLaunches` above zero:

```sh
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  mpirun -n 2 ./cmake-build-release/testAR 128 1M 4 20 5
```

In graph mode, `runs` collective invocations are captured into one graph. One
graph launch is used for warmup, then `graphLaunches` graph launches are timed.
Reported latency is divided by `runs * graphLaunches`. The `warmup` argument is
used only in stream mode.

## Output and correctness

Rank zero writes CSV. `totalBytes` is the aggregate payload across the world;
there are no separate local-size or maximum-peer-size columns. Algorithm
bandwidth retains the logical payload definition for each operation. Latency
and bandwidth are limited to four fractional digits.

Every size has an untimed correctness run built entirely from deterministic
seeded random streams — no second communication library is involved:

- Byte collectives fill each contribution from a stream keyed on the shared
  data seed and the source rank (gather) or the (source, destination) pair
  (all-to-all). Each rank replays the streams it expects to receive locally to
  materialize the full reference buffer.
- Reduction tests fill randomized floating-point inputs the same way; their
  reference kernel accumulates replayed sources in Purlin's guaranteed
  `0 -> 1 -> ... -> world - 1` order.

MatX performs the final buffer comparison in every case. Each reduction TU
selects its element type through its global `DataType` alias. The `error(%)`
column is the maximum mismatch percentage observed by any rank.
