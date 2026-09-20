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
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release -Wno-dev
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

Rank zero writes CSV. `totalBytes` follows the payload definition for each
operation; the variable-count definitions are detailed below. There are no
separate local-size or maximum-peer-size columns. Algorithm bandwidth retains
the logical payload definition for each operation. Latency and bandwidth are
limited to four fractional digits.

Every size has an untimed correctness run built entirely from deterministic
seeded random streams:

- Byte collectives fill each contribution from a stream keyed on the shared
  data seed and the source rank (gather) or the (source, destination) pair
  (all-to-all). Each rank replays the streams it expects to receive locally to
  materialize the full reference buffer.
- Reduction tests fill randomized floating-point inputs the same way; their
  reference kernel accumulates replayed sources in Purlin's guaranteed
  `0 -> 1 -> ... -> world - 1` order.

MatX performs the final buffer comparison in every case.

## Variable-count split policy

`testAGV`, `testRSV`, and `testA2AV` interpret their size arguments as **total
bytes**, with a fixed doubling step. AGV's input partitions sum to the total;
RSV's output partitions sum to the total and each rank supplies that full input;
A2AV sends and receives the total on every rank. Their CSV `totalBytes` and
payload-bandwidth numerator use that same total.

The deterministic split policy is Zipf with exponent 0.125: rank `r` receives a
share proportional to `1 / (r + 1)^0.125`. Shares are apportioned in whole
**units**. The unit is the largest power of two that divides the total and still
leaves eight units per rank, never below 128 bytes, so an eight-rank total is
split into 64 units as `9:9:8:8:8:8:7:7`, four ranks into 32 as `9:8:8:7`, and two
ranks into 16 as `8:8`. Floor each ideal quota, then assign remaining units by
descending fractional remainder, breaking ties by the lowest index. Splits sum
exactly to the requested total, and every size and offset is a multiple of the
unit. The unit reaches
64 KiB, one full 16-stage copy pipeline that stays whole 4 KiB stages after a
partition is divided among sixteen blocks, from `world * 512 KiB` total. Totals
below `world * 1024` bytes have fewer than eight units per rank, which coarsens
the shape and, with the steeper exponents, can produce zero-sized partitions. Totals must be positive
multiples of 128 bytes and the world size at most 32.

`BENCH_ZIPF_EXPONENT` selects another skew: `0.25` (eight ranks split as
`11:9:8:8:7:7:7:7`, largest 1.4x the mean), `0.5` (`15:10:8:7:7:6:6:5`, 1.9x),
`1` (`23:12:8:6:5:4:3:3`, 2.9x) or `0` (equal shares); the default `0.125` gives
1.1x. Only these values are supported: exponent 1 uses exact integer weights
`lcm(1..world) / (r + 1)` and the others need just square roots and divisions, so
every implementation derives identical splits.

AGV and RSV keep the largest partition on rank zero throughout the benchmark.
A2AV rotates the entire rounded vector right by `(source + 1) % world`, putting
the largest split on the next destination rank. Rotating after rounding preserves
both row and column totals. Receive splits are the transpose of the send matrix.

For 8192 bytes and four ranks the unit is 256 bytes and AGV/RSV use
`[2304, 2048, 2048, 1792]`. A2AV's send rows are:

```text
[1792, 2304, 2048, 2048]
[2048, 1792, 2304, 2048]
[2048, 2048, 1792, 2304]
[2304, 2048, 2048, 1792]
```

For 512 KiB and eight ranks the unit is 8 KiB and AGV/RSV use
`[72K, 72K, 64K, 64K, 64K, 64K, 56K, 56K]`.

The host-only `testVariableCounts` target checks examples, rounding, alignment,
exact totals, send/receive consistency, invalid arguments, and overflow
boundaries. After configuring from `csrc` as above:

```sh
cmake --build cmake-build-release --target testVariableCounts
./cmake-build-release/testVariableCounts
```

The fixed-count and separate `*SKEW` benchmark policies are unchanged.

## Empty-partition varlen regression

Varlen collectives announce invocation entry through `varLenSignals` before
data movement and defer the arrival wait until block zero returns. The signal
uses relaxed system-scope packet stores and reads. Ordered kernel completion
then prevents any rank from reusing a staging or signal slot while a peer still
uses its previous contents, without a grid-wide barrier. A2AV's existing
deferred extent exchange supplies this same arrival wait and retains its
payload-based global epoch advance.

`testVarlenEmpty` delays each possible nonempty owner and checks every result
in repeated stream batches and CUDA graph replays. Input values change on each
invocation, including graph replays, to detect stale or future packets as well
as hangs. AGV and RSV put the entire partition on one rank. A2AV tests one-way
traffic, self-only traffic with fully idle peers, and a ring with unequal
extents. Empty buffers use null pointers.

Cases use 128-byte, 8-KiB, 1-MiB and 16-MiB transfers to exercise packet,
nonchunked and chunked paths. A mixed sequence reuses the same context across
all collective kinds and sizes without synchronization between invocations,
also checking signal reuse across grid changes and multi-epoch advances. Each
individual case runs 128 calls per mode; the mixed sequence runs 640 per mode.
These deliberately sparse layouts are regression inputs, not the benchmark
split policy above.

Run from `csrc` after a Release configuration:

```sh
cmake --build cmake-build-release --target testVarlenEmpty
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  timeout 180s mpirun -n 2 ./cmake-build-release/testVarlenEmpty
```
