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

All collective entry points return without GPU work for `world == 1`, and
fixed-count collectives also return for `bytes == 0`. Destination buffers remain
untouched. When NVTX telemetry is enabled, these calls still emit their
collective range and byte payload before returning. The `world == 1` guards are
defensive; initialization continues to require `world > 1`. Multi-rank varlen
calls with empty local partitions still participate in the protocol.

Rank zero writes CSV. `totalBytes` follows the payload definition for each
operation; the variable-count definitions are detailed below. There are no
separate local-size or maximum-peer-size columns. Algorithm bandwidth retains
the logical payload definition for each operation. Latency and bandwidth are
limited to four fractional digits.

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

## Variable-count split policy

`testAGV`, `testRSV`, and `testA2AV` interpret their size arguments as **total
bytes**, with a fixed doubling step. AGV's input partitions sum to the total;
RSV's output partitions sum to the total and each rank supplies that full input;
A2AV sends and receives the total on every rank. Their CSV `totalBytes` and
payload-bandwidth numerator use that same total.

The deterministic split policy uses weights `[4, 1, ..., 1]`. Partition the total
in 128-byte units: floor each ideal quota, then assign remaining units by
descending fractional remainder, breaking ties by the lowest index. Splits are
aligned and sum exactly to the requested total. The target ratio is 4:1;
rounding changes the realized ratio, especially at small sizes, which can
produce zero-sized partitions. Totals must be positive multiples of 128 bytes.

AGV and RSV keep the larger partition on rank zero throughout the benchmark.
A2AV rotates the entire rounded vector right by `(source + 1) % world`, putting
the larger split on the next destination rank. Rotating after rounding preserves
both row and column totals even when smaller splits are unequal. Receive splits
are the transpose of the send matrix.

For 8192 bytes and four ranks, AGV/RSV use `[4736, 1152, 1152, 1152]`. A2AV's
send rows are:

```text
[1152, 4736, 1152, 1152]
[1152, 1152, 4736, 1152]
[1152, 1152, 1152, 4736]
[4736, 1152, 1152, 1152]
```

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
