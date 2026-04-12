# P2P Sweep

Run the full sweep with:

```bash
python3 scripts/p2p_sweep.py \
    --rerun-failed \
    --run-timeout-seconds 900 \
    --graph-launches 2 \
    --runs 64 \
    --warmup 64
```

Useful options:

```bash
# Show the planned tuples without building or running
python3 scripts/p2p_sweep.py --dry-run

# Resume only failed tuples from a prior run
python3 scripts/p2p_sweep.py --rerun-failed

# Restrict the sweep to a subset
python3 scripts/p2p_sweep.py \
  --threads 64,96 \
  --pipe-stages 1,2 \
  --elements-per-thread 4,8 \
  --max-num-ctas 2,4

# Override the launcher if your environment needs a different MPI command
python3 scripts/p2p_sweep.py --launcher "mpirun -np 2"

# Override or clear the NVSHMEM bootstrap mode
python3 scripts/p2p_sweep.py --nvshmem-bootstrap PMI
python3 scripts/p2p_sweep.py --nvshmem-bootstrap ""
```

Artifacts are written under `results/p2p_sweep/` by default:

- `results.json`: machine-readable per-configuration results
- `ranked_summary.txt`: success-first ranking by peak GB/s
- `logs/<tuple>/`: configure, build, and run logs for each tuple

The sweep is resumable. Completed successful tuples are skipped on later runs
unless `--force` is used. Failed tuples are retried only with
`--rerun-failed` or `--force`.

By default the script exports `NVSHMEM_BOOTSTRAP=MPI` for benchmark runs,
which is required in this environment for `mpirun -np 2` to form a two-rank
NVSHMEM world.
