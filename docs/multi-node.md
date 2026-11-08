# Multi-node Ray training

Status: supported by the worker-plan logic, but not covered by automated CI
(there is no multi-node runner).  Verify on your cluster before relying on it.

## Prerequisites

- A Ray cluster with one GPU per worker node.
- The driver/head node must itself see a CUDA (or MPS) accelerator:
  `main()` calls `check_device_compatibility()` before Ray starts and refuses
  to run on a CPU-only driver.
- The same RazorDL version (or the same project checkout) on every node.
- A filesystem visible at the **same absolute paths** on every node for:
  - `data_path`
  - `outputs_dir` (checkpoints and experiment metadata)
  - `resume_from` / `init_from` paths
- NCCL connectivity between nodes (open the ephemeral ports, or set
  `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`).

## Start the Ray cluster

On the head node:

```bash
ray start --head --num-gpus=4
```

On each worker node:

```bash
ray start --address=<head-ip>:6379 --num-gpus=4
```

Check that Ray sees every GPU:

```bash
ray status
```

## Configure RazorDL

Point `ray_kwargs.ray_init.address` at the existing cluster:

```yaml
ray_kwargs:
  ray_init:
    address: auto
```

`main()` asks Ray for `ray.cluster_resources()["GPU"]` and requests one
`TorchTrainer` worker per cluster GPU.  Without this, a multi-node job used to
request only the driver's local GPU count.

## Shared filesystem requirements

`BaseTrainer` writes experiment metadata and checkpoints through global rank 0
only.  Other ranks participate in the collective save and then read from the
same path on resume.  Therefore:

- `outputs_dir` must be on a shared filesystem (NFS/Lustre/GPFS/cloud mount),
  mounted at the same path on every node.
- `data_path` must be readable by every worker at the same absolute path.
  Either use a shared mount or copy the data to the same path on each node.
- `resume_from` and `init_from` must point to paths visible on every node.
- Do not put `outputs_dir` on node-local `/tmp`.

If the filesystem is not shared, rank 0's checkpoint is not visible to the
other nodes on the next run.

## Code availability

Workers must be able to import the same code as the driver:

- Install the same `razordl` package on every node, **or**
- Run from a shared checkout, **or**
- Ship the project via Ray's `runtime_env.working_dir` (larger and slower to
  start, but self-contained).

For custom-mode projects (`src/main.py`), the generated classes must be
importable on workers; a shared checkout is the simplest option.

## Environment variables

Per-node environment variables can be set through Ray:

```yaml
ray_kwargs:
  ray_init:
    runtime_env:
      env_vars:
        NCCL_DEBUG: WARN
        PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
```

`main()` already propagates the driver's `PYTORCH_CUDA_ALLOC_CONF` into Ray's
default runtime env; an explicit value in `ray_kwargs` wins.

## Sequence parallelism across nodes

`sp_size > 1` creates Ulysses process groups over global ranks.  If an SP group
spans nodes, every attention layer's all-to-all crosses the network.  This is
supported by the same code path, but it is much more sensitive to interconnect
bandwidth and NCCL configuration.  Keep `sp_size` within a node when possible.

## Checkpoints and resume

- Full checkpoints are written atomically by global rank 0 after a collective
  FSDP2 save.
- `resume_mode: auto` scans `outputs_dir` and compares the code/config hash;
  the same experiment directory must be visible on the driver and workers.
- `init_from` forks weights into a new experiment directory; the source
  checkpoint must be visible on every node.

## Known limitations

- No automated multi-node CI; this document and the worker-plan unit test are
  the current coverage.
- The driver must have a GPU; CPU-only head nodes are rejected by
  `check_device_compatibility()`.
- `ray.cluster_resources()` reports GPUs reserved by other Ray jobs too; use
  `ray status` and Ray resource isolation if you share a cluster.
- The local `scripts/smoke_fsdp2_sp.py` uses `torch.distributed.run`, not Ray;
  it verifies FSDP2 + SP on one node but not multi-node placement.
