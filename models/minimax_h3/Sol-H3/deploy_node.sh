#!/usr/bin/env bash
# Generic Slurm launcher; see DEPLOYMENT.md for H200/H100 and storage preparation.
set -euo pipefail
umask 077
export H3_DEPLOY_ROOT="${H3_DEPLOY_ROOT:-/home/jeff/sol-h3-h200}"
export H3_LOCAL_ROOT="${H3_LOCAL_ROOT:-/raid/jeff}"
export H3_GPU_TYPE="${H3_GPU_TYPE:-H200}"
: "${SLURM_JOB_ID:?Launch this service inside a Slurm allocation}"
stage="$H3_DEPLOY_ROOT/deployment/staging"
venv="${H3_VENV:-$H3_LOCAL_ROOT/sol-h3-venv}"
checkpoints="${H3_CHECKPOINT_ROOT:-$H3_LOCAL_ROOT/sol-h3-checkpoints}"
export H3_CHECKPOINT_ROOT="$checkpoints"
mkdir -p "$H3_LOCAL_ROOT"
# Serialize staging when two allocations share a node. Never replace environments
# used by another allocation. H100/other layouts require explicitly prepared files.
exec 9>"$H3_LOCAL_ROOT/.sol-h3-stage.lock"
flock -w 1800 9
if [[ ! -x "$venv/bin/python" ]] && [[ "$H3_GPU_TYPE" != H200 || "$venv" != /raid/jeff/sol-h3-venv ]]; then
    echo 'Prepare H3_VENV for this node/path first; the H200 tar has absolute interpreter paths.' >&2
    exit 1
fi
python3 "$H3_DEPLOY_ROOT/Sol-H3/storage_preflight.py" --local-root "$H3_LOCAL_ROOT" \
    --venv "$venv" --checkpoints "$checkpoints" --stage "$stage" \
    --min-free-gib "${H3_MIN_FREE_GIB:-20}"

# The staged environment uses the same absolute path on every node. Its Python
# interpreter lives in shared /home. Never replace an existing environment.
if [[ ! -x "$venv/bin/python" ]]; then
    [[ ! -e "$venv" ]] || { echo "Incomplete existing environment: $venv" >&2; exit 1; }
    python3 - "$stage" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = json.loads((root / 'manifest.json').read_text())['sol-h3-venv.tar']
with (root / 'sol-h3-venv.tar').open('rb') as source:
    # A fresh node's system Python can predate hashlib.file_digest (Python 3.11).
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
    assert digest.hexdigest() == expected['sha256'], 'environment archive checksum'
PY
    tar -C "$H3_LOCAL_ROOT" -xf "$stage/sol-h3-venv.tar"
fi
if [[ ! -d "$checkpoints" ]]; then
    temporary="$checkpoints.partial-$SLURM_JOB_ID"
    cp -a "$stage/sol-h3-checkpoints" "$temporary"
    "$venv/bin/python" - "$stage" "$temporary" <<'PY'
import hashlib, json, sys
from pathlib import Path
stage, target = map(Path, sys.argv[1:])
for name, expected in json.loads((stage / 'manifest.json').read_text()).items():
    if not name.startswith('sol-h3-checkpoints/'):
        continue
    path = target / name.removeprefix('sol-h3-checkpoints/')
    assert path.stat().st_size == expected['bytes'], f'checkpoint size: {path}'
    with path.open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == expected['sha256'], f'checkpoint checksum: {path}'
print('CHECKPOINT_COPY_VERIFIED', flush=True)
PY
    mv "$temporary" "$checkpoints"
fi

exec 9>&-
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$venv/bin:$CUDA_HOME/bin:$PATH"
export CUDNN_PATH="$venv/lib/python3.12/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET
export TORCHINDUCTOR_CACHE_DIR="$H3_LOCAL_ROOT/sol-h3-cache/inductor"
export TRITON_CACHE_DIR="$H3_LOCAL_ROOT/sol-h3-cache/triton"
export CUDA_CACHE_PATH="$H3_LOCAL_ROOT/sol-h3-cache/cuda"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
cd "$H3_DEPLOY_ROOT/Sol-H3"
# Keep process logs on the node as well: shared storage can temporarily reject
# writes without taking down the resident workers or their local output files.
supervisor_logs="$H3_LOCAL_ROOT/sol-h3-supervisor"
mkdir -p "$supervisor_logs"
echo "Supervisor log: $supervisor_logs/supervisor-$SLURM_JOB_ID.log"
exec "$venv/bin/python" serve_node.py >>"$supervisor_logs/supervisor-$SLURM_JOB_ID.log" 2>&1
