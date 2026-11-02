#!/usr/bin/env bash
# Start one Sol-H3 worker (2 H200 GPUs) as an extra step inside its node's running Slurm allocation, with the same
# command and environment as serve_node.py's workers and the 2026-11-01/02 hand revivals (checked against live
# workers on 2026-11-02). watchdog.py runs it inside the allocation:
#   srun --jobid=<job> --overlap --nodes=1 --ntasks=1 --cpus-per-task=160 --mem=0 --cpu-bind=none bash run-worker.sh <node> <port>
# node: 0002|0005|0006|0023; port: 18080-18083, using GPUs 2i,2i+1 and CPUs 40i..40i+39 for i = port - 18080.
# Exit 2 = refused (wrong node, GPUs busy or unhealthy, port in use); the systemd unit does not retry a refusal.
# CUDA numbers GPUs from 0 and skips one it cannot initialize (0023 since its GPU 3 fault), shifting every GPU after
# it, so a real start passes the pair's UUIDs (nvidia-smi indices do not shift). H3_RENDER_ONLY=1 prints the
# environment (with CUDA ordinals), CPU list and arguments instead of starting anything.
set -euo pipefail
umask 077
node=${1:-} port=${2:-}
[[ "$node" =~ ^00(02|05|06|23)$ && "$port" =~ ^1808[0-3]$ ]] || { echo "usage: run-worker.sh <node> <port>" >&2; exit 2; }
i=$((port - 18080))
addr=10.2.201.$((10#$node)) gpus=$((2 * i)),$((2 * i + 1)) cpus=$((40 * i))-$((40 * i + 39))
V=/raid/jeff/sol-h3-venv C=/raid/jeff/sol-h3-checkpoints R=/raid/jeff/sol-h3-supervisor/fukushimadc-02-hgx-$node
cvd=$gpus
if [[ "${H3_RENDER_ONLY:-}" != 1 ]]; then
  [[ "$(hostname)" == "fukushimadc-02-hgx-$node" && -n "${SLURM_JOB_ID:-}" ]] || { echo "wrong node or not inside a Slurm job" >&2; exit 2; }
  health=$(timeout -s KILL 120 nvidia-smi -i "$gpus" --query-gpu=index,ecc.errors.uncorrected.volatile.total,remapped_rows.pending,remapped_rows.failure --format=csv,noheader) \
    || { echo "nvidia-smi failed on GPUs $gpus" >&2; exit 2; }
  [[ $(grep -cE '^[0-9]+, 0, No, No$' <<<"$health") == 2 ]] || { echo "GPUs $gpus unhealthy: ${health//$'\n'/; }" >&2; exit 2; }
  apps=$(timeout -s KILL 120 nvidia-smi -i "$gpus" --query-compute-apps=pid --format=csv,noheader) || { echo "nvidia-smi failed on GPUs $gpus" >&2; exit 2; }
  [[ -z "$apps" ]] || { echo "GPUs $gpus busy: ${apps//$'\n'/ }" >&2; exit 2; }
  if ss -ltn | grep -q "$addr:$port "; then echo "port $port in use" >&2; exit 2; fi
  cvd=$(timeout -s KILL 120 nvidia-smi -i "$gpus" --query-gpu=uuid --format=csv,noheader | paste -sd,) \
    && [[ "$cvd" =~ ^GPU-[0-9a-f-]+,GPU-[0-9a-f-]+$ ]] || { echo "cannot resolve the UUIDs of GPUs $gpus: ${cvd:-}" >&2; exit 2; }
fi
envs=(HOME=/home/jeff USER=jeff LOGNAME=jeff SHELL=/bin/bash LANG=C.UTF-8 TMPDIR=/tmp "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  H3_DEPLOY_ROOT=/home/jeff/sol-h3-h200 H3_LOCAL_ROOT=/raid/jeff H3_GPU_TYPE=H200 H3_GPUS_PER_WORKER=2
  H3_BASE_PORT=18080 H3_CHECKPOINT_ROOT=$C CUDA_VISIBLE_DEVICES=$cvd CUDA_HOME=/usr/local/cuda
  PATH=$V/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  CUDNN_PATH=$V/lib/python3.12/site-packages/nvidia/cudnn LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cudnn/lib
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
  PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET
  TORCHINDUCTOR_CACHE_DIR=/raid/jeff/sol-h3-cache/inductor TRITON_CACHE_DIR=/raid/jeff/sol-h3-cache/triton
  CUDA_CACHE_PATH=/raid/jeff/sol-h3-cache/cuda)
args=(--model $C/MiniMax-H3 --adapter $C/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors
  --output-dir /raid/jeff/sol-h3-service/$port --token-file /home/jeff/sol-h3-h200/deployment/token
  --host $addr --port $port --attention-backend sol --warmup-prompt-file /home/jeff/sol-h3-h200/demo_prompt.json
  --ready-file $R/$port.manual.ready.json --queue-capacity 8 --encoder-threads 8 --gate-failure dense
  --expect-gpu H200 --min-free-gib 20)
if [[ "${H3_RENDER_ONLY:-}" == 1 ]]; then printf '%s\n' "${envs[@]}" "CPUS=$cpus" -- "${args[@]}"; exit 0; fi
log="$R/$SLURM_JOB_ID-$port-manual-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "$(date -u +%FT%TZ) worker log: $log"
cd /home/jeff/sol-h3-h200/Sol-H3
exec env -i "${envs[@]}" taskset --cpu-list "$cpus" "$V/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=2 serve.py "${args[@]}" >>"$log" 2>&1
