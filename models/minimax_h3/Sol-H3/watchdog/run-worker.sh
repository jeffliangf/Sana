#!/usr/bin/env bash
# Start one Sol-H3 worker (2 H200 GPUs) as an extra step inside its node's running Slurm allocation, with the same
# command and environment as serve_node.py's workers and the 2026-11-01/02 hand revivals (checked against live
# workers on 2026-11-05). watchdog.py runs it inside the allocation:
#   srun --jobid=<job> --overlap --nodes=1 --ntasks=1 --cpus-per-task=160 --mem=0 --cpu-bind=none bash run-worker.sh <node> <port>
# node: 002|003|005|006|023|035|048|067 (the 32-worker fleet since 2026-11-07, after the cluster rebuild); port: 18080-18083,
# using GPUs 2i,2i+1 and CPUs 40i..40i+39 for i = port - 18080.
# Since the rebuild /raid is root-owned, so the local root is /var/tmp/jeff-sol-h3, and the workers bind the bond0
# address explicitly (H3_BIND_HOST): the full hostname resolves to a USB link-local address.
# Optional <async-publication 0|1> <code-dir>: the allocation's launch settings from its node registry, which
# watchdog.py passes so a restart uses the same release and publication mode as serve_node.py. Without them (an
# allocation started before they existed) the worker restarts as before: synchronous, from the shared Sol-H3/.
# Exit 2 = refused (wrong node, GPUs busy or unhealthy, port in use); the systemd unit does not retry a refusal.
# CUDA numbers GPUs from 0 and skips one it cannot initialize (0023 since its GPU 3 fault), shifting every GPU after
# it, so a real start passes the pair's UUIDs (nvidia-smi indices do not shift). H3_RENDER_ONLY=1 prints the
# environment (with CUDA ordinals), CPU list and arguments instead of starting anything.
set -euo pipefail
umask 077
node=${1:-} port=${2:-} mode=${3:-} code=${4:-}
[[ "$node" =~ ^0(02|03|05|06|23|35|48|67)$ && "$port" =~ ^1808[0-3]$ ]] \
  || { echo "usage: run-worker.sh <node> <port> [<async-publication 0|1> <code-dir>]" >&2; exit 2; }
if [[ -n "$mode$code" ]]; then
  [[ "$mode" =~ ^[01]$ && "$code" =~ ^/home/jeff/sol-h3-h200/[A-Za-z0-9._/-]+$ && "$code" != *..* ]] \
    || { echo "invalid launch settings: async publication '$mode', code directory '$code'" >&2; exit 2; }
fi
i=$((port - 18080))
addr=10.2.201.$((10#$node)) gpus=$((2 * i)),$((2 * i + 1)) cpus=$((40 * i))-$((40 * i + 39))
L=/var/tmp/jeff-sol-h3 host=dc02-gpu-$node.internal.bigbeargpu.cloud
V=$L/sol-h3-venv C=$L/sol-h3-checkpoints R=$L/sol-h3-supervisor/$host
cvd=$gpus
if [[ "${H3_RENDER_ONLY:-}" != 1 ]]; then
  [[ "$(hostname)" == "$host" && -n "${SLURM_JOB_ID:-}" ]] || { echo "wrong node or not inside a Slurm job" >&2; exit 2; }
  health=$(timeout -s KILL 120 nvidia-smi -i "$gpus" --query-gpu=index,ecc.errors.uncorrected.volatile.total,remapped_rows.pending,remapped_rows.failure --format=csv,noheader) \
    || { echo "nvidia-smi failed on GPUs $gpus" >&2; exit 2; }
  [[ $(grep -cE '^[0-9]+, 0, No, No$' <<<"$health") == 2 ]] || { echo "GPUs $gpus unhealthy: ${health//$'\n'/; }" >&2; exit 2; }
  apps=$(timeout -s KILL 120 nvidia-smi -i "$gpus" --query-compute-apps=pid --format=csv,noheader) || { echo "nvidia-smi failed on GPUs $gpus" >&2; exit 2; }
  [[ -z "$apps" ]] || { echo "GPUs $gpus busy: ${apps//$'\n'/ }" >&2; exit 2; }
  if ss -ltn | grep -q "$addr:$port "; then echo "port $port in use" >&2; exit 2; fi
  cvd=$(timeout -s KILL 120 nvidia-smi -i "$gpus" --query-gpu=uuid --format=csv,noheader | paste -sd,) \
    && [[ "$cvd" =~ ^GPU-[0-9a-f-]+,GPU-[0-9a-f-]+$ ]] || { echo "cannot resolve the UUIDs of GPUs $gpus: ${cvd:-}" >&2; exit 2; }
  [[ -z "$code" || -f "$code/serve.py" ]] || { echo "no serve.py in $code" >&2; exit 2; }
fi
# H3_SERVING_MEMORY_FRACTION is deploy_node.sh's H200 default since 2026-11-05; without it a restarted worker's
# PyTorch pool can fill the GPU on the longest requests.
envs=(HOME=/home/jeff USER=jeff LOGNAME=jeff SHELL=/bin/bash LANG=C.UTF-8 TMPDIR=/tmp "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  H3_DEPLOY_ROOT=/home/jeff/sol-h3-h200 H3_LOCAL_ROOT=$L H3_VENV=$V H3_BIND_HOST=$addr H3_GPU_TYPE=H200 H3_GPUS_PER_WORKER=2
  H3_BASE_PORT=18080 H3_SERVING_MEMORY_FRACTION=0.95 H3_CHECKPOINT_ROOT=$C CUDA_VISIBLE_DEVICES=$cvd CUDA_HOME=/usr/local/cuda
  PATH=$V/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  CUDNN_PATH=$V/lib/python3.12/site-packages/nvidia/cudnn LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cudnn/lib
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
  PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET
  TORCHINDUCTOR_CACHE_DIR=$L/sol-h3-cache/inductor TRITON_CACHE_DIR=$L/sol-h3-cache/triton
  CUDA_CACHE_PATH=$L/sol-h3-cache/cuda)
args=(--model $C/MiniMax-H3 --adapter $C/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors
  --output-dir $L/sol-h3-service/$port --token-file /home/jeff/sol-h3-h200/deployment/token
  --host $addr --port $port --attention-backend sol --warmup-prompt-file /home/jeff/sol-h3-h200/demo_prompt.json
  --ready-file $R/$port.manual.ready.json --queue-capacity 8 --encoder-threads 8 --gate-failure dense
  --expect-gpu H200 --min-free-gib 20)
# deploy_node.sh exports both settings to serve_node.py's workers; serve.py checks the flag against the variable.
if [[ -n "$mode" ]]; then
  envs+=(H3_ASYNC_PUBLICATION=$mode H3_CODE_DIR=$code)
  if [[ "$mode" == 1 ]]; then args+=(--async-publication); fi
fi
code=${code:-/home/jeff/sol-h3-h200/Sol-H3}
if [[ "${H3_RENDER_ONLY:-}" == 1 ]]; then printf '%s\n' "${envs[@]}" "CPUS=$cpus" "CWD=$code" -- "${args[@]}"; exit 0; fi
log="$R/$SLURM_JOB_ID-$port-manual-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "$(date -u +%FT%TZ) worker log: $log"
cd "$code"
exec env -i "${envs[@]}" taskset --cpu-list "$cpus" "$V/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=2 serve.py "${args[@]}" >>"$log" 2>&1
