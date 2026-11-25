# Sol-H3

[Project page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/)

Production inference package for MiniMax-H3 with synchronized video and audio output.

- Text-to-video, first-frame image-to-video, and Ref2VA
- 1344x768 at 24 FPS
- 5, 10, or 15 second output
- 1, 2, 4, or 8 GPUs; all tasks validated on 8x NVIDIA B300, T2V also on 1/2/4/8x H200
- Fast SOL/BSA profile enabled by default
- Optional fused MXFP8 DiT compute on SM100-family GPUs

The default profile prioritizes speed and uses lossy attention/communication acceleration. Select
`dense` when a dense-attention reference is required or when running on one GPU. DiT linear compute
remains BF16 unless `--compute-quant mxfp8` is selected.

To retain separate LoRA branch arithmetic, use `--lora-mode separate`, or
`--lora-mode fused` to fuse its BF16 sums into subsequent kernels. See
[LoRA consumer fusion](LORA_FUSION.md) for requirements and validation.

## Resident H200 service

- **Calling the service:** the [API reference](API.md) covers both request forms,
  job fields, retry rules, measured latency, and a tested example client.
- **Operating it:** the [deployment design and runbook](DEPLOYMENT.md) covers the
  gateway, workers, storage, the serving memory cap, scaling, validation, and drain.

The model is **H3-Base FL2VA plus the eight-step SMD-H3 E adapter** since 2026-11-21
(the four-step FastH3 adapter before; see [Adapter profiles](#adapter-profiles)). A stable CPU
gateway routes requests to two-GPU H200 workers. Each worker generates one video
at a time and queues up to eight more. Workers serve text-to-video and first-frame
image-to-video with stereo audio at 24 fps, in either request form:
- The original `{prompt, duration, seed}` form: 1344x768, 124 or 243 frames.
- The SGLang MiniMax-H3 form: `t2va` or `fl2va`, six aspect ratios or the first
  frame's own, 124–345 frames.

**State on 2026-11-23.** Since 2026-11-21, 20 two-GPU H200 workers serve **SMD-H3 E** on five nodes:
`dc02-gpu-{002,003,005,006,023}.internal.bigbeargpu.cloud`, Slurm jobs 102, 103,
104, 101 and 105, four workers per node on ports 18080–18083.
- **Model settings.** Adapter profile `smd-e-8step`, with no numerical gate
  (`H3_GATE_FAILURE=off`), as on songyang's SMD-E service on 035/067.
- **Release.** The read-only release
  `/home/jeff/sol-h3-h200/releases/20261121-smd-e-11eeb3f94d12/Sol-H3`.
- **Unchanged.** The 0.95 serving cap, overlapped MP4 publication
  (`H3_ASYNC_PUBLICATION=1`), prompts of up to 16,000 characters, the gateway, its URL
  and both request forms.
- **Validation.**
  - 80 direct jobs covering every worker, and 20 through the gateway.
  - The H3 preflight of the RL run `grpo-mixed-hydrogen-group8-16h100-20261118-toolc-16k`,
    with that run's own code, passes against the live gateway.
- **Same videos as songyang's service.** That service is guian's package, which runs
  this engine at Sana `175081e` with E. For 52 of its stored T2V and I2V test requests
  (same prompt, seed and first frame), the decoded frames and audio differ only by video
  encoding. Details are in [Deployment history](#deployment-history).
  - The request forms the RL sends match too: the legacy form, `t2va` at 5 s, and `fl2va` with
    `aspect_ratio: auto`, through the gateway.
  - Two of our own renders of the same request differ by as much: their audio is bit-identical
    and their video is 40–45 dB apart.
  - Every setting that affects the output (`settings-comparison.md` in the evidence) and every
    numeric package version is the same.
- **In production.** From 2026-11-21 to 18:07 UTC on 2026-11-23, the gateway completed 63,198
  jobs, nearly all from the verl RL runs, and all succeeded. A job took a median of 58 s on its worker, and at
  most 109 s. Since 18:22 UTC on 2026-11-23 the fleet serves RL job 1335, until about 2026-11-25
  23:25 UTC; see the fleet freeze in the [runbook](DEPLOYMENT.md).
- **H100 workers render different videos.** On 2026-11-23, four-GPU SMD-E workers on H100 gave a
  different video for the same request and seed, so they were not added
  ([H100 workers](DEPLOYMENT.md#h100-workers-on-another-cluster)).
- **Evidence.** `deployment/smd-e-20261121/`.
- **Other nodes.** Nodes 035, 048 and 067 run other users' jobs.
- **Node-local files.** The rebuilt cluster keeps node-local environments, weights,
  caches, and outputs in `/var/tmp/jeff-sol-h3`.
- **Live capacity.** Check `/healthz`.

The previous fleet, 32 workers from 2026-11-07, passed an extended validation on
2026-11-08:
- 35 API contract checks;
- 85 T2V and 94 I2V cases, and 19 worst-case cases;
- 256 mixed requests with 64 concurrent clients.

The client decoded 456 full videos. Evidence is in
`deployment/fleet-32-dc02-20261107/finish-20261108/final-report.json`. User
lingering keeps the gateway and watchdog running after SSH logout.

### Access the deployed H200 service

Use **`http://10.2.201.187:18080`** for every request. This persistent gateway
dispatches to healthy GPU workers and keeps the same address when GPU counts,
worker ports, or Slurm jobs change. Submission, status polling, and video downloads
all use this one URL and the existing bearer token.

The deployment is on the Fukushima H200 cluster. The `10.2.201.*` addresses are
private and require cluster network access or an SSH tunnel.

| Item | Value |
|---|---|
| Public SSH login / gateway | `157.66.255.187`, port **44222** (`VM-187-Konsttech` since the rebuild) |
| Existing deployment owner | `jeff` |
| Configured SSH alias on this workstation | `h200-login`, through `h200-relay` |
| Jeff's login key on this workstation | `/home/jeff/.ssh/id_ed25519_h200_agent` |
| Shared service token file on the cluster | `/home/jeff/sol-h3-h200/deployment/token` |
| Stable API base URL | **`http://10.2.201.187:18080`** |
| Shared deployment code and logs | `/home/jeff/sol-h3-h200` |

Log in from this workstation, then query the private endpoint:

```bash
ssh h200-login
# On the cluster login host:
curl --noproxy '*' --fail http://10.2.201.187:18080/healthz
```

To call it from another machine, run this tunnel on the **client machine** and
keep the terminal open:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:28080:10.2.201.187:18080 h200-login
```

Then use `http://127.0.0.1:28080` as the base URL on that machine. This single
tunnel reaches the entire worker pool. A tunnel on a
login host does not make that host's loopback address available to Ray actors on
other compute hosts. Run forwarding on each client host or provide network routing.
The aliases above are configured in this workstation's SSH config; another
client needs its own authorized SSH route. Other people need authorized SSH
access and the service bearer token; the listed key and private token file belong
to Jeff's deployment account.

Callers use the [example client](API.md#example-client) and the rules in the
API reference.

For verl, see the **Self-hosted H3 on H200** section of
`/home/jeff/verl/examples/prompt_rewrite/README.md`. Its new-run options are
`--h3-backend self-hosted`, `--h3-endpoints http://10.2.201.187:18080`, and
`--h3-token-file <shared token filename>`. The policy can continue calling
`h3_max(prompt)`; the adapter performs submission, polling, download, and media
validation against this service. It sends the original request form.

### Operate the stable gateway

`gateway.py` is a CPU service on the persistent login host, outside GPU Slurm
allocations. `deploy_h200_gateway.sh` installs its Python environment and systemd
user service/socket. Run it on `jeff@157.66.255.187:44222`:

```bash
bash /home/jeff/sol-h3-h200/Sol-H3/deploy_h200_gateway.sh
systemctl --user status h3-gateway.socket h3-gateway.service
journalctl --user -u h3-gateway.service -n 50
```

User lingering keeps the service running after SSH logout and starts it at boot.
The socket keeps the same listener across service restarts. The deployment script
pins the address in `gateway/endpoint.json` and refuses an accidental address or
port change on later updates. Update gateway code, then run
`systemctl --user restart h3-gateway.service`; GPU workers continue running. The
gateway answers again only after reading every job record, 32 at a time: about 2.3 ms per
record on the cluster (35 minutes for 175,000 records when they were read one at a time).
This is one gateway host, so a host or network outage can still interrupt access.

The gateway reads `deployment/*/node.json` and probes workers every two seconds.
Registries older than 300 seconds, disabled workers, and workers without a ready
health response receive no new jobs. Heartbeats older than 120 seconds are reported
stale during that bounded grace period. Explicit stops and exclusions take effect
immediately. New allocations launched with `deploy_node.sh` join
automatically when ready. Requests go to a worker with the shortest queue.
Original-form requests can go to any ready worker. SGLang-form requests go only to
workers whose `/healthz` reports `api.version` 2 and an `api.num_frames.max` that
covers the request, so workers can be upgraded one at a time.
Each accepted job is pinned to its original worker in `gateway/jobs/<id>.json`;
completed videos are copied to shared `gateway/videos/<id>.mp4` before the
gateway reports success. Those results remain downloadable after that worker
is retired, and gateway restarts resume polling accepted jobs without resubmitting.

For maintenance, atomically write `gateway/pool.json` under
`/home/jeff/sol-h3-h200` with one of these configurations:

```json
{"excluded_workers": ["http://10.2.201.2:18080", "http://10.2.201.2:18081"]}
```

This drains the listed workers: they receive no new gateway jobs, while their
accepted jobs continue polling and archiving. Wait until their gateway job
records are failed or have `video_cached: true`, and their own `/healthz` has
`active_job: null` and `queued: 0`, before stopping their allocation. Direct
clients must also stop submitting to a retiring worker. Abruptly stopping a GPU
worker can interrupt its unfinished jobs; the gateway does not silently repeat
generation on another worker.

`{"workers": ["http://10.2.201.6:18080"]}` explicitly sets the admitted pool.
`{"workers": []}` pauses new submissions; health and new submissions return 503,
but completed status and video downloads still work. Delete `pool.json` or write
`{}` to restore automatic discovery. Pool changes require no gateway restart.

Shared storage remains necessary for gateway job persistence and video archives.
If registry publication stays unavailable for more than 120 seconds, automatic
discovery stops routing new work to those stale entries until publication resumes.
The supervisors keep resident workers running, and completed outputs remain on
local disk.

### Launch another service

`serve.py` keeps one distributed worker loaded and serves authenticated, queued
text-to-video and first-frame image-to-video requests (`h3_request.py` holds the shared
contract): the original `{prompt, duration, seed}` form at 1344x768, or the SGLang
MiniMax-H3 form (`t2va`/`fl2va`, base64 first frame, six aspect ratios or the image's
own, 124-345 frames at 24 FPS) with stereo audio. Install the optional HTTP
dependencies with `pip install -r requirements-service.txt`.

Create a private token file with at least 32 random characters, then launch inside
a two-GPU allocation (the model and adapter paths are the same as `infer.py`):

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=2 serve.py \
  --model /path/to/MiniMax-H3 \
  --adapter /path/to/dense-datafree/adapter_model.safetensors \
  --output-dir /path/to/worker-outputs --token-file /path/to/private-token \
  --warmup-prompt-file ../demo_prompt.json --host 127.0.0.1 --port 18080
```

`GET /healthz` returns 503 during model loading and two warmups at each duration,
then 200 when ready. Other endpoints require `Authorization: Bearer <token>`:

| Method / path | Behavior |
|---|---|
| `POST /v1/videos` | Accept `{"prompt":"…","duration":5,"seed":42}` or the SGLang form (`task`, `conditions`, `target`); return 202 with a job ID and relative status/video URLs. An `Idempotency-Key` UUID becomes the job ID, and repeating it returns that job |
| `GET /v1/jobs/{id}` | Return `queued`, `running`, `succeeded`, or `failed` (with `error_code`), with timings after completion |
| `GET /v1/jobs/{id}/video` | Download the completed MP4; return 409 while unfinished |

Each worker handles one generation at a time and queues up to eight waiting
requests; a full queue returns 429. Jobs and videos remain in that worker's output
directory across restarts. Interrupted requests are marked failed, so callers can
decide whether to resubmit. Keep polling/downloading from the worker that accepted
the request. The service validates the adapter profile's transformer forward count
(eight for `smd-e-8step`) and audio/video stream metadata before marking a request
successful. A failed distributed generation terminates the process group instead of
reusing inconsistent GPU state.

With `--async-publication` (`H3_ASYNC_PUBLICATION=1` in the launchers), a worker
starts the next generation as soon as the previous video is staged in pinned host
memory, and one background thread finishes and publishes that MP4. The job stays
`running`, with `stage: publishing`, until its MP4 is published. The launcher default
is off; the H200 fleet runs with it since 2026-11-09, which measured 7.3% less GPU time
per video for an identical request. See
[Overlapped output publication](DEPLOYMENT.md#overlapped-output-publication).
`publication_benchmark.py` runs the paired benchmark against excluded workers.

The service keeps the numerical correctness gate enabled with its existing
limits. `--gate-failure dense` (the service default) uses dense SDPA when **any
rank** rejects a kernel's result. Rejected results are discarded and the shape
stays on dense attention for that worker's lifetime. Accepted shapes are checked
again on the first eligible sparse call of each new request. A nonfinite dense
reference, CUDA error, or failed collective still terminates the worker.
`--gate-failure error` restores the fatal gate behavior; benchmarks retain that
strict default. No numerical tolerance was increased.

`--gate-failure off` (deploy setting `H3_GATE_FAILURE=off`) runs no gate: every
shape uses Sol attention, as songyang's SMD-E service does. The H200 SMD-E fleet
runs this way since 2026-11-21.
- In the 2026-11-20/21 adapter A/B, every shape the gate had rejected failed by one
  bf16 ULP of `max_abs`.
- In that A/B, Sol-path videos scored at least as well as dense-fallback ones.
- At eight steps, dense fallback costs about 27% on a 10-second clip.

With the gate off, `used_gate_fallback` is always false and `/healthz` reports
`gate_failure_policy: off` and `gate_each_request: false`.

Completed job JSON includes `used_gate_fallback` and per-rank `attention`
counters (`sparse_calls`, `dense_calls`, `gate_fallback_calls`, `gate_failures`)
plus the gate's measured errors. Dense fallback can be slower, so include those
counters when reporting latency. `/healthz` exposes `gate_failure_policy`.

`deploy_node.sh` and `serve_node.py` provide the Slurm node deployment. Defaults are one independent
worker per H200 GPU pair (2/4/6/8 allocated GPUs), consecutive ports
starting at `H3_BASE_PORT` (default 18080), eight CPU encoder threads per
worker, and separate CPU affinities. They bind to the node's internal IP and
pin each worker to its GPU pair by UUID. The supervisor restarts a crashed
worker every time. A crash after at least 30 minutes of readiness starts a fresh
count; a worker that fails three starts in a row is disabled while healthy
siblings continue, and the node registry reports `degraded`. If every worker is
disabled, the supervisor exits and releases the allocation. On H200 the launcher caps
PyTorch's memory pool at 95% after warmup (`H3_SERVING_MEMORY_FRACTION`).
`H3_CODE_DIR` selects the code an allocation runs (normally an immutable release
directory) and `H3_ASYNC_PUBLICATION` its publication mode; the node registry records
both, so watchdog restarts use the same release and mode. It defaults to
`/home/jeff/sol-h3-h200` for shared code/registry and `/raid/jeff` for environments,
checkpoints, caches, and outputs. It expects a private `deployment/token` and can
initialize a fresh node from a checksummed `deployment/staging` snapshot. The
staged venv is tied to `/raid/jeff/sol-h3-venv`. Other paths and H100 nodes require
a prepared environment; `H3_VENV` and `H3_CHECKPOINT_ROOT` select it and the weights.
`H3_GPU_TYPE=H100` defaults to four GPUs per worker with `H3_WEIGHT_SHARDING=1`: a
whole pipeline (about 129 GiB) does not fit in an 80 GB H100, so the worker splits the
text encoder's decoder layers and the DiT blocks across its four GPUs and gathers each
block as it runs (`h3_runtime/weight_sharding.py`; outputs are bitwise those of
replicated weights). A sharded H100 worker peaked at 46.9 GiB allocated per GPU on
345-frame requests (2026-11-06).
See [the runbook](DEPLOYMENT.md#parameterize-the-next-fleet) for all settings.

```bash
sbatch --partition=all --nodes=1 --ntasks=1 --cpus-per-task=160 --mem=1600G \
  --gres=gpu:nvidia_h200:8 --constraint=h200 --time=0 \
  --nodelist=YOUR_NODE deploy_node.sh
```

To add a separate four-GPU allocation on a node already serving ports
18080–18081, reserve idle GPUs through Slurm and use a free port range:

```bash
sbatch --partition=all --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=800G \
  --gres=gpu:nvidia_h200:4 --constraint=h200 --time=0 \
  --export=ALL,H3_BASE_PORT=18082 \
  --nodelist=YOUR_NODE deploy_node.sh
```

Confirm both ports are unused before launching. Exclude the new endpoints in
`gateway/pool.json` during validation, then remove those exclusions to admit
them without restarting the gateway. Keep other operators' pool settings intact.

The default node registry is `deployment/<hostname>/node.json`. A nondefault
base port uses `deployment/<hostname>-p<base-port>/node.json`, so concurrent
allocations keep separate state. Worker logs and readiness records are local to
the GPU node under `$H3_LOCAL_ROOT/sol-h3-supervisor/<registry-name>/` (normally
`/raid/jeff/sol-h3-supervisor/`). Supervisor output is in
`$H3_LOCAL_ROOT/sol-h3-supervisor/supervisor-<slurm-job-id>.log`; the Slurm launch
log prints this path. The shared registry records each worker's local log path.
Check live `/healthz` responses when using the registry,
because a hard node failure can leave stale files. Stop a deployment with
`scancel <its-job-id>`. Outputs are retained until explicitly removed; admission
stops when free disk space falls below 20 GiB. The HTTP endpoints are intended for
the private cluster network or an SSH tunnel. The gateway above provides routing
across workers; use its fixed URL in training clients.

### Adapter profiles

Each worker merges one few-step adapter for its lifetime, together with the sampling schedule that adapter
was distilled for. The AdaLN precompute fixes the step schedule on the first request.
- **Where profiles live.** `adapter_profiles.json` defines them; `h3_runtime/adapters.py` validates them.
- **Selecting one.** Use `serve.py --adapter-profile` or the deploy setting `H3_ADAPTER_PROFILE`, which takes
  one name, or a comma-separated list with one name per worker.
- **What a profile fixes:**
  - the transformer forward count;
  - the video and audio flow shifts and the strength;
  - optionally `final_step_size_scale`;
  - the adapter file, by path, size and SHA-256.
- **Before loading.** `serve.py` checks the file's size and SHA-256 before it loads any weights.
- **Where `/healthz` reports it.** Each worker's `/healthz` reports the profile under `adapter` and its forward
  count under `transformer_forwards`. The gateway does not aggregate them.

| Profile | Adapter | Forwards | Shifts (video/audio) | Notes |
|---|---|---|---|---|
| `fasth3-dense-datafree` | FastVideo FastH3 4-step Preview v1 (dense, data-free) | 4 | 12/3 | T2V/I2V default |
| `lightx2v-fl2v-4step-v1.2`, `-v1.3` | LightX2V MiniMax-H3 Turbo FL2V 4-step | 4 | 6/3 | file alpha 8 at rank 128 |
| `lightx2v-fl2v-8step-v1.2` | LightX2V MiniMax-H3 Turbo FL2V 8-step | 8 | 12/3 | per-layer alpha tensors |
| `lightx2v-ref2v-4step-v0.1` | LightX2V Ref2VA 4-step | 4 | 12/3 | Ref2VA default |
| `smd-ref2va-8step` | SMD-MiniMax-H3-Ref2VA-8step | 8 | 12/3 | last step x0.5; 512-pixel reference short edge; a worker with it serves `ref2va` (DEPLOYMENT.md, Ref2VA gateway) |
| `smd-e-8step` | SMD-H3 E (`smd_h3_div16f_ck3`) | 8 | 12/3 | last step at half size (`final_step_size_scale` 0.5); production since 2026-11-21 |

**LoRA alpha.** The loader never guesses alpha. It reads, in order: per-layer `.alpha` tensors, then the file's
`alpha` metadata, then the profile's `alpha`. FastVideo hybrid files use their rank. E embeds rank 192 and
alpha 192, so its update scale is 1.0.

**SMD's shorter last step.** SMD samples 8 steps from 1 to 0, with the last step half as long as the others.
- The pre-shift grid is 1, .867, .733, .6, .467, .333, .2, .067, 0. It comes from `raw_sigma_grid` in the SMD
  release's `tools/infer_minimax_h3.py`.
- The grid is shifted per modality: 12 for video, 3 for audio.
- `install_explicit_schedule` makes both schedulers use that grid whatever step count the pipeline asks for, as
  SMD's runner does.
- Profiles with even steps keep the scheduler's own grid, so they sample exactly as before.

**Files outside the Hugging Face Hub.** E is not on the Hub. Its profile's `source.copy_from` names the shared
copy `/home/guian/shared/h3-serving/models/smd-e/`. `download_checkpoints.py --adapter-profile smd-e-8step`
copies that file and verifies it.

### Validation of the SGLang request form

On 2026-11-05 the SGLang request form was validated on eight test workers on 0048
and 0067. They ran from an isolated test root and test gateway, not the production
gateway. The form covers `t2va`, first-frame `fl2va`, six aspect ratios or the
image's own, and 124–345 frames. Each phase passed with and without the 0.95
serving cap:
- 8 original-form requests;
- 35 API contract checks;
- 85 T2V cases covering every ratio and frame count;
- 94 I2V cases, including 6 expected rejections;
- 19 worst-case requests.

A warm-latency set of 96 requests also ran with the cap. Testing led to two fixes:
- The VAE decode now runs in fixed rounds of tiles. It is bit-identical to the
  per-clip decode, bounds decode memory, and needs one compiled shape. The old
  decode ran out of memory above about 290 frames and recompiled for every new shape.
- Original-form requests no longer fail on the new fields.

Uncapped, the worst case (8,000-character prompt, 345 frames, dense fallback)
filled the GPU to 139.8 of 140.4 GiB. With `H3_SERVING_MEMORY_FRACTION=0.95` it
peaked at 128.0 GiB allocated and 135.1 GiB in use, so `deploy_node.sh` now sets
the cap on H200 by default. Evidence is in `deployment/api-v2-validation-20261105/`
on the cluster.

### Deployment history

Evidence is in `/home/jeff/sol-h3-h200/deployment/` on the cluster unless noted.

| Date | Change | Evidence |
|---|---|---|
| 2026-10-22 | Stable gateway in front of two eight-H200 nodes (four two-GPU workers each). Numerical-gate repair with dense fallback | `gateway-20261022/`, `gate-repair-20261022/`, `verl-client-20261022/` |
| 2026-10-23 | Workers added on 0048, then on 0002 and 0048 | `scale-0048-20261023/`, `scale-fill-0002-0048-20261023/` |
| 2026-10-24 | Shared-storage `ENOSPC`/`EIO` stopped every allocation. The supervisor now survives failed heartbeats; logs moved to node-local disk | `storage-recovery-20261024/` |
| 2026-10-26 | Fleet stopped | `stop-20261026/` |
| 2026-10-28 | Gateway with caller job IDs, `error_code`, and `video_sha256`. Sixteen workers on 0002, 0005, 0006, 0023 | `redeploy-20261028/`, `h200-0002-20261028/`, `scale-0005-0006-0023-20261028/` |
| 2026-11-02 | Workers crashed after about 570 distinct prompt lengths and ended every allocation. Unlimited restarts, GPU pinning by UUID, and a 0.95 serving memory cap trial on 0002 | `stability-review-20261102/`, `redeploy-20261102/` |
| 2026-11-03 | 0003 and 0035 added: 24 workers | `scale-0003-0035-20261103/` |
| 2026-11-04 | 0048 and 0067 added: 32 workers. Fleet stopped at 22:57 UTC | `scale-0048-0067-20261104/`, `stop-32-workers-20261104T225735Z/` |
| 2026-11-05 | SGLang request form, round-based VAE decode, and the H200 memory cap by default, validated on test workers. Stale configuration archived | `api-v2-validation-20261105/`, `archive-20261105-stale/` |
| 2026-11-05 | That code deployed: gateway updated in place, 16 workers on 0003, 0035, 0048, 0067. The watchdog restarts workers with the cap | `rollout-api-v2-20261105/` |
| 2026-11-07/08 | 32 workers on the rebuilt eight-node cluster; persistent gateway and watchdog restored. T2V/I2V matrix and 256-request concurrent stress run passed | `fleet-32-dc02-20261107/finish-20261108/` |
| 2026-11-09 | 048 and 067 released to another user. Overlapped MP4 publication deployed to all workers from a read-only release; 035 freed; 20 workers on 002, 003, 005, 006, 023 | `release-2-nodes-20261109T174700Z/`, `async-publication-20261109/` |
| 2026-11-10 | 035 back on the same release: 24 workers | `async-publication-20261109/` |
| 2026-11-18 | Prompt limit raised to 16,000 characters; all 24 workers redeployed (Slurm 52–57). At 19:41 UTC 035 was drained and released: 20 workers on 002, 003, 005, 006, 023 | `prompt-16k-20261118/`, `release-035-20261118/` |
| 2026-11-20 | Whole fleet drained and stopped at the user's request; gateway kept answering 503 | `stop-20-workers-20261120T211328Z/` |
| 2026-11-21 | SMD-H3 E on 20 workers (Slurm 101–105): adapter profile `smd-e-8step`, gate off, same gateway and API. The same videos as songyang's SMD-E service on 52 of its stored requests (26 T2V, 26 I2V): video PSNR 33.7–43.0 dB (median 38.4) and audio correlation 0.94–0.9997 (median 0.996) after the encoders' 1024-sample AAC offset. Re-encoding our own output with their encoder settings produces the same gap; a different seed gives 8.3 dB and 0.002. Worker outputs cleaned to about 115 GB free per node | `smd-e-20261121/`, `worker-output-cleanup-20261121/` |
| 2026-11-21/23 | RL training on SMD-E (GMI Slurm 1225–1229, 1261): 63,198 gateway jobs, all succeeded; median 58 s and at most 109 s per job on its worker. Worker outputs cleaned again under load (15:06 UTC on 11-21, 03:16 UTC on 11-23), with no failed job. Job 1261 stopped on 11-23 at 18:10 UTC after one download 404, most likely caused by a `find` over the gateway's `jobs/` directory; job 1335 replaced it | `worker-output-cleanup-20261121T1506Z/`, `worker-output-cleanup-20261123T0316Z/` |
| 2026-11-23 | Three SMD-E H100 workers on GMI gpu-001/005 reached the gateway host through reverse tunnels but rendered different videos from the H200 workers for the same seed (18–31 dB PSNR), so they were not admitted and were stopped | GMI cluster: `deployment/smd-e-h100-20261123/` |

## Validation

All three task paths have been exercised on 8x NVIDIA B300 at 1344x768:

| Task | Result |
|---|---|
| T2V | 124 / 243 / 362 frames: 1.653 / 3.732 / 6.612 s warm 8-GPU pipeline medians |
| I2V | Native first-frame conditioning accepted at 124 frames and in a resident 362-frame regression |
| Ref2VA | 124 / 243 / 362 frames: 2.192 / 4.348 / 5.947 s warm pipeline medians |

The FastH3 preview adapter is published for T2V. I2V combines that adapter with MiniMax-H3's native
first-frame conditioning path, so its validation is a deployment compatibility result rather than
an upstream I2V training claim. Latency excludes checkpoint loading, warmup, and MP4 encoding.

Kernel validation includes SM103 B300 through the SM100-family backend and the H200 SM90 T2V
checks below. H200 first-frame I2V was validated on 2026-11-05 at 124–345 frames and every
canvas shape, for function and memory but not output quality. H200 Ref2VA and the reachable
SM120 paths remain unvalidated.

### T2V benchmarks

The following results were measured on NVIDIA B300 SXM6 AC GPUs. Each cell is the median of three
requests after one warmup. All runs use the same prompt and seed at 1344x768 and 24 FPS, generate
synchronized stereo audio, and include text encoding, DiT denoising, and video/audio VAE decoding.
Checkpoint loading, warmup/compilation, and MP4 encoding are excluded.

| Engine | GPUs | 5 s / 124f | 10 s / 243f | 15 s / 362f |
|---:|---:|---:|---:|---:|
| Sol-H3 | 1 | **13.745 s** | **37.813 s** | **52.260 s** |
| SGLang | 1 | 129.898 s | 376.942 s | 746.885 s |
| Diffusers | 1 | 159.547 s | 442.533 s | 847.486 s |
| Sol-H3 | 4 | **2.918 s** | **6.993 s** | **12.542 s** |
| Diffusers | 4 | 48.552 s | 127.461 s | 237.171 s |
| SGLang | 4 | 35.328 s | 100.440 s | 194.930 s |
| Sol-H3 | 8 | **1.653 s** | **3.732 s** | **6.612 s** |
| SGLang | 8 | 18.250 s | 50.660 s | 99.513 s |
| Diffusers | 8 | 30.673 s | 71.316 s | 131.647 s |

#### Detailed Settings

- Diffusers: Base BF16 dense model with 50 scheduler points (49 DiT forwards); Ulysses on 4/8 GPUs.
- SGLang: Base BF16 dense model with 50 scheduler points (49 DiT forwards); Ulysses on 4/8 GPUs.
- Sol-H3: FastH3 adapter with five scheduler points (four DiT forwards) and BF16 compute; dense
  attention on 1 GPU and SOL/BSA on 4/8 GPUs. The multi-GPU profile uses INT8 QKV transport, FP8
  output transport, and parallel video/audio VAE decoding.

The official and SGLang rows are directly comparable. Sol-H3 uses a distilled four-forward adapter
and its multi-GPU profile uses approximate SOL/BSA attention, so the difference from either
49-forward baseline is not a runtime-only speedup.

### H200 T2V benchmarks

Measured on 2026-10-22 across two eight-H200 nodes, using NVLink within each multi-GPU worker.
One benchmark worker ran per node at a time. All rows use the **same four-forward FastH3
dense-datafree adapter**, merged LoRA, BF16
DiT compute, the repository's structured demo prompt, seed 42, 1344x768 at 24 FPS, and 32-kHz
stereo audio. Each value is the median of three requests after two warmups at that duration.

| GPUs | Attention | 5 s pipeline | 5 s through MP4 | 10 s pipeline | 10 s through MP4 |
|---:|---|---:|---:|---:|---:|
| 1 | Dense | 29.81 s | 31.15 s | 85.81 s | 87.79 s |
| 2 | Sol SM90, expandable allocator | 11.53 s | 12.75 s | 28.95 s | 31.03 s |
| 4 | Dense | 7.68 s | 9.07 s | 21.81 s | 23.94 s |
| 4 | Sol SM90 | 5.90 s | 7.21 s | 14.92 s | 17.05 s |
| 4 | SOL/BSA SM90 | 5.97 s | 7.17 s | 15.16 s | 17.22 s |
| 8 | Dense | 4.00 s | 5.34 s | 11.13 s | 13.15 s |
| 8 | Sol SM90 | 3.17 s | 4.41 s | 7.74 s | 9.85 s |
| 8 | SOL/BSA SM90 | 3.18 s | 4.34 s | 7.85 s | 9.94 s |

Pipeline time includes text encoding, denoising, and video/audio decoding. Through-MP4 time also
includes synchronous MP4 completion. Both exclude model loading, warmup/compilation, validation,
and telemetry. Direct Sol uses BF16 communication; SOL/BSA uses INT8 QKV and FP8 output transport.
Those transport formats do not enable the Blackwell-only MXFP8 compute mode.

Every request executed four transformer forwards. Direct Sol on 2/4/8 GPUs and SOL/BSA on 4/8
GPUs passed the existing all-blocks-selected dense-reference numerical gate on every rank at
both durations, and executed 144 sparse attention calls per rank per request. Two-GPU warm
allocated memory peaked at 129.3 GiB per GPU, with 137.0 GiB reserved by PyTorch; four/eight-GPU
allocated memory peaked at 120.1 GiB, with 136.5 GiB reserved. One GPU peaked at 121.0 GiB allocated
and 137.4 GiB reserved. Model loading must be budgeted separately.

The two-GPU row requires `PYTORCH_ALLOC_CONF=expandable_segments:True` before launching Python.
With the default allocator, 5-second clips passed but the first 10-second batched VAE decode
failed to allocate a 10.75-GiB buffer while 16.14 GiB was reserved but unallocated. The expandable
allocator completed the entire 5/10-second sequence without changing compute precision or VAE
batching. The one-GPU path uses dense attention and the original tiled decoder.

For 10-second clips, mean GPU-seconds through MP4 were 87.95 / 62.24 / 68.27 / 78.58 for 1 / 2 / 4 /
8 GPUs (dense on one, direct Sol on the others). Two GPUs use about 9% fewer GPU-seconds than four,
at 31.03 s versus 17.05 s median latency. These figures describe one serial worker, not measured
multi-worker throughput.

These are warm, single-prompt latency measurements. Sparse routing, communication quantization,
and distillation have not been evaluated for perceptual equivalence to H3-Max here. The matrix
above measures one worker at a time; see the resident service's concurrent validation for
multi-worker results and first-use prompt timings. Direct Sol on SM90 specializes kernels by
sequence length; new prompt lengths can incur compilation.

The model revision is `42ed227ee7df40d41602854ae760620d6eb651fe`; the FastVideo adapter revision
is `f509e629374cac104e7f62daecce6d1488a3041d`. The environment uses Python 3.12.15, PyTorch
2.10.0+cu130, driver 580.173.02, and the pinned Diffusers/cuDNN frontend commits in
`requirements.txt`. Reproduce the measurement boundaries with `benchmark.py` below.

### Optional MXFP8 compute

On 8x B300, fused MXFP8 attention and FFN linears in transformer blocks 2 through 46 reduce the
same 5-second T2V request from 1.655 s to 1.472 s. The selected linear weights occupy 16.65 GiB per
rank instead of 32.30 GiB.

| Compute | 5 s / 124f median | Latency change | Selected weight storage |
|---|---:|---:|---:|
| BF16 | 1.655 s | baseline | 32.30 GiB |
| MXFP8 | **1.472 s** | **-11.1%** | **16.65 GiB** |

This is a lossy mode: the generated video measured 13.82 dB decoded-RGB PSNR and 0.540 SSIM against
the BF16 output from the same prompt and seed. One-GPU dense and eight-GPU SOL/BSA T2V paths passed
end-to-end checks on B300; the table above is the eight-GPU A/B result. T2V, I2V, and Ref2VA all use
the same task-selected transformer integration. With FP8 multi-GPU output transport, quantized
blocks reuse the received E4M3 data directly for the output projection instead of converting it to
BF16 and immediately back to MXFP8. Enable the mode explicitly with `--compute-quant mxfp8`.

## Setup

Recommended environment: Linux, Python 3.12, CUDA 13.0, and PyTorch 2.10.

```bash
conda create -n h3-infer python=3.12 -y
conda activate h3-infer
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130
# The cuDNN frontend source build needs the headers and libraries supplied by PyTorch.
export CUDNN_PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cudnn")')"
pip install -r requirements.txt
```

Accept the MiniMax-H3 model terms on Hugging Face, authenticate, and download the checkpoints:

```bash
hf auth login
python download_checkpoints.py --output-dir ./checkpoints
```

For Ref2VA, download its model partition and four-step adapter instead:

```bash
python download_checkpoints.py --output-dir ./checkpoints --task ref2va
```

## Text-to-video

```bash
torchrun --standalone --nproc_per_node=8 infer.py \
  --model ./checkpoints/MiniMax-H3 \
  --adapter ./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors \
  --task t2v --duration 5 \
  --prompt-file prompt.txt \
  --output output.mp4 --warmup
```

## Image-to-video

```bash
torchrun --standalone --nproc_per_node=8 infer.py \
  --model ./checkpoints/MiniMax-H3 \
  --adapter ./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors \
  --task i2v --duration 10 \
  --image first_frame.png \
  --prompt-file prompt.txt \
  --output output.mp4 --warmup
```

## Ref2VA

References are read in command-line order. Repeat `--reference` to combine an image or video with optional audio.

```bash
torchrun --standalone --nproc_per_node=8 infer.py \
  --model ./checkpoints/MiniMax-H3 \
  --adapter ./checkpoints/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --task ref2va --duration 10 \
  --reference image:subject.png \
  --reference audio:voice.wav \
  --prompt-file prompt.txt \
  --output output.mp4 --warmup
```

Ref2VA accepts image, video, and audio references. Audio cannot be the only reference. Use `sol_bsa` (default) or `dense` for this task. Reference images follow the adapter profile's `reference_image_short_edge` when it sets one (`smd-ref2va-8step`: 512, as SMD trained), otherwise the validated fast `match` profile; pass `--reference-image-resize-mode diffusers` only when official 2048-short-edge preprocessing parity is required.

Main options:

| Option | Values |
|---|---|
| `--task` | `t2v`, `i2v`, or `ref2va` |
| `--duration` | `5`, `10`, or `15` |
| `--attention-backend` | `sol_bsa` (default), `sol`, or `dense` |
| `--compute-quant` | `none` (BF16 default) or `mxfp8` (lossy, SM100-family only) |
| `--lora-mode` | `merged` (default), `separate`, or `fused`; branch modes require BF16 compute |
| `--prompt` / `--prompt-file` | Prompt text or UTF-8 prompt file |
| `--image` | First frame for `i2v` |
| `--reference` | Ordered `image:PATH`, `video:PATH`, or `audio:PATH` for `ref2va`; repeat as needed |
| `--reference-image-resize-mode` | Default: the profile's short edge (`short_edge`) if it sets one, else `match` (fast); `diffusers` is the official 2048 mode |
| `--seed` | Random seed |
| `--output` | Output MP4 path |
| `--warmup` | Warm up the selected duration and prompt before generation |

For a service, keep all worker processes resident and warm up every duration used in production.

## Python API

```python
from h3_runtime import MiniMaxH3Inference

with MiniMaxH3Inference(MODEL_PATH, ADAPTER_PATH) as engine:
    result = engine.generate(prompt, duration=5, seed=1)
    if result is not None:
        result.save("output.mp4")
```

Ref2VA uses the same API with an explicit task and the Diffusers reference type:

```python
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from h3_runtime import MiniMaxH3Inference

references = [MiniMaxH3Reference(image="subject.png")]
with MiniMaxH3Inference(MODEL_PATH, REF2VA_ADAPTER_PATH, task="ref2va") as engine:
    result = engine.generate(prompt, duration=10, references=references)
    if result is not None:
        result.save("output.mp4")
```

## Resident T2V benchmark

`benchmark.py` loads the four-step adapter once, warms each requested duration,
and saves videos and completed measurements to `benchmark.json` after each
request. Launch it from this directory in the environment above:

```bash
torchrun --standalone --nproc_per_node=4 benchmark.py \
  --model ./checkpoints/MiniMax-H3 \
  --adapter ./checkpoints/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors \
  --attention-backend dense --durations 5 10 --warmups 2 --repeats 3 \
  --seed 42 --expect-gpu H200 --output-dir ./outputs/h200-4gpu-dense
```

Use a new output directory for every run. The default prompt is the repository's
structured `demo_prompt.json`; `--prompt-file` also accepts plain UTF-8 text.
The report separates model initialization, warmup, warm inference, and completion
through MP4 writing. It records every rank's GPU model, peak PyTorch allocated and
reserved memory, and denoiser invocation count; a request must execute exactly
four transformer forwards and produce 1344×768/24-fps video with 32-kHz stereo
audio. Warmups are excluded from throughput and GPU-seconds-per-clip summaries.
MP4 validation and telemetry are outside the timed interval. The report retains
completed measurements if a later request fails; restarting in the same output
directory is intentionally rejected.

H100 and H200 both use compute capability 9.0. `--attention-backend sol` requires
the `cute_sm90` kernel on these GPUs. `sol_bsa` instead uses the separate cuDNN BSA
path. Direct Sol disables compile buckets by default outside SM100/SM103 because
its other kernels do not implement padded-token masking. Leave
`H3_SOL_COMPILE_BUCKET_SIZE` unset when changing GPU architectures; BSA retains
its independent 4096-token bucket default.

Both sparse modes run the existing dense-reference kernel check during
warmup and require sparse calls on every rank. The check validates kernel
arithmetic with all blocks selected; it does not establish perceptual equivalence
of the sparse routing policy or of the distilled model. The benchmark fixes DiT
compute to BF16 (`compute_quant=none`); the optional MXFP8 compute mode is limited
to SM100-family GPUs.

For eight GPUs, change `--nproc_per_node` to `8`. Run each worker within one node;
compare GPU-seconds per completed clip as well as latency when choosing between
two four-GPU workers and one eight-GPU worker. Record measured H200 results
separately from the B300 results above.

For the measured two-H200 profile, use `--nproc_per_node=2 --attention-backend sol`
and export `PYTORCH_ALLOC_CONF=expandable_segments:True` before `torchrun`. For
one H200, use `--nproc_per_node=1 --attention-backend dense`.
