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

### Access the deployed H200 service

The deployment is on the Fukushima H200 cluster. These are the connection
details for another client or Codex session; the `10.2.201.*` addresses are
private and require cluster network access or an SSH tunnel.

| Item | Value |
|---|---|
| Public SSH login / gateway | `157.66.255.187`, port **44222** (`VM-187-fukushimadc`) |
| Existing deployment owner | `jeff` |
| Jeff's key on this workstation | `/home/jeff/.ssh/gmi` |
| Shared service token file on the cluster | `/home/jeff/sol-h3-h200/deployment/token` |
| Node `fukushimadc-02-hgx-0002` | `10.2.201.2`, ports `18080`–`18083` |
| Node `fukushimadc-02-hgx-0006` | `10.2.201.6`, ports `18080`–`18083` |
| Node `fukushimadc-02-hgx-0023` | `10.2.201.23`, ports `18080`–`18083` |
| Shared deployment code and logs | `/home/jeff/sol-h3-h200` |

The three service Slurm jobs are `3377`, `3374`, and `3380`, respectively.
Check `/healthz` before using a worker. Each worker uses two H200 GPUs and
accepts 5- or 10-second text-to-video requests at 1344x768 / 24 fps with audio.
The model is **H3-Base FL2VA plus the four-step FastH3 adapter**.

Log in from this workstation, then query the private endpoint:

```bash
ssh -i ~/.ssh/gmi -p 44222 jeff@157.66.255.187
# On the cluster login host:
curl --noproxy '*' --fail http://10.2.201.2:18080/healthz
```

To call it from another machine, run this tunnel on the **client machine** and
keep the terminal open:

```bash
ssh -i ~/.ssh/gmi -p 44222 -N \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:28080:10.2.201.2:18080 jeff@157.66.255.187
```

Then use `http://127.0.0.1:28080` as the base URL on that machine. Add more `-L`
forwards with distinct local ports to reach additional workers. A tunnel on a
login host does not make that host's loopback address available to Ray actors on
other compute hosts. Run forwarding on each client host or provide network routing.
Other people need authorized SSH access and the service bearer token; the listed
key and private token file belong to Jeff's deployment account.

This complete example runs on the cluster login host with only Python 3. It
submits a request, waits for it, and downloads `h3-example.mp4`:

```bash
python3 - <<'PY'
import json
import time
import urllib.request
from pathlib import Path

base = "http://10.2.201.2:18080"
token = Path("/home/jeff/sol-h3-h200/deployment/token").read_text().strip()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with opener.open(urllib.request.Request(base + path, data=data, headers=headers), timeout=60) as response:
        return response.read()

job = json.loads(request("/v1/videos", {
    "prompt": "A red fox walks through a sunlit forest. Birds chirp and leaves rustle.",
    "duration": 10,
    "seed": 42,
}))
print("Job:", job["id"])
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    state = json.loads(request(job["status_url"]))
    if state["status"] == "failed":
        raise RuntimeError(state.get("error", "Generation failed"))
    if state["status"] == "succeeded":
        Path("h3-example.mp4").write_bytes(request(job["video_url"]))
        print("Saved h3-example.mp4")
        break
    time.sleep(1)
else:
    raise TimeoutError(f"Still pending; resume polling {base}{job['status_url']}")
PY
```

For a tunneled client, change `base` and use a protected local copy of the token
file. Keep status and download requests on the same worker that accepted the job.
HTTP 401 means the token is missing/incorrect, 422 means invalid parameters,
429 means the queue is full, and 503 means the worker is starting or unavailable.
The token is not needed for `/healthz`.

For verl, see the **Self-hosted H3 on H200** section of
`/home/jeff/verl/examples/prompt_rewrite/README.md`. Its new-run options are
`--h3-backend self-hosted`, `--h3-endpoints <comma-separated worker URLs>`, and
`--h3-token-file <shared token filename>`. The policy can continue calling
`h3_max(prompt)`; the adapter performs submission, polling, download, and media
validation against this service.

### Launch another service

`serve.py` keeps one distributed worker loaded and serves authenticated, queued
T2V requests. This service currently accepts the H200-tested **5- and 10-second**
profiles only, at 1344x768 / 24 FPS with stereo audio. Install the optional HTTP
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
| `POST /v1/videos` | Accept `{"prompt":"…","duration":5,"seed":42}`; return 202 with a job ID and relative status/video URLs |
| `GET /v1/jobs/{id}` | Return `queued`, `running`, `succeeded`, or `failed`, with timings after completion |
| `GET /v1/jobs/{id}/video` | Download the completed MP4; return 409 while unfinished |

Each worker handles one generation at a time and queues up to eight waiting
requests; a full queue returns 429. Jobs and videos remain in that worker's output
directory across restarts. Interrupted requests are marked failed, so callers can
decide whether to resubmit. Keep polling/downloading from the worker that accepted
the request. The service validates four transformer forwards and audio/video
stream metadata before marking a request successful. A failed distributed
generation terminates the process group instead of reusing inconsistent GPU state.

`deploy_h200_node.sh` and `serve_node.py` provide the Slurm node deployment: four
independent two-GPU workers, ports 18080–18083, eight CPU encoder threads per
worker, and separate CPU affinities. They bind to the node's internal IP and
supervise each process group with up to two restarts. The launcher defaults to
`/home/jeff/sol-h3-h200` for shared code/registry and `/raid/jeff` for environments,
checkpoints, caches, and outputs. It expects a private `deployment/token` and can
initialize a fresh node from a checksummed `deployment/staging` snapshot. The
staged venv must use the same absolute path on every node; `H3_DEPLOY_ROOT` and
`H3_LOCAL_ROOT` override the site paths when preparing another deployment.

```bash
sbatch --partition=all --nodes=1 --ntasks=1 --cpus-per-task=160 --mem=1600G \
  --gres=gpu:nvidia_h200:8 --constraint=h200 --time=0 \
  --nodelist=YOUR_NODE deploy_h200_node.sh
```

The node registry is `deployment/<hostname>/node.json`; worker logs and readiness
records are beside it. Check live `/healthz` responses when using the registry,
because a hard node failure can leave stale files. Stop a deployment with
`scancel <its-job-id>`. Outputs are retained until explicitly removed; admission
stops when free disk space falls below 20 GiB. The HTTP endpoints are intended for
the private cluster network or an SSH tunnel. Cross-worker routing and integration
with a training client are separate from this service.

### Concurrent H200 validation

On 2026-10-22, two eight-H200 nodes ran four two-GPU workers each. All **64 HTTP
generation requests passed**: three simultaneous eight-request rounds per
duration using the benchmark prompt, then one new prompt at each duration on
every worker. Each worker completed two warmups per duration before measurement.

| Nominal clip duration | Warm samples | Service processing median | Client completion median |
|---|---:|---:|---:|
| 5 s | 24 | 12.94 s | 13.08 s |
| 10 s | 24 | 31.40 s | 31.53 s |

Processing includes text encoding, four transformer forwards, video/audio decode,
MP4 encoding, and stream metadata validation. Client time also includes HTTP
submission and polling, but excludes downloading the MP4. Model loading and
warmups are excluded. The first requests for the two new prompt lengths took
25.31 s / 43.34 s median for 5 s / 10 s clips, including first-use compilation and
correctness checks; they are excluded from the warm table. All ranks passed their
sparse numerical gates and executed four transformer forwards. Four samples from
the two nodes were fully decoded with nonzero image variation and audio energy.
This validates the four-worker deployment on two nodes; a third node was queued
because the other six registered H200 nodes had all eight GPUs allocated.

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
checks below. H200 I2V/Ref2VA and the reachable SM120 paths remain unvalidated.

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

Ref2VA accepts image, video, and audio references. Audio cannot be the only reference. Use `sol_bsa` (default) or `dense` for this task. Reference images use the validated fast `match` profile by default; pass `--reference-image-resize-mode diffusers` only when official 2048-short-edge preprocessing parity is required.

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
| `--reference-image-resize-mode` | `match` (fast default) or `diffusers` (official 2048 mode) |
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
