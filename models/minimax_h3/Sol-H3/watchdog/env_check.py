"""Compare live workers with what watchdog/run-worker.sh would start for the same node, port, and launch settings.

Run inside the node's allocation:
  srun --jobid=<job> --overlap --nodes=1 --ntasks=1 python3 env_check.py <run-worker.sh>
For each port 18080-18083 it reads the local-rank-0 process of the live worker and the H3_RENDER_ONLY=1 output of
run-worker.sh, given the launch settings in the node's registry, as watchdog.py passes them (none for allocations
started before they existed). Every variable run-worker.sh sets must have the live value (PATH and LD_LIBRARY_PATH:
as a prefix; CUDA_VISIBLE_DEVICES: run-worker.sh prints ordinals, live workers carry the pair's UUIDs), every live
H3_/PyTorch/NCCL/cache variable must be set by run-worker.sh, the serve.py arguments must match except --ready-file,
and the live working directory must be the code directory run-worker.sh would use. Exit status 1 on any mismatch.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

DEPLOY = Path("/home/jeff/sol-h3-h200")
PREFIXES = ("H3_", "PYTORCH_", "NCCL_", "GLOO_", "TORCHINDUCTOR_", "TRITON_", "CUDA_CACHE_PATH", "CUDA_HOME", "HF_",
            "TOKENIZERS_", "OMP_", "CUDNN_PATH", "PYTHONUNBUFFERED")


def without_ready(args):
    out = list(args)
    if "--ready-file" in out:
        index = out.index("--ready-file")
        del out[index:index + 2]
    return out


def live_worker(port):
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().split("\0")
            environ = dict(item.split("=", 1) for item in (proc / "environ").read_bytes().decode().split("\0") if "=" in item)
            cwd = os.readlink(proc / "cwd")
        except OSError:
            continue
        if any(arg.endswith("serve.py") for arg in argv) and "--port" in argv and argv[argv.index("--port") + 1] == str(port) \
                and environ.get("LOCAL_RANK") == "0":
            return argv, environ, cwd
    return None


def main():
    script = sys.argv[1]
    hostname = socket.gethostname()
    node = hostname.split(".")[0].rsplit("-", 1)[1]  # dc02-gpu-003.internal... -> 003
    registry = json.loads((DEPLOY / "deployment" / hostname / "node.json").read_text())
    launch = registry.get("launch")
    settings = [] if launch is None else [str(int(launch["async_publication"])), launch["code_dir"]]
    report, ok = {}, True
    if registry.get("slurm_job_id") != os.environ.get("SLURM_JOB_ID"):
        ok = False
        report["registry"] = f"registry job {registry.get('slurm_job_id')} is not this job {os.environ.get('SLURM_JOB_ID')}"
    for port in range(18080, 18084):
        lines = subprocess.run(["bash", script, node, str(port), *settings],
                               env={"H3_RENDER_ONLY": "1", "PATH": "/usr/bin:/bin",
                                    "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "")},
                               capture_output=True, text=True, check=True).stdout.splitlines()
        split = lines.index("--")
        rendered = dict(line.split("=", 1) for line in lines[:split])
        rendered.pop("CPUS", None)
        code_dir = rendered.pop("CWD", None)
        rendered_args = lines[split + 1:]
        live = live_worker(port)
        if live is None:
            report[port] = {"ok": False, "error": "no live local-rank-0 process"}
            ok = False
            continue
        argv, environ, cwd = live
        problems = []
        for key, value in rendered.items():
            if key in ("HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "CUDA_VISIBLE_DEVICES"):
                continue
            if key in ("PATH", "LD_LIBRARY_PATH"):
                if not environ.get(key, "").startswith(value.split(":")[0]):
                    problems.append(f"{key}: live {environ.get(key)!r} does not start with {value.split(':')[0]!r}")
            elif environ.get(key) != value:
                problems.append(f"{key}: live {environ.get(key)!r}, run-worker.sh {value!r}")
        for key, value in environ.items():
            if key.startswith(PREFIXES) and key not in rendered:
                problems.append(f"{key}={value!r} is live but not set by run-worker.sh")
        live_args = [a for a in argv[argv.index(next(a for a in argv if a.endswith("serve.py"))) + 1:] if a]
        if without_ready(live_args) != without_ready(rendered_args):
            problems.append(f"arguments differ: live {without_ready(live_args)} vs run-worker.sh {without_ready(rendered_args)}")
        if code_dir is not None and os.path.realpath(cwd) != os.path.realpath(code_dir):
            problems.append(f"code directory: live {cwd!r}, run-worker.sh {code_dir!r}")
        report[port] = {"ok": not problems, "problems": problems, "code_dir": cwd,
                        "publication_mode": "async" if "--async-publication" in live_args else "sync",
                        "serving_memory_fraction": environ.get("H3_SERVING_MEMORY_FRACTION")}
        ok &= not problems
    print(json.dumps({"node": node, "ok": ok, "launch": launch, "ports": report}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
