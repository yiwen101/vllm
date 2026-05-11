# GPU Probing Workflow

This guide is for running OpenVLA/vLLM probes on a rented GPU server over SSH.
It assumes you have already rented a CUDA GPU machine and can SSH into it.

## 1. Recommended Server Shape

For OpenVLA probing:

- GPU: RTX 4090 24GB minimum, 48GB GPU more comfortable
- Disk: 150GB minimum, 250GB safer
- RAM: 48GB+
- OS/template: Ubuntu + CUDA + PyTorch is fine

The OpenVLA weights are about 14GB, but caches, dependencies, logs, and retries
need additional disk.

## 2. First SSH Session

Use `tmux` so work keeps running if your SSH terminal disconnects:

```bash
tmux new -s openvla
```

If you disconnect, reconnect and run:

```bash
tmux attach -t openvla
```

## 3. Clone Your Fork On The GPU Server

Example:

```bash
export REPO_URL="https://github.com/yiwen101/vllm.git"
export BRANCH="probe-workbench"
export REPO_DIR="/workspace/vllm"

cd /workspace
git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
```

For later updates after you force-push from your laptop:

```bash
cd /workspace/vllm
git fetch origin probe-workbench
git checkout probe-workbench
git reset --hard origin/probe-workbench
```

The reset is intentional for this personal workbench branch. Do not use it on a
branch with remote-only changes you care about.

## 4. Set Hugging Face Token

Set the token in the GPU shell before running probes:

```bash
export HF_TOKEN="<paste_huggingface_token_here>"
```

Check without printing the token:

```bash
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN set" || echo "HF_TOKEN missing"
```

If you use `tmux`, set `HF_TOKEN` inside the tmux session that runs the probes.

## 5. Set Up vLLM Environment

On the GPU server:

```bash
cd /workspace/vllm
bash codex/gpu/setup_vllm_gpu.sh
```

This script:

- installs `uv` if missing;
- creates `.venv` with Python 3.12;
- installs lint tooling;
- installs vLLM editably with precompiled native extensions;
- installs extra OpenVLA probing dependencies, including
  `timm>=0.9.10,<1.0.0` because OpenVLA's remote HF code rejects TIMM 1.x.

If the precompiled wheel is temporarily unavailable for the current commit/CUDA
variant, the setup may fail. Save the log and ask for help with the exact error.

## 6. Run OpenVLA Probes

Run the full GPU-side probing sequence:

```bash
cd /workspace/vllm
test -x .venv/bin/python && echo "venv ok" || echo "venv missing"
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN set" || echo "HF_TOKEN missing"
bash codex/gpu/run_openvla_gpu_checks.sh
```

The script runs:

1. environment summary;
2. HF metadata probe;
3. HF repo file inspection;
4. direct HuggingFace OpenVLA `predict_action` probe;
5. vLLM `model_impl="transformers"` generation probe.

Each step writes logs under:

```text
logs/
```

No local config file needs to be edited. The probes read the model metadata
directly from HuggingFace.

## 7. Viewing Output Over SSH

Do not rely on terminal scrollback. Use logs.

Show the latest lines of a log:

```bash
tail -n 120 logs/SOME_LOG.log
```

Follow a running log:

```bash
tail -f logs/SOME_LOG.log
```

Search for errors:

```bash
grep -n "Traceback\|Error\|Exception\|RuntimeError\|ValueError" logs/*.log
```

Show a specific line range:

```bash
sed -n '200,280p' logs/SOME_LOG.log
```

List newest logs:

```bash
ls -lt logs | head
```

## 8. Mirror Logs To Your Laptop

From your laptop, not inside SSH:

```bash
bash codex/gpu/fetch_remote_logs.sh
```

This overwrites the local mirror:

```text
logs/remote_gpu/
```

The script uses these defaults for the current rented server:

```bash
GPU_USER=root
GPU_HOST=175.155.64.225
GPU_PORT=19765
REMOTE_REPO=/workspace/vllm
```

For the next server, override the endpoint inline:

```bash
GPU_HOST=<GPU_HOST> GPU_PORT=<GPU_PORT> bash codex/gpu/fetch_remote_logs.sh
```

## 9. What To Paste For Help

Usually paste:

```bash
tail -n 120 logs/openvla_vllm_transformers_*.log
```

Also paste:

```bash
grep -n "Traceback\|Error\|Exception\|RuntimeError\|ValueError" logs/*.log
```

Do not paste the whole log unless asked.
