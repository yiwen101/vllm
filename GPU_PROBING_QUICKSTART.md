# GPU Probing Quickstart

This branch is a personal probing workbench. It is not intended for an upstream
vLLM PR.

The intended workflow is:

1. edit this branch locally;
2. force-push it to your fork;
3. SSH into a rented GPU server;
4. clone or hard-reset the GPU checkout to this branch;
5. run the GPU probes;
6. fetch/paste logs back for diagnosis.

## Fixed Values For This Workbench

These values are already known for this branch:

```bash
export REPO_URL="https://github.com/yiwen101/vllm.git"
export BRANCH="probe-workbench"
export REPO_DIR="/workspace/vllm"
```

For each rented GPU instance, fill in only the SSH endpoint:

```bash
export GPU_HOST="<GPU_HOST>"
export GPU_PORT="<GPU_SSH_PORT>"
export GPU_USER="root"
```

If Vast gives you a full SSH command with `-p` or `-i`, use that command
manually for SSH. The `GPU_SSH` variable is only for simple `user@host` cases.

## Local: Push Your Current Workbench Branch

From your laptop, inside this repo:

```bash
git checkout probe-workbench
bash codex/gpu/local_force_push_branch.sh
```

This uses `--force-with-lease`, which is safer than plain `--force`.

For this personal branch, if you intentionally rewrote the single workbench
commit, plain force-push is acceptable:

```bash
git push --force origin probe-workbench
```

## GPU: Clone Or Force-Reset To This Branch

SSH into the GPU server. For a custom port:

```bash
ssh -p "$GPU_PORT" "$GPU_USER@$GPU_HOST" -L 8080:localhost:8080
```

Or paste your provider's SSH command.

Then on the GPU server:

```bash
export REPO_URL="https://github.com/yiwen101/vllm.git"
export BRANCH="probe-workbench"
export REPO_DIR="/workspace/vllm"

mkdir -p /workspace
cd /workspace

# If this script is not available yet because the repo is not cloned,
# run the plain git clone first:
git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
```

After the first clone, future updates can be:

```bash
cd /workspace/vllm
bash codex/gpu/remote_reset_current_repo.sh
```

If you prefer the explicit commands:

```bash
cd /workspace/vllm
git fetch origin probe-workbench
git checkout probe-workbench
git reset --hard origin/probe-workbench
```

Alternative one-command clone/reset helper:

```bash
export REPO_URL="https://github.com/yiwen101/vllm.git"
export BRANCH="probe-workbench"
export REPO_DIR="/workspace/vllm"
bash codex/gpu/remote_clone_or_reset_branch.sh
```

## GPU: Set Hugging Face Token

Do not commit tokens into git. Set the token only in the shell that will run the
probe:

```bash
export HF_TOKEN="<paste_huggingface_token_here>"
```

Check that it is present without printing it:

```bash
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN set" || echo "HF_TOKEN missing"
```

If you start a new SSH session or a new `tmux` session, set the token again
unless you wrote it into a private shell profile on the rented server.

## GPU: Set Up vLLM

On the GPU server:

```bash
cd /workspace/vllm
bash codex/gpu/setup_vllm_gpu.sh
```

This creates `.venv`, installs vLLM editably using precompiled native
extensions, and installs extra OpenVLA probing dependencies. It pins
`timm>=0.9.10,<1.0.0` because OpenVLA's remote HF code rejects TIMM 1.x.

## GPU: Run OpenVLA Checks

```bash
cd /workspace/vllm
test -x .venv/bin/python && echo "venv ok" || echo "venv missing"
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN set" || echo "HF_TOKEN missing"
bash codex/gpu/run_openvla_gpu_checks.sh
```

Logs are written to:

```text
/workspace/vllm/logs/
```

## Viewing Output On SSH

Do not rely on terminal scrollback.

List newest logs:

```bash
ls -lt logs | head
```

View the last part of a log:

```bash
tail -n 120 logs/<LOG_FILE>
```

Follow a running log:

```bash
tail -f logs/<LOG_FILE>
```

Find errors:

```bash
grep -n "Traceback\|Error\|Exception\|RuntimeError\|ValueError" logs/*.log
```

Show a specific range:

```bash
sed -n '200,280p' logs/<LOG_FILE>
```

## Laptop: Mirror Logs From GPU

From your laptop, inside this local repo:

```bash
bash codex/gpu/fetch_remote_logs.sh
```

This overwrites the local mirror at:

```text
logs/remote_gpu/
```

Current defaults are set for the active rented server:

```bash
GPU_USER=root
GPU_HOST=175.155.64.225
GPU_PORT=19765
REMOTE_REPO=/workspace/vllm
```

For a future rented server, override only the changed values:

```bash
GPU_HOST=<GPU_HOST> GPU_PORT=<GPU_PORT> bash codex/gpu/fetch_remote_logs.sh
```

## What To Send Back For Help

Usually send:

```bash
tail -n 120 logs/openvla_vllm_transformers_*.log
```

and:

```bash
grep -n "Traceback\|Error\|Exception\|RuntimeError\|ValueError" logs/*.log
```

Also include the exact command you ran.
