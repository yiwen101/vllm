# Probing Workbench

This directory contains personal diagnostic scripts for investigating vLLM model
support. These scripts are not production vLLM code and are not intended to be
included in an upstream feature PR as-is.

The main use cases are:

- inspect what a HuggingFace repository exposes;
- check whether a model requires remote code;
- inspect config, processor, tokenizer, and file layout;
- collect evidence before deciding between vLLM's Transformers backend and a
  native vLLM model implementation.

## Phase 1 Scripts

### `probe_hf_metadata.py`

Inspects metadata through HuggingFace Transformers without loading model
weights.

Example:

```bash
.venv/bin/python codex/probes/probe_hf_metadata.py \
  --model openvla/openvla-7b \
  --trust-remote-code \
  --json-out logs/openvla_hf_metadata.json \
  2>&1 | tee logs/openvla_hf_metadata.log
```

This answers:

- Does `AutoConfig` load?
- Does `AutoProcessor` load?
- Does `AutoTokenizer` load?
- What are `model_type`, `architectures`, and `auto_map`?
- What processor/tokenizer classes are being used?
- Does the processor appear to expose multimodal-token helper methods?

### `inspect_hf_repo_files.py`

Inspects HuggingFace Hub repository files and small JSON/Python metadata files
without downloading large model weights.

Example:

```bash
.venv/bin/python codex/probes/inspect_hf_repo_files.py \
  --model openvla/openvla-7b \
  --json-out logs/openvla_repo_files.json \
  2>&1 | tee logs/openvla_repo_files.log
```

This answers:

- Which files exist in the model repository?
- Which remote-code Python files exist?
- How large are the weight shards?
- What top-level keys appear in `config.json` and other small metadata files?

## Logging Pattern

Use `tee` so terminal output is both visible and saved:

```bash
mkdir -p logs

.venv/bin/python codex/probes/probe_hf_metadata.py \
  --model openvla/openvla-7b \
  --trust-remote-code \
  2>&1 | tee logs/openvla_hf_metadata.log
```

When asking for help, usually paste:

```bash
tail -n 120 logs/openvla_hf_metadata.log
```

For error locations:

```bash
grep -n "Traceback\|Error\|Exception\|RuntimeError\|ValueError" logs/*.log
```

## Scope

Do not add production registry/model/config changes here. Keep the workbench
isolated so it can be carried across future investigations without becoming
part of the actual vLLM contribution.
