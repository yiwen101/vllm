# OpenVLA Transformers Compatibility Findings

This note summarizes the bounded GPU investigation done on the personal
`probe-workbench` branch for `openvla/openvla-7b`. It is intended as supporting
evidence for deciding whether OpenVLA can be integrated through vLLM's generic
Transformers backend, or whether it needs explicit vLLM-side support.

## Environment Tested

The probe ran on the rented GPU server using current vLLM main dependencies:

- Python: 3.12.13
- vLLM: `0.20.2rc1.dev189+g4ed0a718d`
- PyTorch: `2.11.0+cu129`
- Transformers: `5.8.0`
- tokenizers: `0.22.2`
- GPU: RTX 4090-class CUDA GPU
- TIMM was pinned to `0.9.16` for the OpenVLA probe, because OpenVLA's own
  remote code rejects TIMM 1.x.

Relevant mirrored logs are under `logs/remote_gpu/` in the workbench checkout.

## What Loaded Successfully

The lightweight Hugging Face metadata checks succeeded:

- `AutoConfig` loaded with `model_type = "openvla"`.
- `architectures = ["OpenVLAForActionPrediction"]`.
- `auto_map` points to remote code:
  - `AutoConfig -> configuration_prismatic.OpenVLAConfig`
  - `AutoModelForVision2Seq -> modeling_prismatic.OpenVLAForActionPrediction`
- `AutoProcessor` loaded as `PrismaticProcessor`.
- The HF repo contains remote-code files:
  - `configuration_prismatic.py`
  - `modeling_prismatic.py`
  - `processing_prismatic.py`

The repo metadata also confirms OpenVLA is not a plain text model:

- `vision_backbone_id = "dinosiglip-vit-so-224px"`
- `llm_backbone_id = "llama2-7b-pure"`
- `use_fused_vision_backbone = true`
- config includes TIMM-related fields such as `timm_model_ids`

## vLLM-Relevant Processor Finding

`PrismaticProcessor` does not expose the multimodal token-count helpers checked
by the probe:

- `_get_num_multimodal_tokens`: absent
- `get_num_multimodal_tokens`: absent
- `_get_num_image_tokens`: absent
- `get_num_image_tokens`: absent

This is directly relevant to vLLM integration. vLLM needs to know how many
placeholder/multimodal tokens an input image contributes so it can perform
prompt validation, batching, cache planning, and multimodal input alignment.

This finding supports the earlier PR author's concern that a generic
Transformers-backend path may not be enough without OpenVLA-specific processor
or token-accounting logic.

## Raw Hugging Face Compatibility Findings

Before testing vLLM's generic Transformers backend meaningfully, the direct
Hugging Face OpenVLA baseline itself ran into several compatibility issues under
the current vLLM dependency set.

The OpenVLA remote code expects older dependencies. During model loading it
prints:

```text
Expected `transformers==4.40.1` and `tokenizers==0.19.1`
```

but the current vLLM environment uses:

```text
transformers==5.8.0
tokenizers==0.22.2
```

Observed raw-HF compatibility breaks included:

- old OpenVLA processor imports tokenization types from an older Transformers
  module path;
- OpenVLA advertises `AutoModelForVision2Seq`, while the tested Transformers
  version no longer exposes the same auto-class path in the expected way;
- newer Transformers expects `_supports_sdpa`;
- newer Transformers calls `tie_weights(recompute_mapping=...)`, but OpenVLA's
  remote `tie_weights` method has the older signature;
- newer Transformers no longer gives every `PreTrainedModel` `.generate()` for
  free through base-class inheritance;
- after manually adding GenerationMixin behavior for probing, generation then
  required `generation_config`;
- after adding `generation_config`, generation advanced further but failed on a
  deeper signature mismatch:

```text
TypeError: GenerationMixin._expand_inputs_for_generation() got multiple values
for argument 'expand_size'
```

These patches were made only inside diagnostic probe scripts, not vLLM source.
They were used to determine whether the baseline could be reached, not as a
proposed integration approach.

## Interpretation

The investigation indicates two separate concerns:

1. OpenVLA's own HF remote code is not directly compatible with the current
   Transformers version used by vLLM main.
2. Independent of that dependency drift, OpenVLA's processor/model structure
   exposes custom multimodal behavior that vLLM likely needs to understand
   explicitly.

Therefore, a simple "register OpenVLA and rely on the generic Transformers
backend" approach is risky. It may fail before vLLM reaches its own multimodal
runtime path, and even if HF compatibility is patched, vLLM still needs explicit
answers for image preprocessing, multimodal token accounting, and action-token
postprocessing.

## Design Implication

For a short-term OpenVLA PR, the safer design direction is:

- make OpenVLA-specific behavior explicit rather than relying on HF remote-code
  side effects;
- define or adapt the processor path so vLLM can determine multimodal token
  counts deterministically;
- implement or reuse the correct fused DINOv2 + SigLIP preprocessing path;
- load the model in a way compatible with vLLM's model execution contracts;
- add a direct comparison test against a known-good Hugging Face baseline, using
  the dependency versions OpenVLA expects where necessary.

The current probe did not prove that a generic Transformers backend path is
impossible. It did show that, with current vLLM dependencies, that path is not a
straightforward low-risk integration and would need several compatibility and
multimodal-contract fixes before it could be evaluated fairly.

## Proposed Justification For Explicit OpenVLA Support

The justification for an explicit OpenVLA implementation is not simply that
"the Transformers backend failed." The more precise justification is:

OpenVLA combines a custom remote-code model, a custom Prismatic processor, a
fused DINOv2 + SigLIP vision backbone, action-token decoding, and dependency
assumptions that differ from current vLLM main. These are not just registration
details. They affect how vLLM should preprocess images, count multimodal
tokens, align image features with prompt tokens, execute generation, and decode
generated tokens back into robot actions.

The generic Transformers backend is valuable when the Hugging Face model and
processor expose the contracts vLLM expects. For OpenVLA, the probe found that
the processor does not expose token-count helpers such as
`_get_num_multimodal_tokens` or `get_num_multimodal_tokens`. That means vLLM
does not have an obvious generic way to infer the image-token contribution from
the processor alone. For a multimodal model, this is a core runtime contract,
not a cosmetic integration issue.

The probe also found that direct Hugging Face OpenVLA execution does not run
cleanly under current vLLM dependencies. The failures were caused by
Transformers-version drift in OpenVLA's remote code, including moved imports,
changed generation inheritance behavior, changed `tie_weights` calling
conventions, missing `generation_config`, and a later GenerationMixin signature
mismatch. These failures make it hard to treat the current HF remote-code path
as a stable base for vLLM integration.

Therefore, a narrow OpenVLA-specific vLLM path is justified as a pragmatic
short-term solution. It should make the relevant behavior explicit inside vLLM
instead of relying on fragile remote-code side effects. The implementation
should still stay as small as possible and avoid over-generalizing to all VLA
models before the shared abstractions are clear.

The proposed PR should present this as bounded model support, not a broad VLA
architecture refactor. A larger VLA abstraction can be researched separately
after OpenVLA support establishes the concrete processor, preprocessing,
generation, and action-decoding requirements.

## Suggested PR Wording

One possible way to summarize the rationale in a PR or issue comment:

```text
I investigated whether OpenVLA can be supported through vLLM's generic
Transformers backend. The investigation found that OpenVLA's HF repository uses
custom remote code for both the Prismatic model and processor, including a
fused DINOv2 + SigLIP vision stack and action-token decoding. The processor does
not expose the multimodal token-count helpers that vLLM's generic multimodal
path would need for deterministic image-token accounting.

In addition, direct HF execution of OpenVLA does not run cleanly under current
vLLM main dependencies. OpenVLA's remote code expects older Transformers and
tokenizers versions and fails on several newer-Transformers API changes. This
makes the generic remote-code path a fragile base for vLLM support.

For this reason, this PR takes a narrow OpenVLA-specific approach that makes
the required preprocessing, token accounting, model execution, and action
decoding explicit. The goal is to keep the change scoped to OpenVLA first,
while leaving broader VLA abstractions for follow-up design once more model
families have been compared.
```
