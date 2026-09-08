# Inference plane design (`specforge.inference`)

The inference plane is an external-server transport boundary. Online training
does not load or execute a target model in a trainer or producer process.

## Responsibility

`SGLangServerCaptureAdapter` sends model inputs and capture metadata to a
patched SGLang server. The server performs prefill and writes captured tensors
directly to Mooncake. Its response contains only sample ids and feature
key/shape/dtype metadata. The adapter validates those feature specifications,
adopts the server-written objects into the producer store, and returns
`SampleRef`s for `RolloutWorker` to commit.

Algorithm registrations own the requested capture layout, target
representation, collator, and optional modality-specific `ServerInputAdapter`.
The transport owns sampling and capture metadata. This keeps target execution,
algorithm policy, and deployment wiring separate.

```mermaid
flowchart LR
  P["PromptTask lease"] --> I["ServerInputAdapter inputs"]
  I --> A["SGLangServerCaptureAdapter"]
  A --> S["patched SGLang /generate"]
  S --> M["Mooncake tensor writes"]
  S --> V["feature metadata response"]
  V --> C["verify_capture_specs"]
  C --> R["adopt SampleRef"]
  R --> W["RolloutWorker commit"]
```

`RolloutWorker` treats this adapter as a `RefSource`: it leases tasks, calls
`produce_refs`, and commits only metadata. Per-task server failures remain
retryable without failing successful tasks from the same batch. Tensors never
pass through the controller or the producer process.

## Offline capture

The separate `specforge.offline_capture` package is used only by
`scripts/prepare_hidden_states.py`. It contains the version-pinned local SGLang
internals needed to generate precomputed EAGLE3 features. It is not imported by
online application or training assembly.

## Endpoints

| From | Endpoint | Plane |
|---|---|---|
| `RolloutWorker` | `DataFlowController.lease_prompt_tasks` | control |
| `RolloutWorker` | `SGLangServerCaptureAdapter.produce_refs` | inference |
| `SGLangServerCaptureAdapter` | patched SGLang `/generate` | external compute |
| patched SGLang | Mooncake feature keys | data |
| `SGLangServerCaptureAdapter` | `MooncakeFeatureStore.adopt` | data |
| `RolloutWorker` | `DataFlowController.commit_samples` | control |
| producer | `StreamingRefChannel.publish` | data |

## Target rendering and parity gates

`LingRenderer` is the small target-specific boundary used before a prompt is
turned into a `PromptTask`. If a tokenizer provides `apply_chat_template`, its
rendered string is authoritative; otherwise the renderer uses the registered
Ling fallback (`HUMAN`/`ASSISTANT` role markers and `role_end`). The renderer
always tokenizes with `add_special_tokens=False` and returns an input-id/loss
mask pair of equal length. Invalid messages, ambiguous tokenizer batches, and
non-string content fail at this boundary rather than producing a silently
misaligned training sample.

The dependency-free `specforge.inference.parity` helpers are intended for CI
and GPU launch gates. Feature manifests compare schema/model versions, capture
layer IDs, feature names, shapes, dtypes, and optional SHA-256 payload hashes;
token checks report the first differing position; state checks compare a
canonical JSON snapshot. `specforge validate ...` returns 1 on a parity
mismatch and 2 for unreadable/invalid input, so a shell gate cannot accidentally
continue after a failed comparison.

These checks prove transport and representation parity, not numerical equality
of two model implementations. Numerical/serving parity still requires the
opt-in SGLang GPU gate in `tests/test_runtime/test_server_capture_gate.py`.
