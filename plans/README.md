# Draftfit plans

This is a roadmap, not a support matrix or a release commitment. Draftfit 0.2.0
ships the existing main-branch training workflow. The experiments below are
excluded from this release; see [current support](../docs/PUBLIC_SUPPORT.md).

## Recent targets: DSpark and DFlash2

| Target | Planned experiment | Boundary to validate |
| --- | --- | --- |
| [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | New drafts and further training of the [official DSpark draft](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark) | Checkpoint/config compatibility, real-weight load and export |
| [Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B) | New DSpark and DFlash2 drafts | MoE capture, expanded hidden-state layout and rotary settings |
| [ZDTaichu5.0-9B](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B) | Text-only drafts; assess official draft conversion separately | Nested text config and official checkpoint semantics |
| [Hemmingway-1](https://huggingface.co/Altworld/Hemmingway-1) | Text-only DSpark and DFlash2 drafts | Hybrid capture and cache/state correctness |
| [Agnes-3.0-Flash](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash) | Text-only DSpark and DFlash2 drafts | Custom text backend and hybrid state |

For each target, pin weights/tokenizer/config revisions, review its license,
verify chat masks and feature semantics, then exercise real-weight capture,
training, resume, export and compatible serving. Metadata inspection and tiny
model tests alone are not model support. Multimodal inputs and reuse of MTP
heads are separate work, not implicit capabilities of these plans.

Prefer a bounded one-GPU offline experiment when target weights fit. Reuse
captured features between methods only when their feature contracts match.
Start with a small training/holdout smoke before choosing a larger data budget;
neither 50k examples nor 5–10k warm-start examples guarantees improvement.
Long runs need storage-aware capture and recovery, not an unbounded feature dump.

## Personalized inference

Reviewed conversation exports can already enter the training workflow.
Automatic trace ingestion, privacy filtering, session/time-aware evaluation and
continual updates are future work. The next product milestone is a reproducible
before/after benchmark on a held-out personal workload: identical target,
hardware, requests and generation settings, with correctness checked separately.

No speedup claim or automatic model promotion is implied by this roadmap.
