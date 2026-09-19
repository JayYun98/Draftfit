# Draftfit for personal inference workloads

**Your workload. Your draft.** Personalized AI needs workload-specific evaluation,
not just a smaller draft.
The product hypothesis is simple: a draft adapted to recurring tasks may predict
the frozen target more efficiently than a general-purpose draft. It must earn its
place by improving measured inference performance, including its own cost.

## Available now

Reviewed OpenAI/ShareGPT text conversations can use the existing data preparation,
draft training, warm-start, export and held-out benchmark workflow in the README.
No OTel collector is required. No automatic agent-log reader, secret scrubber,
continual-learning scheduler or automatic promotion service is implemented.

The existing online training path already connects GPU 0 teacher capture to
GPU 1 draft training through Mooncake. The bounded Ling run is documented in
[model validation](MODEL_VALIDATION.md). Trace ingestion chooses the workload;
Mooncake transfers training features. These are separate responsibilities, not
alternative implementations. A live training producer does not automatically
capture real end-user agent traffic.

The local integration check uses synthetic, independent conversations through the
actual data CLI, then the benchmark with a mocked tokenizer and HTTP responses:

```sh
python -m unittest tests.test_data.test_personal_workflow \
  tests.test_data.test_prepare tests.test_benchmarks.test_sglang_benchmark \
  tests.test_dflash2_integration
```

It checks conversion, disjoint IDs, removal of the held-out reference from the
prompt and dataset fingerprinting. The separate DFlash2 tests check actual CPU
parameter updates and export/reload with synthetic tensors. These are NOT an
end-to-end training run on trace-derived teacher features, a live OTel integration,
or a speed measurement. No private logs or external model downloads are needed.

## Extension checklist

| Component | Reuse / work required | Validation |
| --- | --- | --- |
| Collection | Add an opt-in adapter for an observed OpenCodex export schema or a pinned OTel schema; do not guess either | Sanitized real export fixture; completed/error/streaming/tool events |
| Privacy | Add selection, retention/deletion and pre-export review; exclude credentials and unwanted private content | Secret fixtures, no raw text in error logs, overwrite/failure checks; patterns alone are not a privacy guarantee |
| Dataset boundary | Reuse normalization; add session grouping and temporal cutoff in a sidecar-aware importer | Sessions and overlapping conversation prefixes cannot cross splits; post-cutoff sessions only in holdout |
| Provenance | Record target/tokenizer/template revisions, method, feature taps and dataset digest outside conversation rows | Reject mismatched caches and mixed-target datasets before training |
| Teacher features | Reuse compatible capture backend on the frozen target | Replay reviewed conversations; compare tokens, loss masks and required feature tensors |
| Candidate update | Reuse weights-only warm start; mix selected historical and recent data | Finite updates, resume, recent AND historical holdout; no automatic claim from loss alone |
| Evaluation | Reuse text benchmark; add paired report and per-request timing if tail latency is required | Target-only vs base draft vs personalized draft, fixed workload/runtime/settings, repeated runs |
| Promotion | Start with explicit artifact selection; automate only after gates are proven | Exact export reload, correctness, performance and rollback to prior artifact |

The current random split groups identical prompt contexts; it is NOT a session-
aware or temporal split. For real longitudinal agent data, externally partition
reviewed complete sessions at a time cutoff and prepare each partition separately
without `--split-eval`. Remove overlap before use. Keep the final evaluation set
out of candidate selection and subsequent updates; use a separate validation set.

The current benchmark rejects tool schemas/calls. Do not silently strip tools to
make an agent trace pass: select genuinely text-only conversations or implement
and validate the tool-rendering contract first. Truncated or failed turns are not
automatically suitable training examples. Provider outputs from a different model
must not be mislabeled as the chosen target's outputs.

## First real experiment

1. Select one supported, pinned target and one recurring text-only task.
2. Review/export completed conversations and partition by session/time. Preserve
   original rendering semantics; record any deliberate transformations.
3. Capture target features for the training partition. Text input/output alone
   does not replace the hidden features required by a draft method.
4. Fine-tune a candidate from the existing draft, retaining an immutable baseline.
5. Compare all three inference modes on unseen sessions using identical hardware,
   generation limits, concurrency, tokenizer/template and cache policy. Check
   output correctness separately. Count teacher capture and training cost too.
6. Apply only if measured improvement clears a predeclared margin across repeated
   runs without correctness or historical-workload regression. Otherwise keep the
   baseline. Do not infer improvement from mocked timings or acceptance alone.

The local tests cover plumbing; steps 3–6 need a compatible real model runtime and
accelerator validation. An automatic background training loop is deliberately not
part of this initial scope. Periodic, manually reviewed candidates are sufficient
to test whether personalization actually pays off.

## Product message

**Train on your work. Measure on your work. Apply only what improves your work.**

Position the platform as a path from reviewed personal conversations to measurable
draft customization—not as guaranteed speedup, universal architecture support or
an already-running self-learning service. The useful launch story is a reproducible
before/after workload experiment, including negative results and total cost.
