---
title: "Draftfit: Your workload. Your draft."
description: "Train, fine-tune, and evaluate speculative drafts on your own workload, with bounded GPU evidence and explicit limits."
date: 2026-09-18
slug: "draftfit-workload-specific-drafts"
categories: ["speculative-decoding", "Draftfit"]
tags: ["Draftfit", "speculative decoding", "draft training"]
draft: true
images: []
---

Draftfit — Your workload. Your draft.

I'm building Draftfit to train, fine-tune, and evaluate speculative drafts on the prompts and responses that reflect your own workload.

The starting point isn't “which draft is fastest for everyone?” It's “does this draft fit the traffic I actually care about?”

That means keeping the target revision, chat template, assistant masks, captured features, and held-out evaluation connected. A lower training loss is useful evidence of an update—not proof of better acceptance or lower serving latency.

There is real but bounded progress. With Qwen3-0.6B on two RTX 3090s, the native HF teacher path passed feature parity on four prompts, 20 updates each for DSpark and DFlash2, exact next-step resume, and HF export/reload. vLLM passed the FP32 parity and deterministic-training gate; its BF16 feature-parity gate failed and stays failed. These are short functional checks, not workload-quality or speedup benchmarks.

Draftfit includes an owned training runtime derived from SpecForge, with design references to TorchSpec and AngelSpec. DSpark is a draft method within the platform—not the product name.

Automatic production-traffic ingestion through OpenTelemetry, continual retraining, and automatic deployment are future directions, not implemented product capabilities. Today, you provide the dataset and choose the runs and evaluation gates.

I'm documenting what it takes to make a draft fit a specific workload without turning a small passing test into a universal claim. If you're exploring workload-specific speculative decoding, I'd like to compare notes.

#Draftfit #SpeculativeDecoding #MachineLearning

## Editorial references — do not include when posting

[Evidence and limits](../docs/MODEL_VALIDATION.md), [series index](README.md), [target and mask preparation](draftfit-target-onboarding-chat-masks.md), [live versus saved features](gpu0-teacher-gpu1-draft-online-offline.md).

Not posted. No public URL or release availability is promised.

## FAQ

**Does this promise a speedup?** No; held-out acceptance and target-only versus draft-assisted serving need separate measurements.

**Can any architecture be used?** No; each target, backend, method, and hardware combination needs its own validation.
