# NgramRL

[NgramRL §3.2](https://arxiv.org/abs/2610.32881v1) describes drafting from verified
rollout history. This [independent
implementation](../../NOTICE.md#independently-implemented-published-algorithms) keeps
per-prompt counts in a bounded window. A snapshot proposes from the longest matching
context; the shared verifier checks the tokens against the target.

**Run.** Set a local dense target and JSONL input in the
[recipe](../../../../examples/configs/research/ngramrl.yaml). Rows need `input_ids` and
a nondecreasing integer `step`; `prompt_id` can further scope reuse. From the repository
root:

```sh
python -m draftfit.research -c examples/configs/research/ngramrl.yaml --plan
```

Remove `--plan` to execute. The [memory runner](../../memory_runner.py) writes verified
history for continuation. This synchronous reference does not implement concurrent
rollout workers or CPU/GPU overlap.

**Tests.** [Memory regressions](../../../../tests/research/test_latest_research.py)
cover snapshot aging, sampling, target parity, and resuming from a state written by the
CLI.
