# Performance evidence gate

The repository does not currently certify production speedup. A benchmark report
measures aggregate output throughput against a running SGLang server. It does not
save generated answers or prove their correctness, server identity, hardware
equivalence, independent repetitions, or immutable model revisions.

## Compare recorded measurements

Use benchmark JSON from `draftfit benchmark --data-path` and `--output-json`
options. Run the target-only baseline and the candidate separately with identical
held-out input, target/tokenizer, generation settings, concurrency and warmup.
Keep each real measurement in a separate file.

```sh
python -m draftfit.benchmarks.compare \
  --baseline baseline-1.json baseline-2.json baseline-3.json \
  --candidate candidate-1.json candidate-2.json candidate-3.json
```

The command checks required evidence, finite positive measurements, internal
throughput arithmetic, completed request count, and matching recorded workload
and settings. Source file paths and server URLs can differ. It requires the JSONL
SHA256; built-in dataset reports lack immutable workload identity and are rejected.
Missing legacy metadata is rejected too. Exit code 2 means comparison rejected;
exit code 0 means a valid comparison, **not** a release gate pass. A slower
candidate is still a valid comparison. Output includes median throughput and
candidate/baseline ratio, with `production_gate: not_evaluated` even for many runs.
Duplicate file paths are rejected, but copied reports cannot be identified as
independent experiments. The CLI does not authenticate report provenance.

## Inspect speed by held-out data domain

For a local messages JSONL, add a short `domain` string to each row or use an
`id` such as `opencodeinstruct:sample-id`. The benchmark groups by `domain`
first, otherwise by the ID prefix before `:`, and uses `unlabeled` when neither
is present. It does not save prompt or generated response text.

```sh
python -m draftfit.cli benchmark \
  --model /path/to/target --data-path heldout.jsonl \
  --num-prompts 53 --concurrency 1 --max-new-tokens 128 \
  --output-json official-1.json
```

Use a fresh output path for every target-only, official-draft and candidate
repetition. The JSON keeps the original run-wide `throughput_tokens_per_second`
and adds `request_metrics` and `domain_metrics`. Each domain reports its number
of measured requests, output token count, median request tok/s, median request
latency and mean available acceptance length. Warmups are excluded. Repeated
prompts count as repeated requests. With concurrency above one, a domain median
includes shared-server contention; it is **not** isolated domain throughput.
Compare each domain across models only with the same ordered held-out file,
model target, output limit and server settings. Use balanced disjoint domains
and at least three independent, order-alternated runs for a firm conclusion.

## GPU evidence still required before deployment

1. Record real target-only and speculative server commands, software versions,
   immutable target/tokenizer/draft revisions, GPU type/count, precision,
   parallelism, cache settings and other traffic. Verify the intended draft loads.
2. Collect repeated independent measurements on that same hardware and held-out
   workload, alternating run order. Record failures, output lengths, run variation,
   latency percentiles, memory use and agreed workload-specific thresholds.
3. Save and evaluate generated answers against held-out references or task tests.
   Check speculative decoding correctness against the target under the actual
   sampling settings; throughput and acceptance length cannot establish quality.
4. Review correctness and repeatability together before any deployment decision.

No GPU experiment is run by this comparison command. Unit-test fixtures are
synthetic schema checks and must never be reported as measured acceleration.

```sh
python -m unittest tests.test_benchmarks.test_compare
```
