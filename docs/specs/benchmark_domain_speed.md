# Held-out benchmark domain speed

## Problem / scope

- The SGLang benchmark currently writes only run-wide throughput. A mixed-source held-out JSONL cannot be audited by data domain after the server is gone.
- Keep the existing aggregate metric unchanged. For local messages JSONL, derive a domain from an explicit `domain` string or the prefix of `id` before `:`; use `unlabeled` when neither exists. Do not write prompt or response text to the result.
- This changes benchmark reporting only. It does not change training, draft weights, export, or serving behavior.

## Contract / implementation

- Each measured request records its input index, domain, completion token count, wall latency, request token rate, and optional acceptance length. Warmup requests are excluded.
- The output JSON reports request count, median request token rate, median latency, output token total, and mean available acceptance length for each domain. Preserve run-wide output throughput as total generated tokens divided by the whole measured wall interval.
- The domain summary is a per-request speed distribution. With concurrency above one, it is not an isolated per-domain server throughput. Repeated prompts count as repeated measured requests.
- Existing built-in datasets use their preset name as the domain. Existing messages JSONL without domain metadata remains valid.

## Acceptance and evidence

| Requirement | Pass criterion | Check | Status |
| --- | --- | --- | --- |
| Correct domain attribution | Local JSONL with two source-prefixed IDs yields two named groups and no warmup row | Focused benchmark integration test | pass |
| Existing aggregate | Output token total, overall throughput calculation, and old result fields remain valid | Benchmark and compare suites, 19 tests | pass |
| Local CLI output | New JSON fields and domain console lines appear through actual CLI dispatch | Local mock-server command, real cached MiniCPM5 tokenizer and 53-row JSONL | pass |
| Real SGLang GPU | Matched target/official/draft runs on a balanced disjoint domain holdout | New GPU required; not authorized in this change | not-run |

## Usage

Use `draftfit benchmark --model TARGET --data-path heldout.jsonl --num-prompts N --concurrency 1 --output-json result.json` against the already running target/draft server. Keep the same ordered JSONL, target revision, sampling, output limit, concurrency, and server settings for every arm. Use at least three warmed repetitions and report the median of per-run domain medians. The existing 53-row MiniCPM5 set has 35/6/6/6 rows; those last three domains are exploratory until a larger disjoint holdout is measured.

## Results / limitations

The historical A100 runs produced aggregate JSON only; their domain rates cannot be reconstructed. Existing 53-row MLX per-request records were grouped into [the domain report](../reports/minicpm5-domain-speed-2026-10-04.json), with source hashes and no prompt text. The model sessions were not interleaved, and the minority domains contain six examples each. The local mock-server CLI run confirms the reporting path, not real GPU speed.
