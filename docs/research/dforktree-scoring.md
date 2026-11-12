# Run the DForkTree scoring reference

Use this recipe to inspect fork-aware candidate probability correction on CPU.
The included synthetic fixture distinguishes spine and post-fork weighting.
It also checks that the probability outside the candidate list remains in the
normalization. See the [mathematical contract](../specs/dforktree-scoring.md).

## Run the included fixture

Run these commands from the repository root:

```sh
python -m draftfit.research -c examples/configs/research/dforktree-scoring.yaml --plan
python -m draftfit.research -c examples/configs/research/dforktree-scoring.yaml
```

Planning reads the recipe and validates its configuration.
It does not read the fixture or create the output directory.
Execution writes `recipe.json` and `result.json` to
`outputs/research-dforktree-scoring`.
Use a new output directory for another run:

```sh
python -m draftfit.research -c examples/configs/research/dforktree-scoring.yaml \
  output_dir=outputs/dforktree-second-run
```

For candidates `[101, 102]` with original probabilities `[0.5, 0.3]`, the
synthetic evidence gives these analytic expectations:

| Context | Corrected candidate probabilities | Corrected tail | Candidate order |
| --- | --- | --- | --- |
| First position | `[0.5, 0.3]` | `0.2` | `101, 102` |
| Parent on the original spine | `[0.4163391976, 0.3786313315]` | `0.2050294709` | `101, 102` |
| Parent has left the spine | `[0.3350392805, 0.4618308844]` | `0.2031298351` | `102, 101` |

These values test the numerical mechanism.
They do not measure acceptance or throughput on a model.
The tail's unnormalized mass remains `0.2`; its normalized mass can change.

## Run from an installed package

Export the packaged recipes into a new directory:

```sh
python -m draftfit.assets export ./recipes
cd recipes
python -m draftfit.research -c examples/configs/research/dforktree-scoring.yaml --plan
python -m draftfit.research -c examples/configs/research/dforktree-scoring.yaml
```

Paths in the recipe resolve from the working directory.
The exported fixture contains synthetic probabilities and no model weights.

## Prepare probability evidence

Use one JSON object per line. The bundled fixture uses the `.json` extension
so the existing asset exporter includes it. A later-position row has this form:

```json
{"id":"fork","candidate_ids":[101,102],"marginals":[0.5,0.3],"conditional":[0.2,0.4],"unigram":[0.4,0.2],"position":2,"parent_path":"forked"}
```

`position` starts at one. `parent_path` is `spine` or `forked`.
Candidate identities must be distinct, nonnegative integers.
The four vectors must be nonempty and have equal lengths.
All probability vectors describe subsets of distributions and must have mass
at most one in CPU floating-point arithmetic.
Conditional and unigram values must be positive, including neutral cases.
Marginal values may be zero.
The public function accepts Python integer and floating-point scalars.
Convert array or tensor scalars explicitly before calling it.

Set `data.input_path` to your file and `data.max_samples` to a positive integer.
Set `input_kind` to `recorded` for recorded probability evidence.
That label does not authenticate a measurement or its model identity.
Execution hashes the complete input file, including the unprocessed suffix.
Only the selected nonblank rows are parsed and scored.

## Use the scoring function

```python
from draftfit.research.dforktree import score_candidates

result = score_candidates(
    [101, 102],
    marginals=[0.5, 0.3],
    conditional=[0.2, 0.4],
    unigram=[0.4, 0.2],
    position=2,
    parent_path="forked",
)
print(result.candidate_probs)
print(result.tail_probability)
```

The result preserves the candidate order and includes the clipped PMI values
and selected weight. The synthetic example gives candidate probabilities
approximately `(0.3350392805, 0.4618308844)` and tail `0.2031298351`.

## Align branch evidence

Supply the original draft probability for each candidate.
Supply positive conditional and unigram probabilities for the same candidate
identities and vocabulary.
The conditional evidence must describe the preceding tokens on that branch.
This component does not estimate these probabilities from a corpus.

Classify the parent path against the drafter's original top-1 continuation.
An edge leaving that continuation uses the spine weight.
Its descendants use the post-fork weight once their parent path has left.
The first draft position receives no correction.
The reference consumes this metadata and does not reconstruct tree history.

PMI means pointwise mutual information.
It is the clipped log ratio of conditional probability to unigram probability.
The default weights are `0.3` on the spine and `0.6` after a fork.
The default clip is `10.0`.
Weights are research settings, not measured optimal values for your workload.

## Keep the runtime limits visible

The reference follows [DForkTree v1](https://arxiv.org/html/2611.12356v1),
equations 2 and 3, with the
[author implementation](https://github.com/Perseus037/DForkTree/tree/eeed8b698b8822aa2767ec5c191efc4e1b6cdbe3)
pinned under Apache-2.0.
The original license and notice ship with the package.

The local implementation uses CPU floating-point arithmetic and a true zero
tail. The author's builder uses float32 and a tail floor of `1e-6`.
The published `ngram_path_fn` also returns an undefined `fn` at the pinned
revision. The local reference does not import or execute that GPU path.

Raw count extraction, smoothing, GPU tree construction, GDN verification,
SGLang integration, and real-model evaluation are not implemented here.
Tokenizer and corpus alignment remain the caller's responsibility.
This recipe establishes no output-equivalence, quality, or speed improvement.
