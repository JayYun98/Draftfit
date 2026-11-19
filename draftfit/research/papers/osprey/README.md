# Osprey: convert a pretrained backbone into a draft

[Conversion](method.py) retains a prefix of a local, unquantized Qwen3 or Llama
checkpoint. The new draft initially reproduces that prefix's logits: the
converted attention projections ignore the added target features until training.
The source must use bias-free full attention and supported RoPE (default, or
Llama3 scaling for Llama).

Use the [Osprey recipe example](../../../../docs/specs/research-methods.md#osprey-conversion-and-training)
with `python -m draftfit.research -c recipe.json`. Set the number of retained
layers, target hidden size, three ascending target layer IDs, and a fresh output
directory. If the target uses another vocabulary, set `model.target_tokenizer`.
[Alignment](tokenizer_alignment.py) copies matching embedding and head rows,
averages source rows for tokens that decompose, and leaves unmapped rows zero.
It also updates the vocabulary buffers and special token IDs. Source-logit
preservation applies before this optional alignment.

The recipe writes a loadable draft and `adaptation.json`. For training, point
`model.draft_model_config` and `model.draft_checkpoint_path` at the converted
directory and set `model.load_target_embedding=false`. The registered
[`osprey` strategy](../../../algorithms/osprey/providers.py) owns training and
checkpoint behavior. The [Qwen3](../../../modeling/draft/osprey_qwen3.py) and
[Llama](../../../modeling/draft/osprey_llama.py) draft classes remain registered
under their original architecture names.

The [source catalog](../../catalog.py) pins LeanModels/Osprey. See the
[source notice](../../NOTICE.md#osprey) for licensing.

The [regression tests](../../tests/test_research_methods.py) cover conversion, tokenizer alignment, and training. The [MIT license](LICENSE) is included with this package.
