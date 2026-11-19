# ZDTaichu5.0-9B text drafts

For the new DFlash2 draft use the [three-layer recipe](../../../../configs/zdtaichu5-9b-dflash2.json).
For official DSpark weights pass the pinned draft directory to `draftfit target prepare --strategy dspark`
via `--draft-checkpoint` and omit `--draft-config`; the [DSpark metadata recipe](../../../../configs/zdtaichu5-9b-dspark.json)
describes the expected geometry. The [preparation guide](../../../../docs/RECENT_MODEL_NEXT_TARGETS.md#official-warm-start-contract)
explains the text-only Transformers teacher and separate five-layer DFlash2 retarget path.

[Warm-start tests](../../../../tests/test_runtime/test_speculators_warmstart.py) check copied embedding/head tensors;
[text capture tests](../../../../tests/test_offline_capture/test_transformers.py) exercise decoder selection.
The [Taichu DFlash2 model](../../../../draftfit/modeling/draft/taichu_dflash2.py) handles its restricted text MRoPE and mixed attention.
Tiny checks do not establish original-weight feature parity, multimodal training or exact serving.
