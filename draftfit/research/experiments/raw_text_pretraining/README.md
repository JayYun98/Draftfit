# Raw-text DFlash-family initialization

Use `draftfit pretrain` with local JSONL text and a compatible draft config, then
`draftfit expand-aux-layers` to widen the projection for verifier-feature distillation.
The [pretraining implementation](../../../../draftfit/pretraining.py) reads the frozen target embedding and head;
it does not run the verifier decoder. The resulting checkpoint starts a new optimizer in the shared trainer.

[Tiny-model tests](../../../../tests/test_pretraining.py) cover all three methods and projection widening.
The [specification](../../../../docs/specs/dflash_raw_pretrain.md) distinguishes exact weight placement from
full-forward outputs that match only within tolerance. The pretraining loop saves weights without optimizer state.
The GPU comparison of acceptance and serving speed against scratch initialization has not run.
