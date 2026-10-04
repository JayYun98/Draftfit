"""Taichu text training wrapper for mixed full and sliding draft layers."""

from draftfit.algorithms.common.dflash_family_model import OnlineDFlash2Model


class OnlineTaichuDFlash2Model(OnlineDFlash2Model):
    supports_mixed_attention = True

    def _block_mask_options(self):
        if len(set(self.draft_model.layer_types)) > 1:
            return {"is_causal": None}
        return super()._block_mask_options()
