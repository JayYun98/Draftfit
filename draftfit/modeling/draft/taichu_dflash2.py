"""Text-only Taichu DFlash2 draft with its published Qwen3.5 RoPE layout."""

from .dflash2 import DFlash2Config, DFlash2DraftModel
from .registry import register_draft


class TaichuDFlash2Config(DFlash2Config):
    supports_text_mrope = True
    supports_mixed_attention = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@register_draft
class TaichuDFlash2DraftModel(DFlash2DraftModel):
    config_class = TaichuDFlash2Config

    def config_for_serving(self):
        config = super().config_for_serving()
        config["architectures"] = ["DFlash2DraftModel"]
        return config
