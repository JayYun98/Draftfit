# AutoDraftModel uses AutoConfig when reopening an HF export. DFlash2's own
# model_type is not built into Transformers, so register its local config.
from transformers import AutoConfig

from .base import Eagle3DraftModel
from .dflash import (
    DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
    sample,
)
from .dflash2 import (
    CandidateSelector,
    DFlash2Config,
    DFlash2DecoderLayer,
    DFlash2DraftModel,
    DFlashGroupedConv,
    dflash2_config_for_serving,
    dflash2_state_dict_for_serving,
)
from .domino import DominoDraftModel
from .dspark import DSparkDraftModel
from .eagle4 import LlamaForCausalLMEagle4
from .llama3_eagle import LlamaForCausalLMEagle3
from .osprey_llama import OspreyLlamaDraftModel
from .osprey_qwen3 import OspreyQwen3DraftModel
from .peagle import PEagleDraftModel
from .registry import DRAFT_REGISTRY, available_drafts, register_draft, resolve_draft
from .taichu_dflash2 import TaichuDFlash2Config, TaichuDFlash2DraftModel

AutoConfig.register(DFlash2Config.model_type, DFlash2Config)

__all__ = [
    "Eagle3DraftModel",
    "DFlashDraftModel",
    "DFlash2Config",
    "DFlash2DecoderLayer",
    "DFlash2DraftModel",
    "TaichuDFlash2Config",
    "TaichuDFlash2DraftModel",
    "DFlashGroupedConv",
    "CandidateSelector",
    "dflash2_config_for_serving",
    "dflash2_state_dict_for_serving",
    "DominoDraftModel",
    "DSparkDraftModel",
    "LlamaForCausalLMEagle3",
    "LlamaForCausalLMEagle4",
    "PEagleDraftModel",
    "OspreyQwen3DraftModel",
    "OspreyLlamaDraftModel",
    "build_target_layer_ids",
    "extract_context_feature",
    "sample",
    "DRAFT_REGISTRY",
    "register_draft",
    "resolve_draft",
    "available_drafts",
]
