"""ALR/IRA tensor step on the existing Trainer spine."""

from draftfit.training.strategies.base import DFlashTrainStrategy, StepOutput

KEYS = (
    "input_ids",
    "loss_mask",
    "hidden_states",
    "target_last_hidden_states",
    "alr_anchor",
)


class ALRTrainStrategy(DFlashTrainStrategy):
    name = "alr_ira"
    required_features = set(KEYS)

    def forward_loss(self, batch, ctx=None):
        self.validate_batch(batch)
        tensors = {
            key: batch.tensors[key].to(self._device(), non_blocking=True)
            for key in KEYS
        }
        loss, accuracy, metrics = self.dflash_model(**tensors)
        return StepOutput(
            loss=loss,
            metrics={"accuracy": accuracy.detach()},
            ratio_metrics=metrics["ratio_metrics"],
            loss_terms=metrics["loss_terms"],
        )
