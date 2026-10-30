"""Two-stage EAGLE-4 training using the shared EAGLE recurrent unroll."""

from draftfit.algorithms.eagle3.model import OnlineEagle3Model


class Eagle4TrainingModel(OnlineEagle3Model):
    def auxiliary_loss(self, hidden, logits, position_mask):
        if self.draft_model.training_stage == "candidates":
            return self.draft_model.candidate_loss(hidden, logits, position_mask)
        return super().auxiliary_loss(hidden, logits, position_mask)
