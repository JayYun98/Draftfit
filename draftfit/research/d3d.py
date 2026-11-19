"""Compatibility exports for the D3D corrected KL component."""

from .papers.d3d.method import (
    corrected_kl_loss,
    corrected_teacher_probs,
    run_d3d_corrected_kl,
)

__all__ = ["corrected_kl_loss", "corrected_teacher_probs", "run_d3d_corrected_kl"]
