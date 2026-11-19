"""Compatibility entry point for the MLX campaign evaluation adapter."""

import sys

from draftfit.research.experiments.mlx_evaluation import campaign_mlx

if __name__ == "__main__":
    campaign_mlx.main()
else:
    sys.modules[__name__] = campaign_mlx
