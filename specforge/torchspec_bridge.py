"""Minimal TorchSpec-style launch/checkpoint bridge.

This is intentionally a planner, not a second distributed runtime.  The
existing SpecForge torchrun/FSDP2 path remains the executor; this module gives
it the same explicit resource and checkpoint contract used by TorchSpec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TorchSpecLaunch:
    nodes: int = 1
    gpus_per_node: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    fsdp_sharding: str = "full_shard"
    fsdp_version: str = "v2"
    checkpoint_format: str = "distributed"
    checkpoint_root: Optional[str] = None
    max_checkpoints: int = 0

    def __post_init__(self) -> None:
        if self.nodes < 1 or self.gpus_per_node < 1:
            raise ValueError("nodes and gpus_per_node must be positive")
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master_port must be between 1 and 65535")
        if self.fsdp_sharding not in {"full_shard", "shard_grad_op", "no_shard"}:
            raise ValueError(f"unsupported FSDP sharding: {self.fsdp_sharding}")
        if self.fsdp_version not in {"v1", "v2"}:
            raise ValueError(f"unsupported fsdp version: {self.fsdp_version}")
        if self.checkpoint_format not in {"distributed", "single_file"}:
            raise ValueError(f"unsupported checkpoint format: {self.checkpoint_format}")
        if self.max_checkpoints < 0:
            raise ValueError("max_checkpoints must be >= 0")

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    def torchrun_argv(self, module: str = "specforge.cli") -> tuple[str, ...]:
        return (
            "torchrun",
            f"--nnodes={self.nodes}",
            f"--nproc-per-node={self.gpus_per_node}",
            f"--master-addr={self.master_addr}",
            f"--master-port={self.master_port}",
            "-m",
            module,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "executor": "specforge_torchrun",
            "world_size": self.world_size,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "fsdp_sharding": self.fsdp_sharding,
            "fsdp_version": self.fsdp_version,
            "checkpoint_format": self.checkpoint_format,
            "checkpoint_contract": {
                "root": self.checkpoint_root,
                "rotation": self.max_checkpoints,
                "shared_filesystem_required": self.nodes > 1,
                "resume_supported": True,
            },
        }


__all__ = ["TorchSpecLaunch"]
