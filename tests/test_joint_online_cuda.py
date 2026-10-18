"""Opt-in actual CUDA coverage of the shared online family CLI."""

import unittest

import torch

from tests import test_joint_online_cpu as online_cpu


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class JointOnlineCUDA(online_cpu.JointOnlineCPU):
    execution_device = "cuda"
