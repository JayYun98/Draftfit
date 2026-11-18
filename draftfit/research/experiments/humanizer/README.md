# Humanizer joint-family preparation

Use `python -m draftfit.humanizer_campaign --help` for the [manifest preparation module](../../../../draftfit/humanizer_campaign.py).
Supply the pinned Gemma4 DFlash seed and either separate training/validation feature directories or token files.
The [preflight manifest](../../../../configs/experiments/humanizer-preflight.json) describes the prompt, split and blocking gates;
it is explicitly non-executable. The [campaign driver](../../../../scripts/run_humanizer_campaign.py) consumes prepared inputs.

[Input tests](../../../../tests/test_humanizer_campaign_inputs.py) check split and row preparation;
[tiny joint tests](../../../../tests/test_humanizer_joint_cpu.py) need a compatible Gemma4Unified text decoder.
See the [joint training guide](../../../../docs/JOINT_ONLINE.md) for the underlying engine.
Neither preparation nor a tiny joint update validates 12B capture, export parity or serving speed.
