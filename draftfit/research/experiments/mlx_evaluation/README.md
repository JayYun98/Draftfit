# MLX campaign evaluation

Select `python -m draftfit.campaign_mlx --profile PROFILE` as the evaluation command in a
[campaign manifest](../../../../configs/campaigns/joint-autopilot.example.json). Fill in the local benchmark,
runtime and workload paths in the [profile](../../../../configs/campaigns/mlx-profile.example.json).
The [adapter](campaign_mlx.py) verifies the trained export hashes, runs the external benchmark,
and aggregates matching target/draft requests by domain.

The [campaign guide](../../../../docs/FAMILY_CAMPAIGN.md#evaluation-and-policy) explains how the gate consumes the report.
[Aggregation tests](../../tests/test_campaign_mlx.py) reject missing trials and token mismatches.
They do not run an MLX engine; a report applies only to its workload and runtime.
