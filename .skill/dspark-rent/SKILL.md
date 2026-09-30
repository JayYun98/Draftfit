---
name: dspark-rent
description: Search Vast offers, rent within authorized budgets, back up results and clean up exact GPU resources for Draftfit.
---

# GPU rental and cleanup

Locate the repository; read [environments](../../docs/MODEL_ENVIRONMENTS.md)
and [runtime profiles](../../docs/RUNTIME_PROFILES.md). Historical prices,
balances and IDs are neither current authorization nor active resources.

Search-only requests permit read-only comparisons. Confirm total budget/time
before rental, without blocking searches.

1. Finish local checks and select GPU gates. Establish disk-inclusive hourly
   and total cost/time limits, GPU count/VRAM/generation and spot permission.
   Search is not rental authorization; ask for missing cost limits before renting.
2. Inventory existing instances/volumes read-only and record them separately.
   Keep keys/tokens secret; preserve other experiments' resources.
3. Check installed CLI help or official API syntax. Spot search uses `-i`.
   Verify units: `gpu_ram` filters use GB; results may use MiB. Compare total
   disk-inclusive cost, availability, reliability and topology. Offers are not
   reservations and can be stale; bound retries of failed offers.
4. Rent only within authorization. Immediately record returned instance IDs,
   distinct from offer IDs, and separate volume IDs/ownership in the manifest.
   Re-query after ambiguous creation before risking duplicates.
5. Check SSH/CUDA/VRAM/disk/backend source; setup/download time counts toward
   budget. Back up small logs/configs/JSON/checkpoint metadata each stage.
   Transfer large weights/features only when justified within user limits.
6. Check balance/cost each stage against the user's warning threshold. Reserve
   time for warning, stopping, backup and deletion; do not keep billing without
   a top-up. Claim continuous monitoring only when actually implemented.
7. At the stop condition, verify local backup sizes/hashes and delete only exact
   IDs owned by this run. Re-query to confirm those IDs are absent; other
   experiments may remain. Stopping alone may still incur disk charges.
   Report remaining resources or backup failures.

Report rental IDs, total hourly price, gates, backup paths and cleanup evidence.
If API/CLI access is unavailable, provide preparation steps and stop. Never
expose credentials from local files.
