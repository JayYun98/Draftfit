# Asynchronous joint capture

Set `training.joint_async_capture: true` in the base config for
`draftfit family --manifest family.json --joint`. Use a built-in Transformers
teacher with `joint_online_capture: true`. Set the queue byte and batch
limits large enough to hold a full accumulation window.

The [producer](queue.py) captures the next batch while the trainer consumes
the previous batch. The trainer commits the cursor only after the optimizer
step. The default is synchronous. See the [configuration and CPU/CUDA checks](../../../../docs/specs/async-joint-capture.md)
and the [queue and recovery contract](../../../../docs/SHARED_TARGET_FEATURE_QUEUE.md).
GPU overlap, memory peaks and speed have not been verified.
