# Spark sequential-prefix verification

With a compatible `spark2_5` Transformers text target, configure
`training.joint_verification` in the [joint family manifest workflow](../../../../docs/JOINT_ONLINE.md#compatibility-and-verification).
The [teacher](../../../../draftfit/offline_capture/transformers.py) captures block outputs before final normalization.
The [verifier](../../../../draftfit/training/joint_verification.py) checks sequential target prefixes
because low-precision block calls can select different tokens.

See the [Spark integration record](../../../../docs/specs/spark-generation-tuning.md),
[target routing test](../../../../tests/test_joint_verification.py) and [prefix stability cases](../../../../tests/test_joint_verification_prefix_stability.py).
The adapter assumes compatible mask and per-layer RoPE APIs. Fixture parity does not certify a native Spark runtime.
