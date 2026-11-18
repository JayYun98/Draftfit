# TableTune

[Run the example](../../../../docs/research/paper-intake.md#run-the-first-experiment)
with the [synthetic recipe](../../../../examples/configs/research/tabletune.yaml).

[method.py](method.py) validates the profiles and constructs the offline table.
[upstream.py](upstream.py) is the pinned author kernel. See the
[source revision](../../NOTICE.md) and [upstream license](TABLETUNE_LICENSE).

The [specification](../../../../docs/specs/paper-intake.md) defines the supported scope.
The [CLI tests](../../tests/test_tabletune.py) check schedules and source hashes.
