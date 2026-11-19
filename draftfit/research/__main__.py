"""Bounded native research entry point, reusable by the main CLI."""

import argparse
import json
import sys

from .catalog import source_catalog
from .config import load_recipe
from .runner import plan, run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["intake"]:
        from .intake import main as intake_main

        return intake_main(argv[1:])
    parser = argparse.ArgumentParser(
        description="Run an experimental research recipe",
        epilog="Track literature with: draftfit research intake --help",
    )
    parser.add_argument("-c", "--config")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--sources", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    if args.sources:
        result = source_catalog()
    else:
        if not args.config:
            parser.error("--config is required unless --sources is selected")
        try:
            config = load_recipe(args.config, args.overrides)
            result = (
                plan(config, args.config) if args.plan else run(config, args.config)
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
