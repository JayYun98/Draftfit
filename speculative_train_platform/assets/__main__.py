"""Dependency-free recipe access: python -m speculative_train_platform.assets list|export DIR."""

import argparse

from . import export_assets, list_assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list bundled recipe/config paths")
    export = commands.add_parser("export", help="copy assets into a NEW directory")
    export.add_argument("destination")
    args = parser.parse_args(argv)
    if args.command == "list":
        print("\n".join(list_assets()))
    else:
        try:
            print(export_assets(args.destination))
        except OSError as exc:
            parser.exit(1, f"Could not export recipes: {exc}\n")


if __name__ == "__main__":
    main()
