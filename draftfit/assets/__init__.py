"""Packaged recipes; root configs/examples remain the source of truth."""

from importlib.resources import files
from pathlib import Path

DOCUMENTS = (
    "LICENSE",
    "THIRDPARTY_NOTICES.md",
    "licenses/torchspec-LICENSE",
    "licenses/angelspec-LICENSE",
    "licenses/z-lab-dflash-LICENSE",
    "docs/PUBLIC_WORKFLOW.md",
    "docs/PUBLIC_RELEASE.md",
    "docs/PUBLIC_SUPPORT.md",
    "docs/SOURCE_ATTRIBUTION.md",
    "docs/RUNTIME_PROFILES.md",
    "docs/NATIVE_LING_REPLAY.md",
    "docs/MODEL_VALIDATION.md",
    "docs/PERSONAL_INFERENCE.md",
    "docs/TARGET_EXTENSION.md",
    "docs/PERFORMANCE_GATE.md",
    "docs/RELEASE_GATES.md",
    "docs/CODE_OWNERSHIP.md",
    "docs/ENVIRONMENT.md",
    "docs/OWNED_GPU_ENVIRONMENT.md",
)


def _root():
    bundled = files(__package__).joinpath("_data")
    if bundled.is_dir():
        return bundled
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "configs").is_dir():
        return checkout
    raise FileNotFoundError(
        "Recipe assets are missing; reinstall a complete draftfit wheel."
    )


def _walk(node, prefix):
    for child in sorted(node.iterdir(), key=lambda item: item.name):
        name = f"{prefix}/{child.name}"
        if child.is_dir():
            yield from _walk(child, name)
        elif child.name.endswith((".json", ".yaml", ".yml")):
            yield name, child


def list_assets():
    """Return bundled relative names, preserving paths referenced by recipes."""
    root = _root()
    return [
        name
        for directory in ("configs", "examples/configs")
        for name, _ in _walk(root.joinpath(directory), directory)
    ] + list(DOCUMENTS)


def export_assets(destination):
    """Export into a new directory; never overwrite a user's customized recipe."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    root = _root()
    for name in list_assets():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(root.joinpath(name).read_bytes())
    return destination
