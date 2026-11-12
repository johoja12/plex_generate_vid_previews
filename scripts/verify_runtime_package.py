"""Fail an image build when its console entry point would load stale source."""

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys


def verify(source_tree: Path, console_script: Path) -> Path:
    source_tree = source_tree.resolve()
    # Match execution of the installed console script, not `python -` from /app.
    sys.path[:] = [str(console_script.resolve().parent)] + [
        path for path in sys.path[1:]
        if Path(path).resolve() != source_tree.parent
    ]
    spec = importlib.util.find_spec("plex_generate_previews")
    if spec is None or spec.origin is None:
        raise RuntimeError("Console entry point cannot find plex_generate_previews")
    installed = Path(spec.origin).resolve().parent
    if installed == source_tree:
        raise RuntimeError("Console entry point resolves to the source tree, not the installed package")
    for source in sorted(source_tree.rglob("*.py")):
        relative = source.relative_to(source_tree)
        # setuptools-scm generates this module while building the wheel.
        if relative == Path("_version.py"):
            continue
        target = installed / relative
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Installed runtime source differs: {relative}")
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, default=Path("/app/plex_generate_previews"))
    parser.add_argument("--console-script", type=Path)
    args = parser.parse_args()
    console = args.console_script or shutil.which("plex-previews-web")
    if not console or not Path(console).is_file():
        parser.error("Installed plex-previews-web console script was not found")
    installed = verify(args.source_tree, Path(console))
    print(f"Console runtime source verified: {installed}")


if __name__ == "__main__":
    main()
