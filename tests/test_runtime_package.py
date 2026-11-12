"""Exercise console import resolution with separate source and installed trees."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("stale", [False, True])
def test_console_runtime_checks_installed_code_not_cwd(tmp_path, stale):
    source = tmp_path / "source" / "plex_generate_previews"
    installed = tmp_path / "installed" / "plex_generate_previews"
    for package in (source, installed):
        (package / "web").mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "web" / "scheduler.py").write_text("CURRENT = True\n")
    if stale:
        (installed / "web" / "scheduler.py").write_text("CURRENT = False\n")
    console = tmp_path / "bin" / "plex-previews-web"
    console.parent.mkdir()
    console.write_text("# installed console entry point\n")
    env = dict(os.environ, PYTHONPATH=str(installed.parent))
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_runtime_package.py"
    result = subprocess.run(
        [sys.executable, str(script), "--source-tree", str(source), "--console-script", str(console)],
        cwd=source.parent, env=env, text=True, capture_output=True,
    )
    if stale:
        assert result.returncode != 0
        assert "Installed runtime source differs: web/scheduler.py" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert str(installed) in result.stdout
