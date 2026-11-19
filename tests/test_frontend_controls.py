from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_control_workflow_in_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed; JavaScript workflow harness was not run.")
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, str(project_root / "tests" / "dashboard_controls.test.js")],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Frontend workflow passed" in result.stdout
