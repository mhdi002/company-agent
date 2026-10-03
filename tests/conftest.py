import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PA_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network tests disabled (set PA_NETWORK_TESTS=1)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tmp_logs(tmp_path):
    from core import logging as plog
    plog.configure(tmp_path / "logs", run_id="test")
    return tmp_path / "logs"
