"""App smoke test: every tab loads without a Streamlit runtime exception.

    python -m pytest tests/test_app_tabs.py -q
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_every_tab_loads_without_error() -> None:
    at = AppTest.from_file("streamlit_app.py", default_timeout=60)
    at.run()
    assert not at.exception
    for tab in at.tabs:
        tab.run()
        assert not at.exception
