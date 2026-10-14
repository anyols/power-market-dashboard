"""Render every page headlessly, offline, and fail on any exception."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["board", "ratios", "exposure", "seasonality", "power", "oil", "notes", "backtests", "about"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("MARKETS_SOURCE", "synthetic")
    monkeypatch.setenv("CRUDE_SOURCE", "sample")
    monkeypatch.setenv("DATA_SOURCE", "sample")


def _page_script(module_name: str) -> None:
    import importlib

    from src.views.common import persist_crude_signal_state, persist_signal_state

    persist_signal_state()
    persist_crude_signal_state()
    importlib.import_module(f"src.views.{module_name}").render(None)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_function(_page_script, args=(page,), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_power_page_other_zones():
    for zone in ["FR", "ES", "IT_NORD"]:
        at = AppTest.from_function(_page_script, args=("power",), default_timeout=120)
        at.session_state["zone"] = zone
        at.run()
        assert not at.exception, (zone, [e.message for e in at.exception])


def test_full_app_boots():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
