"""Render every dashboard page headlessly on the sample data and fail on any exception."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["overview", "drivers", "forecast_errors", "spikes", "flows", "signal_lab", "daily_brief", "risk", "methodology"]


def _page_script(module_name: str, zone: str) -> None:
    import importlib
    from datetime import date

    from src.views.common import build_context, persist_signal_state

    persist_signal_state()
    ctx = build_context(zone, date(2024, 3, 1), date(2024, 4, 30), "sample", "Percentile", 90.0, 150.0)
    importlib.import_module(f"src.views.{module_name}").render(ctx)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page, monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "sample")
    at = AppTest.from_function(_page_script, args=(page, "DE_LU"), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def _crude_page_script(module_name: str) -> None:
    import importlib

    from src.views.common import build_crude_context, persist_crude_signal_state

    persist_crude_signal_state()
    ctx = build_crude_context("5 years")
    importlib.import_module(f"src.views.{module_name}").render(ctx)


CRUDE_PAGES = ["crude_overview", "crude_inventories", "crude_curve", "crude_signal_lab", "crude_brief"]


@pytest.mark.parametrize("page", CRUDE_PAGES)
def test_crude_page_renders_without_exceptions(page, monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "sample")
    at = AppTest.from_function(_crude_page_script, args=(page,), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_data_status_page_renders():
    at = AppTest.from_function(lambda: __import__("src.views.data_status", fromlist=["render"]).render(None),
                               default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("zone", ["FR", "ES", "IT_NORD"])
def test_other_zones_render(zone, monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "sample")
    for page in ["overview", "flows", "daily_brief"]:
        at = AppTest.from_function(_page_script, args=(page, zone), default_timeout=120)
        at.run()
        assert not at.exception, (page, [e.message for e in at.exception])


def test_full_app_boots(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "sample")
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
