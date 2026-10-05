"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


def _fixture_pairs() -> list[Path]:
    """Discover every (foo.md, foo.html) pair under tests/fixtures/."""
    return sorted(p for p in FIXTURES.glob("*.md") if p.with_suffix(".html").exists())


@pytest.fixture(params=_fixture_pairs(), ids=lambda p: p.stem)
def fixture_pair(request) -> tuple[Path, Path]:
    md = request.param
    return md, md.with_suffix(".html")
