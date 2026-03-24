"""Shared pytest configuration and fixtures."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--container",
        action="store_true",
        default=False,
        help="Run container smoke tests (requires docker compose up)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--container"):
        return
    skip = pytest.mark.skip(reason="need --container flag to run")
    for item in items:
        if "container" in item.keywords:
            item.add_marker(skip)
