"""Tests for downloading and parsing the website's catalogs."""

import json
from pathlib import Path

import pytest

from see_a_satellite import CatalogError, dump_catalog, fetch_site_catalog, parse_site_catalog
from see_a_satellite.catalog import SITE_CATALOG_BASE_URL

from .conftest import FakeSession

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def site_entries() -> list[dict]:
    return json.loads((FIXTURES / "brightest-sats.json").read_text())


def test_dump_catalog_round_trip(site_entries) -> None:
    sats = parse_site_catalog(site_entries[:20])
    again = parse_site_catalog(json.loads(json.dumps(dump_catalog(sats))))
    assert [(s.norad_id, s.name, s.magnitude, s.model, s.epoch) for s in again] == [
        (s.norad_id, s.name, s.magnitude, s.model, s.epoch) for s in sats
    ]


async def test_fetch_site_catalog(site_entries) -> None:
    session = FakeSession()
    session.add(SITE_CATALOG_BASE_URL + "brightest-sats.json", json.dumps(site_entries))
    session.add(SITE_CATALOG_BASE_URL + "starlink-sats.json", 503)
    assert len(await fetch_site_catalog(session)) == 864  # type: ignore[arg-type]
    with pytest.raises(CatalogError):
        await fetch_site_catalog(session, "starlink")  # type: ignore[arg-type]


def test_string_magnitude_is_a_number(site_entries) -> None:
    entry = {**site_entries[0], "MAG": "-1.75"}
    assert parse_site_catalog([entry])[0].magnitude == -1.75


@pytest.mark.parametrize(
    "body", ["not json", "[]", '[{"MAG": 1}]', '{"error": "rate limited"}', "[1, 2]"]
)
async def test_fetch_malformed_site_catalog(body: str) -> None:
    session = FakeSession()
    session.add(SITE_CATALOG_BASE_URL + "brightest-sats.json", body)
    with pytest.raises(CatalogError):
        await fetch_site_catalog(session)  # type: ignore[arg-type]
