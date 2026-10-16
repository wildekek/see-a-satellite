"""The See A Satellite Tonight website's satellite catalogs.

The site's author collects the catalogs: orbital elements (mostly from
Space-Track, Starlink from CelesTrak) with intrinsic magnitudes and brightness
models curated from observations. This module only downloads and parses them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
from typing import Any

import aiohttp

from .predict import Satellite

SITE_CATALOG_BASE_URL = "https://pub-ac416a81df124ba7997caa2baca8bc2c.r2.dev/"
SITE_CATALOGS = {
    "brightest": "brightest-sats.json",
    "starlink": "starlink-sats.json",
}

_TIMEOUT = aiohttp.ClientTimeout(total=120)
DEFAULT_USER_AGENT = "see-a-satellite (+https://pypi.org/project/see-a-satellite/)"


class CatalogError(Exception):
    """A catalog could not be downloaded or parsed."""


def parse_site_catalog(entries: Iterable[Mapping[str, Any]]) -> list[Satellite]:
    """Parse the website's JSON catalog (OMM fields plus MAG and MODEL)."""
    satellites = []
    for entry in entries:
        fields = {
            k: v for k, v in entry.items() if k not in ("MAG", "MODEL", "TLE_LINE1", "TLE_LINE2")
        }
        mag = entry.get("MAG")
        satellites.append(
            Satellite.from_omm(fields, None if mag is None else float(mag), entry.get("MODEL"))
        )
    return satellites


async def fetch_site_catalog(
    session: aiohttp.ClientSession,
    name: str = "brightest",
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[Satellite]:
    """Download one of the website's catalogs ("brightest" or "starlink")."""
    url = SITE_CATALOG_BASE_URL + SITE_CATALOGS[name]
    try:
        async with session.get(url, timeout=_TIMEOUT, headers={"User-Agent": user_agent}) as resp:
            resp.raise_for_status()
            data = await resp.read()
    except (aiohttp.ClientError, TimeoutError) as err:
        raise CatalogError(f"Error fetching {url}: {err}") from err
    try:
        satellites = parse_site_catalog(json.loads(data))
    # AttributeError: entries that aren't objects, e.g. a JSON error body.
    except (ValueError, KeyError, TypeError, AttributeError) as err:
        raise CatalogError(f"Malformed site catalog: {err}") from err
    if not satellites:
        raise CatalogError("Site catalog is empty")
    return satellites


def dump_catalog(satellites: Iterable[Satellite]) -> list[dict[str, Any]]:
    """Serialize a catalog to JSON-compatible dicts in the site's format."""
    out = []
    for sat in satellites:
        entry: dict[str, Any] = dict(sat.omm)
        entry["OBJECT_NAME"] = sat.name
        if sat.magnitude is not None and math.isfinite(sat.magnitude):
            entry["MAG"] = sat.magnitude
        if sat.model:
            entry["MODEL"] = sat.model
        out.append(entry)
    return out
