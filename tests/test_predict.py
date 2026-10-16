"""Check the predictions against passes computed by the website itself."""

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest

from see_a_satellite import (
    Satellite,
    brightest_possible,
    group_transits,
    parse_site_catalog,
    predict_transits,
)

FIXTURES = Path(__file__).parent / "fixtures"
START = datetime(2026, 10, 10, 19, tzinfo=UTC)


@pytest.fixture(scope="module")
def satellites() -> list[Satellite]:
    return parse_site_catalog(json.loads((FIXTURES / "brightest-sats.json").read_text()))


@pytest.fixture(scope="module")
def transits(satellites):
    return predict_transits(satellites, 52.0277, 5.5583, 194, start=START)


def test_matches_website(transits) -> None:
    """Every pass the site shows is predicted, with timing within seconds.

    The reference was captured from the site with
    ``#loc=52.0277,5.5583&epoch=2026-10-10T19:00:00Z`` and the catalog in
    fixtures/brightest-sats.json.
    """
    reference = json.loads((FIXTURES / "site_reference_veenendaal.json").read_text())
    ours = [t for g in group_transits(transits) for t in g.transits]
    expected = [s for g in reference for s in g["sats"]]
    assert len(expected) == 73
    for ref in expected:
        start = datetime.fromisoformat(ref["start"])
        end = datetime.fromisoformat(ref["end"])
        match = next(
            t
            for t in ours
            if t.satellite.norad_id == ref["id"] and abs((t.start - start).total_seconds()) <= 20
        )
        assert abs((match.start - start).total_seconds()) <= 10
        assert abs((match.end - end).total_seconds()) <= 20
        assert abs(match.min_magnitude - ref["mag"]) < 0.05
        ours.remove(match)
    # Two ~10 second slivers at magnitude 2.9 on the edge of the Earth's
    # shadow, where the site's sun ephemeris differs slightly from ours.
    assert len(ours) <= 2


def test_look_angles(transits) -> None:
    """The first pass is described from rise to set."""
    first = group_transits(transits)[0].primary
    assert first.satellite.name == "BlueWalker 3"
    assert first.rise.elevation >= 10
    assert first.start <= first.brightest_time <= first.end
    assert first.brightest.elevation >= first.rise.elevation


def test_brightest_possible_is_a_bound(transits) -> None:
    """No pass is brighter than the prefilter's bound for its satellite."""
    for t in transits:
        assert t.min_magnitude >= brightest_possible(t.satellite) - 1e-6


def test_starlink_trains_only_keep_large_groups(satellites, transits) -> None:
    """In train mode, groups of five or fewer satellites are dropped."""
    groups = group_transits(transits, trains=True, catalog_size=len(satellites))
    assert all(g.satellite_count > 5 for g in groups)


def test_temporary_catalog_number(transits) -> None:
    """Objects with CelesTrak's temporary 9-digit catalog numbers are predicted.

    sgp4 rejects numbers that don't fit a TLE (above 339999), and one such
    satellite used to stop the whole prediction.
    """
    original = transits[0].satellite
    temporary = Satellite.from_omm(
        {**original.omm, "NORAD_CAT_ID": "799501674"}, original.magnitude, original.model
    )
    ours = predict_transits([original, temporary], 52.0277, 5.5583, 194, start=START)
    by_id = {
        sat_id: [(t.start, t.end) for t in ours if t.satellite.norad_id == sat_id]
        for sat_id in (original.norad_id, 799501674)
    }
    assert by_id[original.norad_id]
    assert by_id[799501674] == by_id[original.norad_id]
    assert temporary.omm["NORAD_CAT_ID"] == "799501674"


def test_future_epoch_stays_in_window(transits) -> None:
    """A launch prediction is propagated from its epoch to the end of the window.

    It used to be propagated for the full number of days from its epoch, so
    passes days after the window came back.
    """
    original = next(t.satellite for t in transits if t.satellite.name == "BlueWalker 3")
    epoch = START + timedelta(days=2)
    launch = Satellite.from_omm(
        {**original.omm, "EPOCH": epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")},
        original.magnitude,
        original.model,
    )
    ours = predict_transits([launch], 52.0277, 5.5583, 194, start=START, days=5)
    assert ours
    assert all(epoch <= t.start and t.end <= START + timedelta(days=5) for t in ours)


@pytest.mark.parametrize(
    "fields",
    [
        {"EPOCH": "2026-10-09T22:48:38.797056Z"},
        {"EPOCH": "2026-10-09T22:48:38"},
        {"CLASSIFICATION_TYPE": None, "ELEMENT_SET_NO": None, "REV_AT_EPOCH": None},
    ],
)
def test_satrec_accepts_omm_variants(satellites, fields) -> None:
    """Epoch formats and label fields sgp4 is strict about don't stop prediction."""
    original = next(s for s in satellites if s.norad_id == 25544)
    omm_fields = {k: v for k, v in {**original.omm, **fields}.items() if v is not None}
    sat = Satellite.from_omm(omm_fields, original.magnitude)
    error, position, _ = sat.satrec().sgp4(2461323.5, 0.0)
    assert error == 0
    assert 6700 < sum(x * x for x in position) ** 0.5 < 6900


def test_satellite_from_omm() -> None:
    """OMM fields as published by CelesTrak (all strings) are accepted."""
    sat = Satellite.from_omm(
        {
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "EPOCH": "2026-10-09T22:48:38.797056",
            "MEAN_MOTION": "15.48795780",
            "ECCENTRICITY": ".00066723",
            "INCLINATION": "51.6315",
            "RA_OF_ASC_NODE": "92.1575",
            "ARG_OF_PERICENTER": "244.0714",
            "MEAN_ANOMALY": "115.9587",
            "EPHEMERIS_TYPE": "0",
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": "25544",
            "ELEMENT_SET_NO": "999",
            "REV_AT_EPOCH": "58954",
            "BSTAR": ".13073644E-3",
            "MEAN_MOTION_DOT": ".6711E-4",
            "MEAN_MOTION_DDOT": "0",
        },
        -1.75,
    )
    assert sat.norad_id == 25544
    assert sat.epoch == datetime(2026, 10, 9, 22, 48, 38, 797056, tzinfo=UTC)
    assert 400 < sat.perigee_km < 430
    assert not sat.is_starlink
    error, position, _ = sat.satrec().sgp4(2461323.5, 0.0)
    assert error == 0
    assert 6700 < sum(x * x for x in position) ** 0.5 < 6900
