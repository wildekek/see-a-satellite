"""Visible satellite pass prediction.

A port of the prediction pipeline from James Darpinian's "See A Satellite
Tonight" (https://james.darpinian.com/satellites/), used with permission:

* ``sgp4-worker.js``: SGP4 propagation at one-minute steps, TEME rotated to
  Earth-fixed by GMST.
* ``visibility-worker.js``: apparent magnitude (intrinsic magnitude, range,
  atmospheric extinction, phase angle or a constellation-specific model),
  limiting magnitude from the sun's depression, and the requirement that
  sunlight reaches the satellite above the troposphere. A coarse per-minute
  pass is refined at one-second resolution.
* ``index.html``: grouping of overlapping passes and pruning of incidental
  Starlink passes.

Everything is vectorized with numpy; a 5 day prediction for the ~900
satellite catalog takes under a second.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import math

import numpy as np
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

# WGS84 ellipsoid
_A = 6378137.0
_B = 6356752.3142451793
_E2 = 1.0 - (_B * _B) / (_A * _A)
_AU_M = 149597870700.0
_EARTH_MU = 398600.4418  # km^3/s^2

DAYS_TO_CHECK = 5
DEFAULT_MAX_MAGNITUDE = 3.0
MIN_ELEVATION = math.radians(10)
MIN_GRAZING_HEIGHT_M = 15000.0
# Coarse search admits passes up to 2 magnitudes fainter than the cutoff;
# the one-second refinement then applies the real cutoff.
_COARSE_MAGNITUDE_MARGIN = 2.0
# Passes separated by at most this much are grouped together.
_GROUP_GAP_S = 60
_CHUNK_SATS = 48
# Largest catalog number a TLE's five-character (Alpha-5) field can hold.
_MAX_TLE_SATNUM = 339999
# OMM fields sgp4 requires but SGP4 itself doesn't use.
_SGP4_LABEL_DEFAULTS = {
    "CLASSIFICATION_TYPE": "U",
    "OBJECT_ID": "",
    "ELEMENT_SET_NO": "0",
    "REV_AT_EPOCH": "0",
}

# Limiting magnitude by sun angle from zenith. Weaver 1947, "The Visibility
# Of Stars Without Optical Aid", PASP 59, 232, fig. 2, with daylight made a
# bit brighter and truncated at 4.
_LIMITING_MAGNITUDES = np.array([-4, -3.5, -2.5, -1, 0, 1, 2, 3, 4], dtype=float)
_SUN_ANGLES = np.radians([0, 50, 82, 90, 93.2, 95.2, 96.4, 97.5, 100])

# Atmospheric extinction by elevation (Green 1992, International Comet
# Quarterly 14, 55-59; summer table, sea level).
_EXTINCTIONS = np.array(
    [12.68, 8.33, 6.11, 4.78, 3.9, 3.28, 2.82, 2.47, 2.19, 1.97, 1.79, 1.64,
     1.51, 1.4, 1.3, 1.22, 1.15, 1.08, 1.02, 0.97, 0.93, 0.85, 0.78, 0.72,
     0.68, 0.63, 0.55, 0.49, 0.45, 0.41, 0.37, 0.34, 0.32, 0.32]
)  # fmt: skip
_EXTINCTION_ELEVATIONS = np.radians(
    90 - np.array(
        [90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74,
         73, 72, 71, 70, 68, 66, 64, 62, 60, 55, 50, 45, 40, 30, 20, 10, 0],
        dtype=float,
    )
)  # fmt: skip


@dataclass(slots=True)
class Satellite:
    """A satellite with its orbital elements and brightness."""

    norad_id: int
    name: str
    # Intrinsic magnitude at 1000 km range, the convention of the site's
    # catalog. None if only a brightness model applies.
    magnitude: float | None
    # Constellation-specific brightness model, see _BRIGHTNESS_MODELS.
    model: str | None
    # CCSDS Orbit Mean-elements Message fields as strings, as published by
    # CelesTrak and Space-Track (OBJECT_NAME, EPOCH, MEAN_MOTION, ...).
    omm: Mapping[str, str]
    epoch: datetime
    prediction: bool = False

    @classmethod
    def from_omm(
        cls,
        fields: Mapping[str, object],
        magnitude: float | None,
        model: str | None = None,
        name: str | None = None,
    ) -> Satellite:
        """Build from OMM fields (values as published, strings or numbers)."""
        omm_fields = {k: str(v) for k, v in fields.items()}
        omm_fields.setdefault("EPHEMERIS_TYPE", "0")
        name = name or omm_fields["OBJECT_NAME"]
        epoch = datetime.fromisoformat(omm_fields["EPOCH"].removesuffix("Z"))
        return cls(
            norad_id=int(omm_fields["NORAD_CAT_ID"]),
            name=name,
            magnitude=magnitude,
            model=model,
            omm=omm_fields,
            epoch=epoch.replace(tzinfo=UTC),
            prediction=name.endswith(" prediction"),
        )

    @property
    def is_starlink(self) -> bool:
        """Whether this is a Starlink satellite."""
        return self.name.upper().startswith("STARLINK")

    @property
    def perigee_km(self) -> float:
        """Perigee height above a spherical Earth."""
        n = float(self.omm["MEAN_MOTION"]) * 2 * math.pi / 86400
        a = (_EARTH_MU / (n * n)) ** (1 / 3)
        return float(a * (1 - float(self.omm["ECCENTRICITY"])) - 6378.137)

    def satrec(self) -> Satrec:
        """SGP4 propagator for this satellite."""
        fields = dict(self.omm)
        # CelesTrak gives new launches temporary 9-digit catalog numbers until
        # they are cataloged. sgp4 rejects numbers that don't fit a TLE, but
        # SGP4 only uses the number as a label.
        if int(fields["NORAD_CAT_ID"]) > _MAX_TLE_SATNUM:
            fields["NORAD_CAT_ID"] = "0"
        # sgp4 also requires other labels and parses the epoch with one strict
        # format (no "Z", six fractional digits).
        for key, default in _SGP4_LABEL_DEFAULTS.items():
            fields.setdefault(key, default)
        fields["EPOCH"] = self.epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")
        sat = Satrec()
        omm.initialize(sat, fields)
        return sat


@dataclass(slots=True)
class LookAngle:
    """Where to look: degrees clockwise from north, degrees above horizon."""

    azimuth: float
    elevation: float


@dataclass(slots=True)
class Transit:
    """One satellite's visible pass."""

    satellite: Satellite
    start: datetime
    end: datetime
    min_magnitude: float
    before_sunset: bool
    limiting_magnitude: float
    rise: LookAngle
    brightest: LookAngle
    brightest_time: datetime
    set: LookAngle


@dataclass(slots=True)
class TransitGroup:
    """Passes close enough in time to be shown as one event."""

    start: datetime
    end: datetime
    min_magnitude: float
    limiting_magnitude: float
    before_sunset: bool
    transits: list[Transit] = field(default_factory=list)

    @property
    def primary(self) -> Transit:
        """The first satellite of the group (the one the site names)."""
        return self.transits[0]

    @property
    def satellite_count(self) -> int:
        """Number of satellites; a Starlink launch prediction stands for 60."""
        return sum(
            60 if t.satellite.prediction and t.satellite.is_starlink else 1 for t in self.transits
        )

    @property
    def bright(self) -> bool:
        """Whether the site would label this pass as bright."""
        return (
            self.min_magnitude < 1
            and not self.before_sunset
            and self.limiting_magnitude - self.min_magnitude > 3
        )


# --- Geometry helpers --------------------------------------------------------


def _jd_parts(when: datetime) -> tuple[float, float]:
    """Julian date split into whole day and fraction, UTC."""
    ts = when.timestamp()
    days = ts / 86400.0
    whole = math.floor(days)
    return 2440587.5 + whole, days - whole


def _gmst(jd: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal time (IAU-82, as used by SGP4's TEME frame)."""
    t = (jd - 2451545.0) / 36525.0
    seconds = (
        -6.2e-6 * t**3 + 0.093104 * t**2 + (876600.0 * 3600 + 8640184.812866) * t + 67310.54841
    )
    return np.radians((seconds % 86400.0) / 240.0)


def _rotate_z(vec: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate inertial vectors (..., 3) into the Earth-fixed frame by angle."""
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty_like(vec)
    out[..., 0] = vec[..., 0] * c + vec[..., 1] * s
    out[..., 1] = -vec[..., 0] * s + vec[..., 1] * c
    out[..., 2] = vec[..., 2]
    return out


def _sun_ecef(jd: np.ndarray) -> np.ndarray:
    """Sun position in Earth-fixed meters (Astronomical Almanac, ~0.01 deg)."""
    n = jd - 2451545.0
    mean_lon = np.radians(280.460 + 0.9856474 * n)
    g = np.radians(357.528 + 0.9856003 * n)
    lam = mean_lon + np.radians(1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.0000004 * n)
    dist = (1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)) * _AU_M
    eci = np.stack(
        [
            dist * np.cos(lam),
            dist * np.cos(eps) * np.sin(lam),
            dist * np.sin(eps) * np.sin(lam),
        ],
        axis=-1,
    )
    return _rotate_z(eci, _gmst(jd))


# The ellipsoid grown by the minimum grazing height on both axes; it stays
# within 2 cm of the true surface at that geodetic height.
_GRAZING_RADII = np.array([_A, _A, _B]) + MIN_GRAZING_HEIGHT_M


def _sunlit(origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Whether rays towards the sun clear the atmosphere (MIN_GRAZING_HEIGHT_M).

    The site checks the lowest height along the ray with Cesium's
    IntersectionTests.grazingAltitudeLocation. Equivalently, the ray must
    miss the grown ellipsoid, which is a quadratic in scaled space where that
    ellipsoid is the unit sphere.
    """
    so, sd = origin / _GRAZING_RADII, direction / _GRAZING_RADII
    a = np.einsum("ij,ij->i", sd, sd)
    b = np.einsum("ij,ij->i", so, sd)
    c = np.einsum("ij,ij->i", so, so) - 1.0
    # Shadowed if the origin is inside, or the ray heads inwards and meets it.
    return (c >= 0) & ((b >= 0) | (b * b - a * c <= 0))


def _look_angle(observer: np.ndarray, lat: float, lon: float, pos: np.ndarray) -> LookAngle:
    """Azimuth/elevation of an Earth-fixed position from the observer."""
    d = pos - observer
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    east = -so * d[0] + co * d[1]
    north = -sl * co * d[0] - sl * so * d[1] + cl * d[2]
    up = cl * co * d[0] + cl * so * d[1] + sl * d[2]
    azimuth = math.degrees(math.atan2(east, north)) % 360
    elevation = math.degrees(math.atan2(up, math.hypot(east, north)))
    return LookAngle(round(azimuth, 1), round(elevation, 1))


# --- Brightness --------------------------------------------------------------


def _distance_term(distance_m: np.ndarray) -> np.ndarray:
    return 5 * np.log10(distance_m * 1e-6)


def _starlink_cubic(
    c0: float, c1: float, c2: float, c3: float
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def model(phase: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
        a = np.clip(np.degrees(phase), 15, 155)
        return c0 + c1 * a + c2 * a * a + c3 * a * a * a + _distance_term(distance_m)

    return model


def _bluebird(phase: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    # AST SpaceMobile BlueBird linear phase trend (Cole et al. 2025,
    # arXiv:2505.05820 fig. 4), floored at the observed peak of mag 0.5.
    return np.maximum(0.5, 2.1 + 0.022 * np.degrees(phase)) + _distance_term(distance_m)


# Constellation-specific models selected by the catalog's MODEL field
# (Mallama et al. 2025, arXiv:2502.03651 table 1; arXiv:2405.12007).
_BRIGHTNESS_MODELS = {
    "bluebird": _bluebird,
    "starlink-mini": _starlink_cubic(5.822, -0.00879, 8.483e-4, -5.784e-6),
    "starlink-dtc": _starlink_cubic(3.365, 0.06376, -1.716e-4, -1.139e-6),
    "starlink-raising": _starlink_cubic(3.365 - 1.89, 0.06376, -1.716e-4, -1.139e-6),
}


_MIN_EXTINCTION = float(_EXTINCTIONS.min())
_PHASE_GRID = np.radians(np.arange(0.0, 181.0))


def brightest_possible(sat: Satellite) -> float:
    """Return a lower bound on the apparent magnitude a satellite can reach.

    Seen at zenith from perigee with the most favorable phase; used to skip
    satellites that can never get bright enough.
    """
    perigee_m = max(sat.perigee_km, 100.0) * 1000.0
    model = _BRIGHTNESS_MODELS.get(sat.model or "")
    if model is not None:
        best = float(model(_PHASE_GRID, np.full_like(_PHASE_GRID, perigee_m)).min())
    elif sat.magnitude is None:
        return math.inf
    else:
        best = sat.magnitude + 5 * math.log10(perigee_m * 1e-6)
    return best + _MIN_EXTINCTION


@dataclass(slots=True)
class _Observer:
    lat: float
    lon: float
    position: np.ndarray
    normal: np.ndarray


def _make_observer(lat_deg: float, lon_deg: float, height_m: float) -> _Observer:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
    position = np.array(
        [
            (n + height_m) * math.cos(lat) * math.cos(lon),
            (n + height_m) * math.cos(lat) * math.sin(lon),
            (n * (1 - _E2) + height_m) * math.sin(lat),
        ]
    )
    normal = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    return _Observer(lat, lon, position, normal)


def _magnitudes(
    observer: _Observer,
    pos: np.ndarray,
    sun: np.ndarray,
    intrinsic: np.ndarray,
    model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apparent magnitude of satellites at flat arrays of positions.

    Returns (magnitude, sun_angle). Magnitude is +inf where the satellite is
    below the minimum elevation, too faint for the sky brightness, or in the
    Earth's shadow.
    """
    user_to_sat = pos - observer.position
    distance = np.linalg.norm(user_to_sat, axis=-1)
    elevation = np.pi / 2 - np.arccos(np.clip(user_to_sat @ observer.normal / distance, -1, 1))
    sun_norm = np.linalg.norm(sun, axis=-1)
    sun_angle = np.arccos(np.clip(sun @ observer.normal / sun_norm, -1, 1))
    mag = np.full(len(pos), np.inf)

    above = (elevation >= MIN_ELEVATION) & np.isfinite(distance)
    if not above.any():
        return mag, sun_angle
    idx = np.nonzero(above)[0]
    sat_to_user = -user_to_sat[idx]
    sat_to_sun = sun[idx] - pos[idx]
    cos_phase = np.einsum("ij,ij->i", sat_to_user, sat_to_sun) / (
        distance[idx] * np.linalg.norm(sat_to_sun, axis=-1)
    )
    phase = np.arccos(np.clip(cos_phase, -1, 1))
    extinction = np.interp(elevation[idx], _EXTINCTION_ELEVATIONS, _EXTINCTIONS)

    # Generic: intrinsic magnitude at 1000 km + range + extinction + phase
    # term of a diffuse sphere.
    f = (np.sin(phase) + (np.pi - phase) * np.cos(phase)) / np.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        m = intrinsic[idx] + _distance_term(distance[idx]) + extinction - 2.5 * np.log10(f)
    m = np.where(np.isfinite(m), m, np.inf)
    models = model[idx]
    for name, fn in _BRIGHTNESS_MODELS.items():
        sel = models == name
        if sel.any():
            m[sel] = fn(phase[sel], distance[idx][sel]) + extinction[sel]

    limiting = np.interp(sun_angle[idx], _SUN_ANGLES, _LIMITING_MAGNITUDES, left=-100.0)
    m[m > limiting - 0.3] = np.inf

    lit = np.isfinite(m)
    if lit.any():
        li = np.nonzero(lit)[0]
        m[li[~_sunlit(pos[idx][li], sat_to_sun[li])]] = np.inf
    mag[idx] = m
    return mag, sun_angle


def _limiting_magnitude(sun_angle: float) -> float:
    return float(np.interp(sun_angle, _SUN_ANGLES, _LIMITING_MAGNITUDES, left=-100.0))


# --- Prediction ---------------------------------------------------------------


@dataclass(slots=True)
class _Batch:
    """Satellites sharing a propagation time grid."""

    first_minute: int
    satellites: list[Satellite]


def _propagate(satrecs: Sequence[Satrec], jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """Earth-fixed positions in meters, shape (sats, times, 3); NaN on error."""
    err, teme, _ = SatrecArray(list(satrecs)).sgp4(jd, fr)
    teme = teme * 1000.0
    teme[err != 0] = np.nan
    return _rotate_z(teme, _gmst(jd + fr)[None, :])


def _transits_for_satellite(
    sat: Satellite,
    observer: _Observer,
    positions: np.ndarray,
    coarse_mag: np.ndarray,
    *,
    t0: datetime,
    jd0: tuple[float, float],
    max_magnitude: float,
) -> list[Transit]:
    """Refine the coarse per-minute visibility of one satellite."""
    visible = coarse_mag < max_magnitude + _COARSE_MAGNITUDE_MARGIN
    # Rising and falling edges of the per-minute visibility; a pass still in
    # progress at the end of the window is dropped, as on the site.
    edges = np.diff(visible.astype(np.int8))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    if visible[0]:
        starts.insert(0, 0)
    ends = list(np.nonzero(edges == -1)[0] + 1)
    transits: list[Transit] = []
    intrinsic = np.array([sat.magnitude if sat.magnitude is not None else np.inf])
    model = np.array([sat.model or ""], dtype=object)
    n_minutes = len(positions)
    for start in starts:
        later = [e for e in ends if e > start]
        if not later:
            break
        end = later[0]
        first = max(start - 1, 0)
        seconds = np.arange(first * 60, end * 60, dtype=float)
        minutes = (seconds // 60).astype(int)
        frac = (seconds % 60 / 60)[:, None]
        nxt = np.minimum(minutes + 1, n_minutes - 1)
        pos = positions[minutes] * (1 - frac) + positions[nxt] * frac
        jd = jd0[0] + (jd0[1] + seconds / 86400.0)
        sun = _sun_ecef(jd)
        mag, sun_angle = _magnitudes(
            observer,
            pos,
            sun,
            np.repeat(intrinsic, len(pos)),
            np.repeat(model, len(pos)),
        )
        fine = mag < max_magnitude
        if not fine.any():
            continue
        rise = int(np.argmax(fine))
        after = np.nonzero(~fine[rise:])[0]
        fall = rise + int(after[0]) if len(after) else len(fine)
        brightest = rise + int(np.argmin(mag[rise:fall]))
        start_time = t0 + timedelta(seconds=float(seconds[rise]))
        end_time = t0 + timedelta(seconds=float(seconds[fall] if fall < len(fine) else end * 60))
        last_visible = max(fall - 1, rise)
        transits.append(
            Transit(
                satellite=sat,
                start=start_time,
                end=end_time,
                min_magnitude=float(mag[brightest]),
                before_sunset=bool(sun_angle[rise] < np.pi / 2),
                limiting_magnitude=_limiting_magnitude(float(sun_angle[rise])),
                rise=_look_angle(observer.position, observer.lat, observer.lon, pos[rise]),
                brightest=_look_angle(
                    observer.position, observer.lat, observer.lon, pos[brightest]
                ),
                brightest_time=t0 + timedelta(seconds=float(seconds[brightest])),
                set=_look_angle(observer.position, observer.lat, observer.lon, pos[last_visible]),
            )
        )
    return transits


def predict_transits(
    satellites: Sequence[Satellite],
    latitude: float,
    longitude: float,
    elevation_m: float,
    *,
    start: datetime | None = None,
    days: float = DAYS_TO_CHECK,
    max_magnitude: float = DEFAULT_MAX_MAGNITUDE,
) -> list[Transit]:
    """Predict every visible pass of the given satellites, sorted by start."""
    if start is None:
        start = datetime.now(UTC) - timedelta(minutes=5)
    start = start.astimezone(UTC).replace(microsecond=0)
    observer = _make_observer(latitude, longitude, elevation_m)
    n_minutes = int(days * 24 * 60)

    # Satellites whose elements are from the future (launch predictions) are
    # propagated from their epoch to the end of the window.
    batches: dict[int, list[Satellite]] = {}
    bounds: dict[int, float] = {}
    for sat in satellites:
        if sat.magnitude is None and not sat.model:
            continue
        first = max(0.0, (sat.epoch - start).total_seconds() / 60)
        if first >= n_minutes:
            continue
        # Skip satellites that can never get bright enough.
        bound = brightest_possible(sat)
        if bound >= max_magnitude:
            continue
        bounds[id(sat)] = bound
        batches.setdefault(int(first), []).append(sat)

    transits: list[Transit] = []
    for first_minute, sats in batches.items():
        t0 = start + timedelta(minutes=first_minute)
        jd_whole, jd_frac = _jd_parts(t0)
        batch_minutes = n_minutes - first_minute
        offsets = np.arange(batch_minutes, dtype=float) / 1440.0
        jd = np.full(batch_minutes, jd_whole)
        fr = jd_frac + offsets
        sun = _sun_ecef(jd + fr)
        sun_angle = np.arccos(np.clip(sun @ observer.normal / np.linalg.norm(sun, axis=-1), -1, 1))
        sky_limit = np.interp(sun_angle, _SUN_ANGLES, _LIMITING_MAGNITUDES, left=-100.0) - 0.3
        # Chunks of similar brightness share the minutes when the sky is dark
        # enough for their brightest member; the rest are never visible.
        by_brightness = sorted(sats, key=lambda s: bounds[id(s)])
        for i in range(0, len(by_brightness), _CHUNK_SATS):
            chunk = by_brightness[i : i + _CHUNK_SATS]
            dark = sky_limit >= bounds[id(chunk[0])]
            # Neighboring minutes too: refinement interpolates up to them.
            dark[1:] |= dark[:-1].copy()
            dark[:-1] |= dark[1:].copy()
            minutes = np.nonzero(dark)[0]
            if not len(minutes):
                continue
            satrecs = [s.satrec() for s in chunk]
            positions = np.full((len(chunk), batch_minutes, 3), np.nan)
            positions[:, minutes] = _propagate(satrecs, jd[minutes], fr[minutes])
            intrinsic = np.repeat(
                [s.magnitude if s.magnitude is not None else np.inf for s in chunk],
                len(minutes),
            )
            model = np.repeat(np.array([s.model or "" for s in chunk], dtype=object), len(minutes))
            dark_mag, _ = _magnitudes(
                observer,
                positions[:, minutes].reshape(-1, 3),
                np.tile(sun[minutes], (len(chunk), 1)),
                intrinsic,
                model,
            )
            mag = np.full((len(chunk), batch_minutes), np.inf)
            mag[:, minutes] = dark_mag.reshape(len(chunk), len(minutes))
            for j, sat in enumerate(chunk):
                if not np.any(mag[j] < max_magnitude + _COARSE_MAGNITUDE_MARGIN):
                    continue
                transits.extend(
                    _transits_for_satellite(
                        sat,
                        observer,
                        positions[j],
                        mag[j],
                        t0=t0,
                        jd0=(jd_whole, jd_frac),
                        max_magnitude=max_magnitude,
                    )
                )
    # Passes starting together keep the catalog's order.
    order = {id(s): i for i, s in enumerate(satellites)}
    transits.sort(key=lambda t: (t.start, order[id(t.satellite)]))
    return transits


def group_transits(
    transits: Sequence[Transit], trains: bool = False, catalog_size: int = 0
) -> list[TransitGroup]:
    """Group overlapping passes as the site does.

    Normally, a few Starlinks tagging along with another satellite are noise
    and dropped. With ``trains`` (the site's Starlink page) only groups of
    more than five satellites are kept, unless the catalog itself is tiny.
    """

    def keep(group: TransitGroup) -> bool:
        return not trains or catalog_size < 6 or group.satellite_count > 5

    groups: list[TransitGroup] = []
    current: TransitGroup | None = None
    for transit in sorted(transits, key=lambda t: t.start):
        if current and (transit.start - current.end).total_seconds() > _GROUP_GAP_S:
            if keep(current):
                groups.append(current)
            current = None
        if current is None:
            current = TransitGroup(
                start=transit.start,
                end=transit.end,
                min_magnitude=transit.min_magnitude,
                limiting_magnitude=transit.limiting_magnitude,
                before_sunset=transit.before_sunset,
                transits=[transit],
            )
        else:
            current.transits.append(transit)
            _absorb(current, transit)
    if current is not None and keep(current):
        groups.append(current)
    if trains:
        return groups

    result = []
    for group in groups:
        starlinks = sum(t.satellite.is_starlink for t in group.transits)
        if 0 < starlinks < 5:
            kept = [t for t in group.transits if not t.satellite.is_starlink]
            if not kept:
                continue
            pruned = TransitGroup(
                start=kept[0].start,
                end=kept[0].end,
                min_magnitude=kept[0].min_magnitude,
                limiting_magnitude=kept[0].limiting_magnitude,
                before_sunset=kept[0].before_sunset,
                transits=kept,
            )
            for t in kept[1:]:
                _absorb(pruned, t)
            result.append(pruned)
        else:
            result.append(group)
    return result


def _absorb(group: TransitGroup, transit: Transit) -> None:
    group.start = min(group.start, transit.start)
    group.end = max(group.end, transit.end)
    group.min_magnitude = min(group.min_magnitude, transit.min_magnitude)
    group.limiting_magnitude = min(group.limiting_magnitude, transit.limiting_magnitude)
    group.before_sunset = group.before_sunset or transit.before_sunset
