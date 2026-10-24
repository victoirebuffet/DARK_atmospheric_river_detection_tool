#!/usr/bin/env python3
"""Portable DARK atmospheric-river detector.

One configuration drives ERA5, MERRA-2, model output, and the optional
children classification.  Input may contain IVT directly or its two horizontal
components.  The implementation deliberately has no dataset-specific paths.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, splprep, splev
from skimage.morphology import skeletonize
from geopy.distance import geodesic
import networkx as nx


def _read_config(path: str) -> dict[str, Any]:
    """Read JSON everywhere; YAML too when PyYAML is installed."""
    text = Path(path).read_text()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config needs PyYAML; use JSON or install pyyaml.") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail early for configuration errors, without touching input data."""
    missing = {"dataset", "output_dir", "years", "data", "percentile", "dark"} - set(cfg)
    if missing:
        raise ValueError(f"Configuration is missing required keys: {sorted(missing)}")
    years = cfg["years"]
    if not isinstance(years, (list, tuple)) or len(years) != 2 or int(years[0]) > int(years[1]):
        raise ValueError("years must be [first_year, last_year].")
    data = cfg["data"]
    if data.get("grid", "latlon") != "latlon":
        raise ValueError("data.grid must be 'latlon' for now. Projected grids (for example MAR x/y) need the forthcoming projected-grid adapter.")
    kind = data.get("ivt_type")
    if kind not in {"scalar", "vector"}:
        raise ValueError("data.ivt_type must be 'scalar' or 'vector'.")
    if not any(data.get(key) for key in ("file_pattern", "files", "u_file_pattern")):
        raise ValueError("data needs file_pattern, files, or u_file_pattern/v_file_pattern.")
    if kind == "scalar" and not data.get("ivt_variable"):
        raise ValueError("Scalar IVT needs data.ivt_variable.")
    if kind == "vector" and not (data.get("u_variable") and data.get("v_variable")):
        raise ValueError("Vector IVT needs data.u_variable and data.v_variable.")
    if bool(data.get("u_file_pattern")) != bool(data.get("v_file_pattern")):
        raise ValueError("Provide both data.u_file_pattern and data.v_file_pattern.")
    periodic = data.get("periodic_longitude", "auto")
    if not isinstance(periodic, bool) and periodic != "auto":
        raise ValueError("data.periodic_longitude must be true, false, or 'auto'.")
    if not 0 < float(cfg["percentile"].get("quantile", -1)) < 1:
        raise ValueError("percentile.quantile must be strictly between 0 and 1.")
    for key in ("min_area_km2", "min_length_km"):
        if float(cfg["dark"].get(key, 0)) <= 0: raise ValueError(f"dark.{key} must be positive.")
    reference_years = cfg["percentile"].get("reference_years", years)
    if not isinstance(reference_years, (list, tuple)) or len(reference_years) != 2 or int(reference_years[0]) > int(reference_years[1]):
        raise ValueError("percentile.reference_years must be [first_year, last_year].")
    adaptive = cfg.get("adaptive_threshold", {})
    if adaptive.get("enabled"):
        source = adaptive.get("data", {})
        if not source.get("file_pattern") or not source.get("variable"):
            raise ValueError("Enabled adaptive_threshold needs adaptive_threshold.data.file_pattern and .variable.")
        if float(adaptive.get("smoothing_sigma_years", 2)) < 0:
            raise ValueError("adaptive_threshold.smoothing_sigma_years cannot be negative.")
    children = cfg.get("children", {})
    if children.get("enabled") and not {"land_mask_file", "land_mask_variable"} <= set(children):
        raise ValueError("Enabled children need children.land_mask_file and children.land_mask_variable.")


def _normalise(ds: xr.Dataset, data_cfg: dict[str, Any], require_time: bool = True) -> xr.Dataset:
    """Rename configurable coordinate/dimension names to time/lat/lon.

    `coordinates` names coordinate variables and `dimensions` names their
    underlying dimensions, which are often different in model output.
    """
    coordinates = data_cfg.get("coordinates", {})
    dimensions = data_cfg.get("dimensions", {})
    defaults = {"lat": "latitude", "lon": "longitude", "time": "valid_time"}
    renames: dict[str, str] = {}
    for standard in ("lat", "lon", "time"):
        source_coordinate = coordinates.get(standard, standard if standard in ds.coords else defaults[standard])
        source_dimension = dimensions.get(standard, source_coordinate)
        if source_dimension in ds.dims and source_dimension != standard: renames[source_dimension] = standard
        if source_coordinate in ds.coords and source_coordinate != standard: renames[source_coordinate] = standard
    ds = ds.rename(renames)
    for dim in ("expver", "member", "number"):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)
    required = {"lat", "lon"} | ({"time"} if require_time else set())
    missing = required - set(ds.coords)
    if missing:
        raise ValueError(f"Input is missing coordinates: {sorted(missing)}")
    if ds.lat.ndim != 1 or ds.lon.ndim != 1:
        raise ValueError("DARK currently needs one-dimensional rectilinear latitude/longitude coordinates. Use the projected-grid adapter for MAR/curvilinear grids.")
    if data_cfg.get("longitude_domain", "native") == "-180_180" and float(ds.lon.min()) >= 0:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    return ds


def _region(ds: xr.Dataset, data_cfg: dict[str, Any]) -> xr.Dataset:
    """Crop to an optional [min, max] latitude/longitude region."""
    region = data_cfg.get("region")
    if not region: return ds
    if "latitude" in region:
        lo, hi = region["latitude"]; lat = ds.lat.values
        ds = ds.sel(lat=slice(hi, lo) if lat[0] > lat[-1] else slice(lo, hi))
    if "longitude" in region:
        lo, hi = region["longitude"]
        if lo > hi:  # a region crossing the longitude seam
            ds = xr.concat([ds.sel(lon=slice(lo, None)), ds.sel(lon=slice(None, hi))], "lon")
        else: ds = ds.sel(lon=slice(lo, hi))
    return ds


def _time_subset(ivt: xr.DataArray, data_cfg: dict[str, Any]) -> xr.DataArray:
    """Optional inclusive [start, end] window, useful for validation runs."""
    window = data_cfg.get("time_slice")
    return ivt.sel(time=slice(window[0], window[1])) if window else ivt


def open_percentile(path: str | Path) -> xr.DataArray:
    """Read both the new name and the legacy `ivt_pctl` convention."""
    ds = xr.open_dataset(path)
    for name in ("ivt_percentile", "ivt_pctl"):
        if name in ds: return ds[name]
    raise KeyError(f"No IVT percentile variable in {path}; expected ivt_percentile or ivt_pctl.")


def _open_ivt(year: int, cfg: dict[str, Any]) -> xr.DataArray:
    data = cfg["data"]
    ivt_type = data.get("ivt_type")
    if ivt_type not in {"scalar", "vector"}:
        raise ValueError("data.ivt_type must be explicitly set to 'scalar' or 'vector'.")
    values = {"year": year, "member": data.get("member", "")}
    # CESM-style output is commonly split in decade/multi-year files rather
    # than one file per year.  `files` is therefore an explicit list of such
    # files; only the requested year is retained after combining them.
    if data.get("files"):
        paths = [str(p).format(**values) for p in data["files"]]
        ds = _region(_normalise(xr.open_mfdataset(paths, combine="by_coords"), data), data)
        ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31T23:59:59"))
        if not ds.sizes.get("time", 0):
            raise ValueError(f"No input timestamps found for year {year}")
        if ivt_type == "scalar":
            return _time_subset(ds[data["ivt_variable"]].rename("IVT").transpose("time", "lat", "lon"), data)
        return _time_subset(np.hypot(ds[data["u_variable"]], ds[data["v_variable"]]).rename("IVT").transpose("time", "lat", "lon"), data)
    pattern = data.get("file_pattern")
    if pattern:
        ds = _region(_normalise(xr.open_dataset(pattern.format(**values)), data), data)
        if ivt_type == "scalar":
            return _time_subset(ds[data["ivt_variable"]].rename("IVT").transpose("time", "lat", "lon"), data)
        if "u_variable" not in data or "v_variable" not in data:
            raise ValueError("Vector IVT in one file needs data.u_variable and data.v_variable.")
        return _time_subset(np.hypot(ds[data["u_variable"]], ds[data["v_variable"]]).rename("IVT").transpose("time", "lat", "lon"), data)
    if ivt_type != "vector":
        raise ValueError("Separate u_file_pattern/v_file_pattern are only valid with data.ivt_type: vector.")
    u = _region(_normalise(xr.open_dataset(data["u_file_pattern"].format(**values)), data), data)
    v = _region(_normalise(xr.open_dataset(data["v_file_pattern"].format(**values)), data), data)
    # ERA5 component files can differ by an endpoint or coordinate rounding;
    # their common grid is the valid IVT grid (and matches the legacy code).
    u, v = xr.align(u, v, join="inner")
    return _time_subset(np.hypot(u[data["u_variable"]], v[data["v_variable"]]).rename("IVT").transpose("time", "lat", "lon"), data)


def _area(lat: np.ndarray, lon: np.ndarray, ellipsoidal: bool = False) -> np.ndarray:
    # Works for ascending or descending regular lat/lon grids.
    if ellipsoidal:
        ylat, xlon = np.meshgrid(lat, lon, indexing="ij")
        a, b = 6_378_137.0, 6_356_752.3142
        e2 = 1.0 - (b ** 2 / a ** 2); lat_gc = np.arctan((1.0 - e2) * np.tan(np.deg2rad(ylat)))
        radius = (a * np.sqrt(1.0 - e2)) / np.sqrt(1.0 - e2 * np.cos(lat_gc) ** 2)
        return abs(np.deg2rad(np.gradient(ylat, axis=0)) * radius * np.deg2rad(np.gradient(xlon, axis=1)) * radius * np.cos(np.deg2rad(ylat)))
    r = 6_371_008.8
    dlat = np.deg2rad(np.gradient(lat))[:, None]
    dlon = abs(np.deg2rad(np.gradient(lon)).mean())
    cell_area = abs(r * r * dlat * dlon * np.cos(np.deg2rad(lat))[:, None])
    return np.broadcast_to(cell_area, (len(lat), len(lon))).copy()


def periodic_longitude(data_cfg: dict[str, Any], lon: np.ndarray) -> bool:
    """Infer whether longitude closes on itself, unless explicitly overridden.

    Global grids normally end near 360° (starting near 0°) or near +180°
    (starting near -180°).  A tolerance of two grid cells accommodates both
    cell-centre grids (e.g. 0.125..359.875) and coarse model grids.
    """
    requested = data_cfg.get("periodic_longitude", "auto")
    if isinstance(requested, bool): return requested
    values = np.sort(np.mod(np.asarray(lon, dtype=float), 360.0))
    if len(values) < 2: return False
    gaps = np.diff(np.r_[values, values[0] + 360.0])
    spacing = float(np.nanmedian(gaps))
    if not np.isfinite(spacing) or spacing <= 0: return False
    tolerance = max(1e-6, 2.0 * spacing)
    # A seam-crossing regional selection can have both 0° and 360° values;
    # its missing middle is revealed by one gap far larger than cell spacing.
    closes_without_hole = float(np.nanmax(gaps)) <= 2.5 * spacing
    zero_360 = values[0] <= tolerance and values[-1] >= 360.0 - tolerance
    signed_180 = values[0] <= tolerance and values[-1] >= 360.0 - tolerance
    return bool(closes_without_hole and (zero_360 or signed_180))


class _UF:
    def __init__(self, n: int): self.p = np.arange(n + 1)
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b: self.p[b] = a


def label_periodic(mask: np.ndarray, periodic_longitude: bool = True) -> tuple[np.ndarray, int]:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if not n or not periodic_longitude: return labels, n
    uf = _UF(n)
    for y in range(mask.shape[0]):
        if labels[y, 0]:
            for yy in (y - 1, y, y + 1):
                if 0 <= yy < mask.shape[0] and labels[yy, -1]: uf.union(labels[y, 0], labels[yy, -1])
    roots = np.array([uf.find(i) for i in range(n + 1)])
    ids = {root: i + 1 for i, root in enumerate(np.unique(roots[1:]))}
    remap = np.array([0] + [ids[roots[i]] for i in range(1, n + 1)])
    out = remap[labels]
    return out, len(ids)


def _filter_area(mask: np.ndarray, area: np.ndarray, minimum_km2: float, periodic_longitude: bool = True) -> np.ndarray:
    labels, n = label_periodic(mask, periodic_longitude)
    if not n: return np.zeros_like(mask, bool)
    sizes = np.bincount(labels.ravel(), weights=area.ravel(), minlength=n + 1) / 1e6
    return np.isin(labels, np.flatnonzero(sizes >= minimum_km2)[1:])


def _roll_by_largest_gap(mask: np.ndarray) -> tuple[np.ndarray, int]:
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(cols): return mask, 0
    gaps = np.diff(np.r_[cols, cols[0] + mask.shape[1]]) - 1
    if gaps.max() <= 0: return mask, 0
    seam = (cols[int(gaps.argmax())] + 1 + int(gaps.max()) // 2) % mask.shape[1]
    return np.roll(mask, -seam, axis=1), -seam


def _legacy_centerline_length(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray, periodic_longitude: bool = True) -> float:
    """The PCA/spline centreline algorithm used by the reference scripts."""
    rolled, shift = _roll_by_largest_gap(mask) if periodic_longitude else (mask, 0); lon_rolled = np.roll(lon, shift)
    skeleton = skeletonize(rolled); graph = nx.Graph()
    for y, x in np.argwhere(skeleton):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and 0 <= y + dy < skeleton.shape[0] and 0 <= x + dx < skeleton.shape[1] and skeleton[y + dy, x + dx]: graph.add_edge((y, x), (y + dy, x + dx))
    endpoints = [node for node in graph if graph.degree[node] == 1]
    paths = []
    for start_index, start in enumerate(endpoints):
        for end in endpoints[start_index + 1:]:
            try: paths.append(nx.shortest_path(graph, start, end))
            except nx.NetworkXNoPath: pass
    if paths:
        points = np.argwhere(rolled); centred = points - points.mean(axis=0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0] if len(points) > 1 else np.array([1., 0.])
        def score(path):
            path = np.asarray(path); vector = path[-1] - path[0]
            return np.linalg.norm(np.diff(path, axis=0), axis=1).sum() * abs(np.dot(vector / (np.linalg.norm(vector) + 1e-9), axis)) * (np.ptp(path[:, 0]) / rolled.shape[0])
        centreline = np.asarray(max(paths, key=score), dtype=float)
    else:
        ys, xs = np.where(rolled.any(axis=1))[0], np.where(rolled.any(axis=0))[0]
        if not len(ys) or not len(xs): return 0.0
        centreline = np.column_stack([np.arange(ys.min(), ys.max() + 1), np.full(ys.max() - ys.min() + 1, round(xs.mean()))]).astype(float)
    # Preserve the reference algorithm's small end extension before smoothing.
    if len(centreline) >= 2:
        def extend(first, second):
            direction = first - second; direction /= np.linalg.norm(direction) + 1e-9
            for step in range(1, 26):
                point = np.rint(first + step * direction).astype(int)
                if not (0 <= point[0] < rolled.shape[0] and 0 <= point[1] < rolled.shape[1]): break
                if rolled[tuple(point)]: continue
                return np.rint(first + (step - 1) * direction)
            return first
        centreline = np.vstack([extend(centreline[0], centreline[1]), centreline[1:-1], extend(centreline[-1], centreline[-2])])
    if len(centreline) >= 3:
        try:
            spline, _ = splprep([centreline[:, 0], centreline[:, 1]], s=300)
            centreline = np.column_stack(splev(np.linspace(0, 1, 200), spline))
        except Exception: pass
    centreline[:, 0] = np.clip(centreline[:, 0], 0, len(lat) - 1); centreline[:, 1] = np.mod(centreline[:, 1], len(lon))
    grid = (np.arange(len(lat)), np.arange(len(lon)))
    plat = RegularGridInterpolator(grid, np.tile(lat[:, None], (1, len(lon))), bounds_error=False, fill_value=np.nan)(centreline)
    sin = RegularGridInterpolator(grid, np.tile(np.sin(np.deg2rad(lon_rolled))[None, :], (len(lat), 1)), bounds_error=False, fill_value=np.nan)(centreline)
    cos = RegularGridInterpolator(grid, np.tile(np.cos(np.deg2rad(lon_rolled))[None, :], (len(lat), 1)), bounds_error=False, fill_value=np.nan)(centreline)
    plon = np.rad2deg(np.arctan2(sin, cos)); plon = np.mod(plon, 360) if lon.min() >= 0 else (plon + 180) % 360 - 180
    valid = np.isfinite(plat) & np.isfinite(plon); plat, plon = plat[valid], plon[valid]
    return float(sum(geodesic((float(plat[i]), float((plon[i] + 180) % 360 - 180)), (float(plat[i + 1]), float((plon[i + 1] + 180) % 360 - 180))).km for i in range(len(plat) - 1)))


def dark_mask(exceed: np.ndarray, area: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    settings = cfg["dark"]
    merra_legacy = settings.get("compatibility_profile") == "merra2_legacy"
    periodic = periodic_longitude(cfg["data"], lon)
    clean = _filter_area(exceed, area, float(settings["min_area_km2"]), periodic)
    labels, n = label_periodic(clean, periodic)
    out = np.zeros_like(clean, dtype=np.int8)
    for component in range(1, n + 1):
        component_mask = labels == component
        if component_mask.sum() < 30: continue
        rolled, shift = _roll_by_largest_gap(component_mask) if periodic else (component_mask, 0); lon_rolled = np.roll(lon, shift)
        # MERRA's historical script applied the inexpensive bbox gate before
        # moving the seam; ERA5 sensitivity used the dateline-safe variant.
        ys, xs = np.where(component_mask if merra_legacy else rolled)
        proxy_lon = lon if merra_legacy else lon_rolled
        proxy = 2 * 6371.0088 * np.arcsin(np.sqrt(np.sin(np.deg2rad(lat[ys.max()] - lat[ys.min()]) / 2) ** 2 + np.cos(np.deg2rad(lat[ys.min()])) * np.cos(np.deg2rad(lat[ys.max()])) * np.sin(np.deg2rad(proxy_lon[xs.max()] - proxy_lon[xs.min()]) / 2) ** 2))
        if proxy >= 0.5 * float(settings["min_length_km"]) and _legacy_centerline_length(component_mask, lat, lon, periodic) >= float(settings["min_length_km"]): out[component_mask] = 1
    return out


def _canonical_doy(time: xr.DataArray) -> xr.DataArray:
    # A 366-day climatology: non-leap years skip Feb 29.
    doy = time.dt.dayofyear
    leap = time.dt.is_leap_year
    return xr.where((~leap) & (doy >= 60), doy + 1, doy)


def _reference_years(cfg: dict[str, Any]) -> range:
    start, end = cfg["percentile"].get("reference_years", cfg["years"])
    return range(int(start), int(end) + 1)


def _adaptive_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["output_dir"]) / "adaptive_threshold_factors.nc"


def _open_adaptive_driver(year: int, cfg: dict[str, Any]) -> xr.DataArray:
    adaptive = cfg["adaptive_threshold"]
    source = dict(cfg["data"]); source.update(adaptive["data"])
    ds = _normalise(xr.open_dataset(source["file_pattern"].format(year=year, member=source.get("member", ""))), source)
    region_cfg = dict(source)
    if "region" in adaptive: region_cfg["region"] = adaptive["region"]
    return _region(ds, region_cfg)[source["variable"]]


def build_adaptive_factors(cfg: dict[str, Any]) -> Path | None:
    """Build annual threshold multipliers following the legacy adaptive method."""
    adaptive = cfg.get("adaptive_threshold", {})
    if not adaptive.get("enabled"): return None
    path = _adaptive_path(cfg)
    if path.exists(): return path
    analysis = range(int(cfg["years"][0]), int(cfg["years"][1]) + 1)
    reference = _reference_years(cfg)
    years = range(min(analysis.start, reference.start), max(analysis.stop - 1, reference.stop - 1) + 1)
    means = np.asarray([float(_open_adaptive_driver(year, cfg).mean(skipna=True).values) for year in years])
    sigma = float(adaptive.get("smoothing_sigma_years", 2))
    smoothed = means if sigma == 0 else ndimage.gaussian_filter1d(means, sigma=sigma)
    reference_mask = np.isin(np.asarray(list(years)), list(reference))
    baseline = float(np.mean(smoothed[reference_mask]))
    if not np.isfinite(baseline) or baseline == 0: raise ValueError("Adaptive-threshold reference mean is invalid.")
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"annual_mean": ("year", means), "smoothed_annual_mean": ("year", smoothed), "adaptive_factor": ("year", smoothed / baseline)},
        coords={"year": list(years)},
        attrs={"description": "Annual threshold multiplier; smoothed driver mean divided by its reference-period mean."},
    ).to_netcdf(path)
    return path


def adaptive_factor(cfg: dict[str, Any], year: int) -> float:
    if not cfg.get("adaptive_threshold", {}).get("enabled"): return 1.0
    path = build_adaptive_factors(cfg)
    with xr.open_dataset(path) as ds:
        return float(ds["adaptive_factor"].sel(year=year).values)


def build_percentiles(cfg: dict[str, Any]) -> Path:
    if cfg.get("percentile_file"):
        path = Path(cfg["percentile_file"])
        if not path.exists(): raise FileNotFoundError(f"Percentile file is missing: {path}")
        build_adaptive_factors(cfg)
        return path
    years = _reference_years(cfg)
    window = int(cfg["percentile"].get("window_days", 15))
    if window % 2 == 0: raise ValueError("percentile.window_days must be odd")
    all_ivt = xr.concat([_open_ivt(y, cfg) for y in years], "time")
    doy = _canonical_doy(all_ivt.time)
    parts = []
    for day in range(1, 367):
        wanted = ((np.arange(day - window // 2, day + window // 2 + 1) - 1) % 366) + 1
        sample = all_ivt.where(doy.isin(wanted), drop=True)
        # Keeping all 366 DOYs is important for a standard climatology even
        # when a short test/model experiment does not cover every calendar day.
        value = sample.quantile(cfg["percentile"]["quantile"], dim="time") if sample.sizes["time"] else xr.full_like(all_ivt.isel(time=0, drop=True), np.nan)
        parts.append(value.assign_coords(dayofyear=day).expand_dims("dayofyear"))
    out = Path(cfg["output_dir"]) / "percentiles.nc"; out.parent.mkdir(parents=True, exist_ok=True)
    xr.concat(parts, "dayofyear").rename("ivt_percentile").to_netcdf(out, encoding={"ivt_percentile": {"zlib": True, "complevel": 1}})
    build_adaptive_factors(cfg)
    return out


@dataclass
class ChildrenTracker:
    land: np.ndarray
    periodic_longitude: bool = True
    previous: np.ndarray | None = None
    tracks: dict[int, tuple[bool, bool]] = None  # id -> (ever_parent, born_ok)
    label_tracks: dict[int, int] = None
    next_id: int = 1
    def __post_init__(self): self.tracks, self.label_tracks = {}, {}
    def step(self, exceed: np.ndarray, parent: np.ndarray) -> np.ndarray:
        labels, n = label_periodic(exceed, self.periodic_longitude)
        current: dict[int, int] = {}
        for label in range(1, n + 1):
            overlaps = self.previous is not None and np.any((labels == label) & (self.previous > 0))
            prior = 0
            if overlaps:
                candidates = self.previous[(labels == label) & (self.previous > 0)]
                prior = int(np.bincount(candidates).argmax())
            if prior in self.label_tracks: current[label] = self.label_tracks[prior]
            else: current[label] = self.next_id; self.tracks[self.next_id] = (False, True); self.next_id += 1
        children = np.zeros_like(parent, dtype=np.int8)
        for label, track in current.items():
            component = labels == label; had_parent, born_ok = self.tracks[track]
            now_parent = bool(np.any(component & parent))
            self.tracks[track] = (had_parent or now_parent, born_ok)
            if had_parent and np.any(component & self.land) and not now_parent: children[component] = 1
        self.previous, self.label_tracks = labels, current
        return children


def run(cfg: dict[str, Any]) -> list[Path]:
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    pctl = open_percentile(cfg.get("percentile_file") or build_percentiles(cfg))
    build_adaptive_factors(cfg)
    children_cfg = cfg.get("children", {"enabled": False})
    tracker = None
    outputs = []
    if children_cfg.get("enabled"):
        mask = _region(_normalise(xr.open_dataset(children_cfg["land_mask_file"]), cfg["data"], require_time=False), cfg["data"])[children_cfg["land_mask_variable"]]
    for year in range(int(cfg["years"][0]), int(cfg["years"][1]) + 1):
        ivt = _open_ivt(year, cfg); threshold = pctl.sel(dayofyear=_canonical_doy(ivt.time)) * adaptive_factor(cfg, year)
        area = _area(ivt.lat.values, ivt.lon.values, cfg["dark"].get("compatibility_profile") in {"merra2_legacy", "era5_sensitivity"})
        if children_cfg.get("enabled") and tracker is None:
            tracker = ChildrenTracker(mask.sel(lat=ivt.lat, lon=ivt.lon).values > float(children_cfg.get("land_mask_threshold", 0)), periodic_longitude(cfg["data"], ivt.lon.values))
        parents = np.zeros(ivt.shape, dtype=np.int8); kids = np.zeros(ivt.shape, dtype=np.int8)
        for t in range(ivt.sizes["time"]):
            exceed = (ivt.isel(time=t).values > threshold.isel(time=t).values)
            parents[t] = dark_mask(exceed, area, ivt.lat.values, ivt.lon.values, cfg)
            if tracker: kids[t] = tracker.step(_filter_area(exceed, area, float(cfg["dark"]["min_area_km2"]), periodic_longitude(cfg["data"], ivt.lon.values)), parents[t].astype(bool))
        data = {"ar_binary_tag": (("time", "lat", "lon"), ((parents | kids) if tracker else parents).astype("i1"))}
        if tracker: data["ar_type"] = (("time", "lat", "lon"), (parents + 2 * kids).astype("i1"))
        ds = xr.Dataset(data, coords={"time": ivt.time, "lat": ivt.lat, "lon": ivt.lon}, attrs={"scheme": "DARK", "percentile": cfg["percentile"]["quantile"]})
        path = out_dir / f"{cfg['dataset']}.DARK.{year}.nc"; ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 1} for v in data}); outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DARK (and optionally children) from one portable config.")
    parser.add_argument("config", help="JSON or YAML configuration")
    parser.add_argument("--build-percentiles-only", action="store_true")
    parser.add_argument("--validate-config", action="store_true", help="Check configuration only; do not read data or write files.")
    args = parser.parse_args(); cfg = _read_config(args.config)
    validate_config(cfg)
    if args.validate_config:
        print("Configuration is valid."); return
    paths = [build_percentiles(cfg)] if args.build_percentiles_only else run(cfg)
    print("\n".join(map(str, paths)))

if __name__ == "__main__": main()
