#!/usr/bin/env python3
"""Scheduler-neutral annual DARK/children pipeline.

Schedulers only arrange dependencies. This module owns the portable on-disk
contract between the parallel DARK stage and children stages that wait only
for their current and previous annual parent artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

import dark


def _paths(cfg, year: int) -> tuple[Path, Path]:
    root = Path(cfg["output_dir"])
    work = root / ".dark-work"
    return work / f"{year}.nc", root / f"{cfg['dataset']}.DARK.{year}.nc"


def _percentiles(cfg) -> xr.DataArray:
    path = Path(cfg.get("percentile_file") or Path(cfg["output_dir"]) / "percentiles.nc")
    if not path.exists():
        raise FileNotFoundError(f"Percentile file is missing: {path}. Run 'build-percentiles' once before dark-year jobs.")
    return dark.open_percentile(path)


def dark_year(cfg: dict, year: int) -> Path:
    """Run one annual DARK stage.

    With children this publishes the disposable parent/exceedance artifact.
    Without children it publishes the final catalogue directly.
    """
    intermediate, output = _paths(cfg, year)
    children_enabled = bool(cfg.get("children", {}).get("enabled"))
    target = intermediate if children_enabled else output
    target.parent.mkdir(parents=True, exist_ok=True)
    ivt = dark._open_ivt(year, cfg)
    threshold = _percentiles(cfg).sel(dayofyear=dark._canonical_doy(ivt.time)) * dark.adaptive_factor(cfg, year)
    area = dark._area(ivt.lat.values, ivt.lon.values, cfg["dark"].get("compatibility_profile") in {"merra2_legacy", "era5_sensitivity"})
    parents = np.zeros(ivt.shape, dtype=np.int8)
    exceedances = np.zeros(ivt.shape, dtype=np.int8)
    minimum = float(cfg["dark"]["min_area_km2"])
    for index in range(ivt.sizes["time"]):
        exceed = ivt.isel(time=index).values > threshold.isel(time=index).values
        exceedances[index] = dark._filter_area(exceed, area, minimum, dark.periodic_longitude(cfg["data"], ivt.lon.values))
        parents[index] = dark.dark_mask(exceed, area, ivt.lat.values, ivt.lon.values, cfg)
    if children_enabled:
        ds = xr.Dataset(
            {"dark_parent": (("time", "lat", "lon"), parents),
             "exceedance": (("time", "lat", "lon"), exceedances)},
            coords={"time": ivt.time, "lat": ivt.lat, "lon": ivt.lon},
            attrs={"temporary": "Input to children-year; delete only after a successful children step."},
        )
        encoding = {"dark_parent": {"zlib": True, "complevel": 1}, "exceedance": {"zlib": True, "complevel": 1}}
    else:
        ds = xr.Dataset(
            {"ar_binary_tag": (("time", "lat", "lon"), parents)},
            coords={"time": ivt.time, "lat": ivt.lat, "lon": ivt.lon},
            attrs={"scheme": "DARK", "percentile": cfg["percentile"]["quantile"]},
        )
        encoding = {"ar_binary_tag": {"zlib": True, "complevel": 1}}
    ds.to_netcdf(target, encoding=encoding)
    return target


def _warm_tracker(previous_parent: Path, land: np.ndarray, periodic_longitude: bool) -> dark.ChildrenTracker:
    """Rebuild boundary tracking from the preceding parent/exceedance file."""
    if not previous_parent.exists():
        raise FileNotFoundError(f"Previous DARK artifact is missing: {previous_parent}")
    tracker = dark.ChildrenTracker(land, periodic_longitude)
    with xr.open_dataset(previous_parent) as prior:
        for index in range(prior.sizes["time"]):
            tracker.step(prior.exceedance.isel(time=index).values.astype(bool), prior.dark_parent.isel(time=index).values.astype(bool))
    return tracker


def children_year(cfg: dict, year: int, previous_parent: Path | None) -> Path:
    """Classify one year after optionally replaying the preceding parent file."""
    intermediate, output = _paths(cfg, year)
    if not intermediate.exists(): raise FileNotFoundError(f"DARK artifact is missing: {intermediate}")
    child_cfg = cfg.get("children", {})
    if not child_cfg.get("enabled"): raise ValueError("children.enabled must be true for children-year")
    artifact = xr.open_dataset(intermediate)
    land_ds = dark._region(dark._normalise(xr.open_dataset(child_cfg["land_mask_file"]), cfg["data"], require_time=False), cfg["data"])
    land = land_ds[child_cfg["land_mask_variable"]].sel(lat=artifact.lat, lon=artifact.lon).values > float(child_cfg.get("land_mask_threshold", 0))
    periodic = dark.periodic_longitude(cfg["data"], artifact.lon.values)
    tracker = _warm_tracker(previous_parent, land, periodic) if previous_parent else dark.ChildrenTracker(land, periodic)
    parent = artifact.dark_parent.values.astype(bool); kids = np.zeros_like(parent, dtype=np.int8)
    for index in range(artifact.sizes["time"]): kids[index] = tracker.step(artifact.exceedance.isel(time=index).values.astype(bool), parent[index])
    output.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"ar_binary_tag": (("time", "lat", "lon"), (parent | kids).astype("i1")),
         "ar_type": (("time", "lat", "lon"), (parent + 2 * kids).astype("i1"))},
        coords={"time": artifact.time, "lat": artifact.lat, "lon": artifact.lon},
        attrs={"scheme": "DARK", "children_definition": "0=none, 1=parent, 2=child"},
    ).to_netcdf(output, encoding={"ar_binary_tag": {"zlib": True, "complevel": 1}, "ar_type": {"zlib": True, "complevel": 1}})
    artifact.close()
    land_ds.close()
    # Artifacts must remain available while the next year may still be
    # warming from them. They can be removed after the complete workflow.
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable annual DARK/children pipeline stages.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dark-year", "children-year"):
        cmd = sub.add_parser(name); cmd.add_argument("config"); cmd.add_argument("year", type=int)
        if name == "children-year":
            cmd.add_argument("--previous-parent", type=Path)
    build = sub.add_parser("build-percentiles"); build.add_argument("config")
    args = parser.parse_args(); cfg = dark._read_config(args.config); dark.validate_config(cfg)
    if args.command == "build-percentiles": print(dark.build_percentiles(cfg)); return
    if args.command == "dark-year": print(dark_year(cfg, args.year))
    else:
        print(children_year(cfg, args.year, args.previous_parent))


if __name__ == "__main__": main()
