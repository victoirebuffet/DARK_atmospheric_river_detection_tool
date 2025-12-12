#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 09:37:07 2025

@author: vb234125
"""
import os, argparse
import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, splprep, splev
from skimage.morphology import skeletonize
from geopy.distance import geodesic
import networkx as nx
import matplotlib.pyplot as plt

# =========================
# Default config
# =========================
ALGO_SCHEME      = "Buffet"    # ar_binary_tag:scheme
ALGO_VERSION     = "1.0"       # ar_binary_tag:version
IVT_QUANTILE     = 0.98        # used by percentile builder
ROLLING_WINDOW_D = 15          # rolling window size (must be odd)
MIN_AREA_KM2     = 20000       # drop patches smaller than this (km^2)
MIN_LENGTH_KM    = 2000        # min AR length (km)
START_YEAR       = 1980        # Start year (inclusive)
END_YEAR         = 2017        # End year (inclusive)

YEARS = range(START_YEAR, END_YEAR+1)

# ---- Files & directories (edit to your machine) ----
ERA5_U_PATTERN   = "/work/crct/vb234125/tilted_ARs/IVT/vertical_integral_of_eastward_water_vapour_flux_{year}_reanaHS.nc"
ERA5_V_PATTERN   = "/work/crct/vb234125/tilted_ARs/IVT/vertical_integral_of_northward_water_vapour_flux_{year}_reanaHS.nc"
PCTL_DOY_PATH    = "/work/crct/vb234125/tilted_ARs/IVT/percentile_98_IVT_rolling_15D_era5.nc"
OUT_DIR_DEFAULT  = "/work/crct/vb234125/tilted_ARs/IVT"

# ---- ERA5 variable names (override via CLI if different) ----
ERA5_U_NAME      = "viwve"
ERA5_V_NAME      = "viwvn"

# ---- Longitude domain for BOTH build & tag: "keep" (0..360) or "-180" (-180..180) ----
ERA5_LON_DOMAIN  = "keep"

# =========================
# Geo helpers
# =========================
def earth_radius(lat_deg):
    """WGS84 geocentric radius (m) for geodetic latitude (deg)."""
    a = 6378137.0
    b = 6356752.3142
    e2 = 1.0 - (b**2 / a**2)
    lat = np.deg2rad(lat_deg)
    lat_gc = np.arctan((1.0 - e2) * np.tan(lat))
    return (a * np.sqrt(1.0 - e2)) / np.sqrt(1.0 - e2 * np.cos(lat_gc)**2)

def gridcell_area(lat_1d, lon_1d):
    """
    Area per grid cell on a regular lat/lon grid [m^2].
    Works for any (monotonic) latitude ordering and any uniform lon spacing.
    """
    lat = np.asarray(lat_1d)
    lon = np.asarray(lon_1d)

    if lat.size < 2 or lon.size < 2:
        raise ValueError("Need at least 2 points in both lat and lon to compute cell areas.")

    # latitude spacing (respect actual spacing and ordering)
    dlat = np.empty_like(lat, dtype=float)
    dlat[1:-1] = (lat[2:] - lat[:-2]) / 2.0
    dlat[0]    =  lat[1]  - lat[0]
    dlat[-1]   =  lat[-1] - lat[-2]
    dlat_rad = np.deg2rad(dlat)[:, None]  # (lat, 1)

    # longitude spacing (assumed uniform)
    dlon_step = np.diff(lon).mean()
    dlon_rad  = np.deg2rad(dlon_step)     # scalar (broadcasts across lon)

    # 2D latitude field for radius/cos(lat)
    lat2d = np.broadcast_to(lat[:, None], (lat.size, lon.size))
    R = earth_radius(lat2d)

    dy = dlat_rad * R
    dx = dlon_rad  * R * np.cos(np.deg2rad(lat2d))

    return np.abs(dy * dx)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

# =========================
# Dateline-aware labeling (toroidal in lon)
# =========================

def _timestep_minutes_strict(time_values):
    """
    Return the timestep in minutes from the first two entries.
    Raises if timesteps are not uniform.
    """
    t = np.asarray(time_values)
    if t.size < 2:
        return None  # or raise ValueError("Not enough timesteps to infer resolution.")
    # minutes between first two steps
    dt0 = (t[1] - t[0]).astype("timedelta64[m]").astype(int)
    # sanity check: all steps equal to dt0?
    dt_all = np.diff(t).astype("timedelta64[m]").astype(int)
    if not np.all(dt_all == dt0):
        raise ValueError(f"Non-uniform timestep detected. First step={dt0} min, uniques={np.unique(dt_all)}")
    return int(dt0)


def _yyyymmdd(t64):
    # e.g. 19800101
    return np.datetime_as_string(t64, unit="D").replace("-", "")

def _freq_label_from_time(ds):
    """
    Infer frequency label from first two timesteps. Assumes constant step.
    Returns one of "hourly", "3hourly", "6hourly", "daily", or "<N>hourly".
    """
    t = ds["time"].values
    if t.size < 2:
        return "unknown"
    dt_h = float((t[1] - t[0]) / np.timedelta64(1, "h"))
    m = {1.0: "hourly", 3.0: "3hourly", 6.0: "6hourly", 24.0: "daily"}
    return m.get(round(dt_h), f"{dt_h:g}hourly")


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n+1); self.r = np.zeros(n+1, dtype=int)
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]: self.p[ra] = rb
        elif self.r[ra] > self.r[rb]: self.p[rb] = ra
        else: self.p[rb] = ra; self.r[ra] += 1

def align_pctl_to_ivt_grid(pctl_da, ivt_da):
    """
    Force percentile (dayofyear, lat, lon) onto IVT grid (exact coords/order).
    """
    p = pctl_da
    if "latitude" in p.dims or "longitude" in p.dims:
        p = p.rename({"latitude": "lat", "longitude": "lon"})
    if "dayofyear" not in p.dims:
        raise ValueError("Percentile file is missing 'dayofyear' dimension.")
    # exact match, preserve IVT order
    p = p.sel(lat=ivt_da["lat"], lon=ivt_da["lon"], drop=True)
    return p

def label_torus_lon(binary2d_bool):
    """8-connected labeling with periodic connectivity in longitude."""
    lab, nlab = ndimage.label(binary2d_bool, structure=np.ones((3,3), int))
    if nlab == 0:
        return lab, 0
    uf = UnionFind(nlab)
    left, right = lab[:, 0], lab[:, -1]
    H = lab.shape[0]
    for i in range(H):
        li = left[i]
        if li == 0: continue
        for di in (-1,0,1):
            j = i + di
            if 0 <= j < H:
                rj = right[j]
                if rj > 0: uf.union(li, rj)
    roots = np.zeros(nlab+1, dtype=int)
    for k in range(1, nlab+1):
        roots[k] = uf.find(k)
    uniq = np.unique(roots[roots > 0])
    new_id = {r:i+1 for i, r in enumerate(uniq)}
    # remap labels to {1..nuniq}
    maparr = np.zeros(nlab+1, dtype=int)
    for k in range(1, nlab+1):
        maparr[k] = new_id[roots[k]]
    out = np.zeros_like(lab, dtype=int)
    nz = lab > 0
    out[nz] = maparr[lab[nz]]
    return out, len(uniq)

def roll_by_largest_gap(mask2d):
    """Roll columns so largest empty gap in used columns becomes the seam."""
    nlon = mask2d.shape[1]
    cols = np.where(mask2d.any(axis=0))[0]
    if cols.size == 0: return mask2d, 0
    cols = np.unique(np.sort(cols))
    diffs = np.diff(np.r_[cols, cols[0] + nlon])
    gaps = diffs - 1
    k = int(np.argmax(gaps))
    gap_len = int(gaps[k])
    if gap_len <= 0:
        shift = 0
    else:
        start = cols[k]
        seam = (start + 1 + gap_len // 2) % nlon
        shift = -seam
    return np.roll(mask2d, shift=shift, axis=1), shift

# ===== Skeleton → graph → PCA → smooth centerline =====
def skeleton_to_graph(skel):
    G = nx.Graph()
    coords = np.argwhere(skel)
    for y, x in coords:
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy == 0 and dx == 0: continue
                ny, nx_ = y+dy, x+dx
                if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                    if skel[ny, nx_]: G.add_edge((y,x), (ny,nx_))
    return G

def extract_candidate_paths(G):
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    paths = []
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            try:
                paths.append(nx.shortest_path(G, endpoints[i], endpoints[j]))
            except nx.NetworkXNoPath:
                pass
    return paths

def compute_pca_direction(mask):
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    pts0 = pts - pts.mean(0)
    _, _, vh = np.linalg.svd(pts0, full_matrices=False)
    return vh[0]

def path_score(path, pca_axis, mask_shape):
    coords = np.array(path)
    if len(coords) < 2: return -np.inf
    length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    unit_vec = coords[-1] - coords[0]
    unit = unit_vec / (np.linalg.norm(unit_vec) + 1e-9)
    directionality = abs(np.dot(unit, pca_axis))
    lat_extent = np.ptp(coords[:,0]) / mask_shape[0]
    return length * directionality * lat_extent

def fallback_centerline(mask):
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    if ys.size == 0 or xs.size == 0: return None
    lats = np.arange(ys.min(), ys.max()+1)
    xmid = int(np.round(xs.mean()))
    return np.column_stack([lats, np.full_like(lats, xmid)])

def extend_path_to_edge(path, mask, max_extension=25):
    if path is None or len(path) < 2: return path
    def extend_one_end(p0, p1):
        v = (p0 - p1).astype(float)
        v /= (np.linalg.norm(v) + 1e-9)
        for i in range(1, max_extension+1):
            yy, xx = np.round(p0 + i*v).astype(int)
            if not (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]): break
            if mask[yy, xx]: continue
            return np.round(p0 + (i-1)*v).astype(int)
        return p0
    P = np.asarray(path)
    a = extend_one_end(P[0], P[1])
    b = extend_one_end(P[-1], P[-2])
    return np.vstack([a, P[1:-1], b])

def smooth_single_skeleton(skeleton, n_points=200, spline_s=300):
    """B-spline smoothing of [y,x] path; returns (n_points, 2)."""
    if skeleton is None or len(skeleton) < 3:
        return skeleton
    y, x = skeleton[:,0].astype(float), skeleton[:,1].astype(float)
    try:
        tck, _ = splprep([y, x], s=spline_s)
        u = np.linspace(0, 1, n_points)
        ys, xs = splev(u, tck)
        return np.column_stack([ys, xs])
    except Exception:
        return skeleton

def centerline_length_smooth(mask2d, lat_vals, lon_vals, n_points=200, spline_s=300):
    """
    Dateline-safe centerline with smoothing and subgrid geodesic length.
    Returns: length_km, (lon_path, lat_path)
    """
    rolled, shift = roll_by_largest_gap(mask2d)
    lon_rolled = np.roll(lon_vals, shift)

    skel = skeletonize(rolled)
    G = skeleton_to_graph(skel)
    paths = extract_candidate_paths(G)
    if paths:
        pca_axis = compute_pca_direction(rolled)
        best = max(paths, key=lambda p: path_score(p, pca_axis, rolled.shape))
        cl = np.array(best)
    else:
        cl = fallback_centerline(rolled)
        if cl is None:
            return 0.0, (None, None)

    cl = extend_path_to_edge(cl, rolled, max_extension=25)
    cl = smooth_single_skeleton(cl, n_points=n_points, spline_s=spline_s)

    H, W = len(lat_vals), len(lon_rolled)
    cl[:, 0] = np.clip(cl[:, 0], 0, H - 1)     # y/lat index
    cl[:, 1] = np.mod(cl[:, 1], W)            # x/lon index (wrap)

    iy = np.arange(H); ix = np.arange(W)
    lat_sfc = np.tile(lat_vals[:, None], (1, W))
    lat_ip = RegularGridInterpolator((iy, ix), lat_sfc, bounds_error=False, fill_value=np.nan)
    plat = lat_ip(cl)  # keep latitude linear

    # circular interpolation for longitude
    lon_rad = np.deg2rad(lon_rolled)
    sin2d = np.tile(np.sin(lon_rad)[None, :], (H, 1))
    cos2d = np.tile(np.cos(lon_rad)[None, :], (H, 1))
    
    sin_ip = RegularGridInterpolator((iy, ix), sin2d, bounds_error=False, fill_value=np.nan)
    cos_ip = RegularGridInterpolator((iy, ix), cos2d, bounds_error=False, fill_value=np.nan)
    
    psin = sin_ip(cl)
    pcos = cos_ip(cl)
    plon = np.rad2deg(np.arctan2(psin, pcos))          # in (-180, 180]
    
    # map back to dataset domain for outputs, if needed
    if lon_vals.min() >= 0:                     # 0..360 grid
        plon = (plon + 360.0) % 360.0

    valid = np.isfinite(plat) & np.isfinite(plon)
    plat = np.clip(plat[valid], -90.0, 90.0)
    plon = plon[valid]

    # helper: normalize to [-180, 180]
    def to180(x):
        return ((x + 180.0) % 360.0) - 180.0

    # wrap-aware geodesic length
    L = 0.0
    for i in range(len(plat) - 1):
        lat1, lon1 = float(plat[i]),  float(plon[i])
        lat2, lon2 = float(plat[i+1]), float(plon[i+1])

        # make lon2 the nearest representation to lon1
        d = lon2 - lon1
        if d > 180.0:
            lon2 -= 360.0
        elif d < -180.0:
            lon2 += 360.0

        # normalize both to [-180, 180] for geodesic
        lon1_180 = to180(lon1)
        lon2_180 = to180(lon2)

        try:
            L += geodesic((lat1, lon1_180), (lat2, lon2_180)).kilometers
        except ValueError:
            L += haversine_km(lat1, lon1_180, lat2, lon2_180)

    return float(L), (plon, plat)

# =========================
# Area filtering
# =========================
def filter_by_area_torus(binary3d, area_m2, min_area_km2=MIN_AREA_KM2):
    """
    binary3d: (time, lat, lon) int {0,1}
    area_m2: (lat, lon) float
    Returns: (time, lat, lon) int8 after removing components smaller than min_area_km2.
    """
    T, _, _ = binary3d.shape
    out = np.zeros_like(binary3d, dtype=np.int8)
    area_flat = area_m2.ravel()
    for t in range(T):
        lab, nlab = label_torus_lon(binary3d[t].astype(bool))
        if nlab == 0:
            continue
        sums = np.bincount(lab.ravel(), weights=area_flat, minlength=nlab+1)  # m^2
        keep_ids = np.where((sums/1e6) >= min_area_km2)[0]
        keep_ids = keep_ids[keep_ids > 0]
        if keep_ids.size:
            out[t] = np.isin(lab, keep_ids).astype(np.int8)
    return out

def area_filter_2d(mask2d_bool, area_m2, min_area_km2=MIN_AREA_KM2):
    """Single-timestep area filter (toroidal)."""
    lab, nlab = label_torus_lon(mask2d_bool)
    if nlab == 0:
        return np.zeros_like(mask2d_bool, dtype=np.int8)
    sums = np.bincount(lab.ravel(), weights=area_m2.ravel(), minlength=nlab+1)
    keep_ids = np.where((sums/1e6) >= min_area_km2)[0]
    keep_ids = keep_ids[keep_ids > 0]
    return np.isin(lab, keep_ids).astype(np.int8) if keep_ids.size else np.zeros_like(mask2d_bool, dtype=np.int8)

# =========================
# Rolling DOY 98th (ERA5: IVT from u/v)
# =========================
def _pick_first_var(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(f"None of the variables {candidates} found in {list(ds.data_vars)}")

def _normalize_era5_coords(ds, to_180=False):
    """
    Rename ERA5 coords to lat/lon, ensure 'time' exists, handle expver, optionally convert lon to -180..180.
    """
    if "latitude" in ds.coords:  ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})
    if "valid_time" in ds.coords: ds = ds.rename({"valid_time": "time"})

    if "expver" in ds.dims:
        try:
            ds = ds.sel(expver=ds["expver"].max())
        except Exception:
            ds = ds.isel(expver=-1)
        ds = ds.drop_vars("expver", errors="ignore")
        ds = ds.reset_coords(names=["expver"], drop=True)

    if to_180 and float(ds["lon"].min()) >= 0.0:
        lon_new = ((ds["lon"] + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(lon=lon_new).sortby("lon")

    return ds

def _open_era5_ivt_year(
    year,
    u_pattern, v_pattern,
    u_names=("viwve", "vertical_integral_of_eastward_water_vapour_flux", "uIVT"),
    v_names=("viwvn", "vertical_integral_of_northward_water_vapour_flux", "vIVT"),
    to_180=False,
):
    """
    Open ERA5 yearly uIVT & vIVT, compute IVT magnitude lazily, return Dataset with IVT(time,lat,lon).
    """
    uf = u_pattern.format(year=year)
    vf = v_pattern.format(year=year)
    if not os.path.exists(uf): raise FileNotFoundError(uf)
    if not os.path.exists(vf): raise FileNotFoundError(vf)

    du = xr.open_dataset(uf)
    dv = xr.open_dataset(vf)

    du = _normalize_era5_coords(du, to_180=to_180)
    dv = _normalize_era5_coords(dv, to_180=to_180)

    du, dv = xr.align(du, dv, join="inner")

    u = _pick_first_var(du, u_names)
    v = _pick_first_var(dv, v_names)

    ivt = xr.apply_ufunc(
        np.hypot, u, v,
        dask="allowed", output_dtypes=[u.dtype]
    ).rename("IVT")

    out = xr.Dataset({"IVT": ivt})
    for dim in ("time", "lat", "lon"):
        if dim not in out.dims and dim in out.coords:
            out = out.swap_dims({dim: dim})
    return out

def compute_rolling_98th_for_day_era5(dayofyear, years, window_size,
                                      u_pattern, v_pattern, to_180=False,
                                      u_names=("viwve","vertical_integral_of_eastward_water_vapour_flux","uIVT"),
                                      v_names=("viwvn","vertical_integral_of_northward_water_vapour_flux","vIVT")):
    """
    Same semantics as MERRA2, but IVT is computed from ERA5 u/v components.
    The lon domain (0–360 vs −180–180) is controlled by to_180 and MUST match when tagging.
    """
    if window_size % 2 == 0:
        raise ValueError("ROLLING_WINDOW_D must be odd (e.g., 15).")
    half = window_size // 2
    W = ((np.arange(dayofyear - half, dayofyear + half + 1) - 1) % 366) + 1  # leap-based 1..366

    parts = []
    for y in years:
        ds = _open_era5_ivt_year(y, u_pattern, v_pattern, u_names=u_names, v_names=v_names, to_180=to_180)
        doy = ds["time"].dt.dayofyear
        is_leap = (y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0))
        ldoy = doy if is_leap else xr.where(doy >= 60, doy + 1, doy)
        sel = ds["IVT"].sel(time=ldoy.isin(W))
        if sel.sizes.get("time", 0) > 0:
            parts.append(sel.load())  # materialize window before closing ds
        ds.close()

    if not parts:
        raise ValueError(f"No ERA5 IVT data for DOY={dayofyear}")

    big = xr.concat(parts, dim="time")
    return big.quantile(IVT_QUANTILE, dim="time").rename("ivt_pctl")

def build_all_days_and_write_era5(out_path, years, window_size, u_pattern, v_pattern,
                                  to_180=False, u_names=("viwve","vertical_integral_of_eastward_water_vapour_flux","uIVT"),
                                  v_names=("viwvn","vertical_integral_of_northward_water_vapour_flux","vIVT")):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    das = []
    for doy in range(1, 367):
        print(f"[ERA5 DOY {doy:03d}] computing...")
        da = compute_rolling_98th_for_day_era5(
            doy, years, window_size, u_pattern, v_pattern, to_180=to_180,
            u_names=u_names, v_names=v_names
        )
        das.append(da.assign_coords(dayofyear=doy).expand_dims("dayofyear"))
    pctl = xr.concat(das, dim="dayofyear")
    enc = {"ivt_pctl": {"zlib": True, "complevel": 1, "shuffle": True,
                        "chunksizes": (8, pctl.sizes["lat"], pctl.sizes["lon"])}}
    pctl.to_netcdf(out_path, format="NETCDF4", engine="netcdf4", encoding=enc)
    print("Wrote:", out_path)
    return out_path

# =========================
# Yearly tagging
# =========================
def _is_leap(y): return (y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0))

def write_artmip_year_era5(year, u_pattern, v_pattern, pctl_path, out_dir,
                           to_180=False,
                           algo_scheme=ALGO_SCHEME, algo_version=ALGO_VERSION,
                           min_area_km2=MIN_AREA_KM2, min_length_km=MIN_LENGTH_KM,
                           u_names=("viwve","vertical_integral_of_eastward_water_vapour_flux","uIVT"),
                           v_names=("viwvn","vertical_integral_of_northward_water_vapour_flux","vIVT")):
    """
    Compute IVT from ERA5 u/v for the year, threshold vs ERA5 pctl file (same lon domain),
    filter by area & length, and write ARTMIP NetCDF (ERA5.ar_tag...).
    """
    os.makedirs(out_dir, exist_ok=True)

    ds = _open_era5_ivt_year(year, u_pattern, v_pattern, u_names=u_names, v_names=v_names, to_180=to_180)
    lat = ds["lat"].values
    lon = ds["lon"].values
    area_m2 = gridcell_area(lat, lon)
    freq_label = _freq_label_from_time(ds)

    start = f"{year}0101"; end = f"{year}1231"
    
    if not os.path.exists(pctl_path):
        raise FileNotFoundError(f"Rolling DOY percentile not found: {pctl_path}")
    pctl = xr.open_dataset(pctl_path)["ivt_pctl"]
    if "latitude" in pctl.dims: pctl = pctl.rename({"latitude":"lat", "longitude":"lon"})
    if to_180 and float(pctl["lon"].min()) >= 0.0:
        pctl = pctl.assign_coords(lon=((pctl["lon"] + 180.0) % 360.0) - 180.0).sortby("lon")

    # align pctl to the IVT grid (exact coords/order)
    pctl = align_pctl_to_ivt_grid(pctl, ds["IVT"])

    # map timesteps to leap-based DOY
    doy = ds["time"].dt.dayofyear
    is_leap = (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))
    if not is_leap and (doy >= 60).any():
        doy = xr.where(doy >= 60, doy + 1, doy)

    ivt_thresh = pctl.sel(dayofyear=doy)
    exceed = (ds["IVT"] > ivt_thresh).astype(np.int8).values

    clean = filter_by_area_torus(exceed, area_m2, min_area_km2=min_area_km2)

    keep = np.zeros_like(clean, dtype=np.int8)
    for t in range(clean.shape[0]):
        lab, nlab = label_torus_lon(clean[t].astype(bool))
        for lab_id in range(1, nlab+1):
            comp = (lab == lab_id)
            if comp.sum() < 30:
                continue

            # Seam-free bbox proxy
            comp_rolled, shift = roll_by_largest_gap(comp)
            lon_r = np.roll(lon, shift)
            ys, xs = np.where(comp_rolled)
            proxy = haversine_km(float(lat[ys.min()]), float(lon_r[xs.min()]),
                                 float(lat[ys.max()]), float(lon_r[xs.max()]))

            if proxy < 0.5 * min_length_km:
                continue

            L, _ = centerline_length_smooth(comp, lat, lon, n_points=200, spline_s=300)
            if L >= min_length_km:
                keep[t][comp] = 1

    ar_tag = xr.DataArray(
        keep, dims=("time", "lat", "lon"),
        coords={"time": ds["time"], "lat": lat, "lon": lon},
        name="ar_binary_tag"
    ).astype("i1")
    ds_out = xr.Dataset({"ar_binary_tag": ar_tag})

    ds_out["ar_binary_tag"].attrs.update({
        # You asked to use this exact string as *description*
        "description": "binary indicator of atmospheric river",
        "scheme": algo_scheme,
        "version": algo_version,
        # keep a short functional note as well:
        "credits": "Developed by Victoire Buffet, Vincent Favier, and Benjamin Pohl",
    })
    ds_out["lat"].attrs.update({
        "standard_name": "latitude", "long_name": "Latitude",
        "units": "degrees_north", "axis": "Y",
    })
    ds_out["lon"].attrs.update({
        "standard_name": "longitude", "long_name": "Longitude",
        "units": "degrees_east", "axis": "X",
    })
    ds_out["time"].attrs.update({
        "standard_name": "time", "long_name": "Time", "axis": "T",
    })

    # inherit time encoding (units, calendar) from the source ERA5 file
    # (prevents conflicts and matches the example if the source has 1900-01-01 hours)
    ds_out["time"].encoding.update({
    "units": f"minutes since {year}-01-01 00:00:00",
    "calendar": "standard",
})

    # --- global attributes (optional but matches example style) ---
    ds_out.attrs.update({
        "Conventions": "CF-1.6",
        "description": "ARTMIP file format (Ullrich)",
    })

    # -----------------------
    # File naming (dataset, algo, frequency, dates) + .nc
    # -----------------------
    freq_label = _freq_label_from_time(ds)  # e.g. "3hourly"
    start = _yyyymmdd(ds["time"].values[0])
    end   = _yyyymmdd(ds["time"].values[-1])

    # Use ERA5 as dataset tag here. For MERRA2 runs, change to "MERRA2".
    dataset_tag = "ERA5"
    fname = f"{dataset_tag}.ar_tag.{algo_scheme}.{freq_label}.{start}-{end}.nc"
    out_path = os.path.join(out_dir, fname)

    # only compress the data var; coords don’t need it
    enc = {
        "ar_binary_tag": {
            "zlib": True, "complevel": 1, "shuffle": True,
            "chunksizes": (8, len(lat), len(lon)),
            "dtype": "i1",
        }
    }

    # Write NetCDF-4 in a .nc file, with unlimited time dim (like example shows)
    ds_out.to_netcdf(
        out_path,
        engine="netcdf4",
        format="NETCDF4",
        encoding=enc,
        unlimited_dims="time",
    )
    ds.close()
    return out_path

# =========================
# Quick 3-panel plot for a given timestep
# =========================
def _plot_path_dateline_safe(ax, plon, plat, lon_axis_min, wrap_thresh=180.0, **kw):
    """Plot a path without drawing a long segment across the wrap."""
    plon = np.asarray(plon, dtype=float)
    plat = np.asarray(plat, dtype=float)

    if lon_axis_min >= 0:                  # 0..360 domain
        plon = np.mod(plon, 360.0)
    else:                                  # -180..180 domain
        plon = (plon + 180.0) % 360.0 - 180.0

    jumps = np.where(np.abs(np.diff(plon)) > wrap_thresh)[0] + 1
    lon_segs = np.split(plon, jumps)
    lat_segs = np.split(plat, jumps)

    for xseg, yseg in zip(lon_segs, lat_segs):
        if xseg.size >= 2:
            ax.plot(xseg, yseg, **kw)

def plot_three_panel_era5(year, idx, u_pattern, v_pattern, pctl_path,
                          lon_domain="keep",
                          u_name="viwve", v_name="viwvn",
                          min_area_km2=MIN_AREA_KM2, min_length_km=MIN_LENGTH_KM):
    to_180 = (lon_domain == "-180")
    ds = _open_era5_ivt_year(
        year, u_pattern, v_pattern,
        u_names=(u_name, "vertical_integral_of_eastward_water_vapour_flux", "uIVT"),
        v_names=(v_name, "vertical_integral_of_northward_water_vapour_flux", "vIVT"),
        to_180=to_180
    )
    lat = ds["lat"].values
    lon = ds["lon"].values
    area_m2 = gridcell_area(lat, lon)
    tstamp = np.datetime_as_string(ds["time"].values[idx], unit="h")

    pctl = xr.open_dataset(pctl_path)["ivt_pctl"]
    if "latitude" in pctl.dims:
        pctl = pctl.rename({"latitude": "lat", "longitude": "lon"})
    if to_180 and float(pctl["lon"].min()) >= 0.0:
        pctl = pctl.assign_coords(lon=((pctl["lon"] + 180) % 360) - 180).sortby("lon")
    pctl = align_pctl_to_ivt_grid(pctl, ds["IVT"])

    doy_val = int(ds["time"].dt.dayofyear.isel(time=idx).item())
    is_leap = (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))
    if (not is_leap) and (doy_val >= 60):
        doy_val += 1
    thr_da  = pctl.sel(dayofyear=doy_val)
    ivt_t_da = ds["IVT"].isel(time=idx)

    ex_t  = (ivt_t_da > thr_da).values
    ivt_t = ivt_t_da.values
    area_t = area_filter_2d(ex_t, area_m2, min_area_km2=min_area_km2).astype(bool)

    lab, nlab = label_torus_lon(area_t)
    keep_t = np.zeros_like(area_t, dtype=np.int8)
    paths = []
    for lab_id in range(1, nlab+1):
        comp = (lab == lab_id)
        if comp.sum() < 30:
            continue

        comp_rolled, shift = roll_by_largest_gap(comp)
        lon_r = np.roll(lon, shift)
        ys, xs = np.where(comp_rolled)
        proxy = haversine_km(float(lat[ys.min()]), float(lon_r[xs.min()]),
                             float(lat[ys.max()]), float(lon_r[xs.max()]))
        if proxy < 0.5 * min_length_km:
            continue

        L, (plon, plat) = centerline_length_smooth(comp, lat, lon, n_points=200, spline_s=300)
        if L >= min_length_km and plon is not None:
            keep_t[comp] = 1
            paths.append((plon, plat))

    # plot with ascending-lat arrays to keep imshow+contour aligned
    lat_inc = np.all(np.diff(lat) > 0)
    if lat_inc:
        lat_p, ivt_p, ex_p, area_p, keep_p = lat, ivt_t, ex_t, area_t, keep_t
    else:
        lat_p   = lat[::-1]
        ivt_p   = ivt_t[::-1, :]
        ex_p    = ex_t[::-1, :]
        area_p  = area_t[::-1, :]
        keep_p  = keep_t[::-1, :]

    extent = [lon.min(), lon.max(), lat_p.min(), lat_p.max()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    ax = axes[0]
    im1 = ax.imshow(ivt_p, extent=extent, origin="lower", cmap="viridis",
                    aspect="auto", interpolation="nearest")
    ax.contour(lon, lat_p, ex_p.astype(float), levels=[0.5], colors="crimson", linewidths=1.6)
    ax.set_title(f"Exceedance • {tstamp}"); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="IVT")

    ax = axes[1]
    im2 = ax.imshow(ivt_p, extent=extent, origin="lower", cmap="viridis",
                    aspect="auto", interpolation="nearest")
    ax.contour(lon, lat_p, area_p.astype(float), levels=[0.5], colors="dodgerblue", linewidths=1.6)
    ax.set_title("Area-filtered (toroidal)"); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="IVT")

    ax = axes[2]
    im3 = ax.imshow(ivt_p, extent=extent, origin="lower", cmap="viridis",
                    aspect="auto", interpolation="nearest")
    ax.contour(lon, lat_p, keep_p.astype(float), levels=[0.5], colors="lime", linewidths=1.8)
    for (plon, plat) in paths:
        _plot_path_dateline_safe(ax, plon, plat, lon_axis_min=lon.min(),
                                 wrap_thresh=180.0, color="black", lw=2.0)
    ax.set_title(f"Length-filtered (≥{min_length_km} km) + centerline")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label="IVT")

    plt.show()
    ds.close()

# =========================
# CLI (commented out for notebook use)
# =========================
# def main():
#     p = argparse.ArgumentParser(description="ARTMIP AR tagging (ERA5): IVT built from u/v flux.")
#     sub = p.add_subparsers(dest="cmd", required=True)
#     ...
# if __name__ == "__main__":
#     main()

# =========================
# For interactive execution
# =========================
# build_all_days_and_write_era5(
#     PCTL_DOY_PATH, YEARS, ROLLING_WINDOW_D, ERA5_U_PATTERN, ERA5_V_PATTERN,
#     to_180=False,
#     u_names=("viwve","vertical_integral_of_eastward_water_vapour_flux","uIVT"),
#     v_names=("viwvn","vertical_integral_of_northward_water_vapour_flux","vIVT")
# )

year = os.environ.get("year")

if year is None:
    raise ValueError("No year provided. Make sure the 'year' environment variable is set.")

print(f"Running script for year: {year}")

# Now you can use `year` as a string (or convert to int if needed)
year = int(year)

print(year)
write_artmip_year_era5(
    year, ERA5_U_PATTERN, ERA5_V_PATTERN, PCTL_DOY_PATH, OUT_DIR_DEFAULT,
    to_180=False,
    algo_scheme=ALGO_SCHEME, algo_version=ALGO_VERSION,
    min_area_km2=MIN_AREA_KM2, min_length_km=MIN_LENGTH_KM,
    u_names=("viwve","vertical_integral_of_eastward_water_vapour_flux","uIVT"),
    v_names=("viwvn","vertical_integral_of_northward_water_vapour_flux","vIVT")
)
