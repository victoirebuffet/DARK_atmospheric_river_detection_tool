#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:56:38 2025

@author: buffetv
"""
 
import os
from glob import glob
import numpy as np
import xarray as xr
from scipy import ndimage
from netCDF4 import Dataset as NC4  # only for writing per-year outputs

# ============================================================
# ERA5 helpers: coords, IVT from u/v, cropping, lon domain
# ============================================================
def _normalize_era5_coords(ds, to_180=False):
    """Rename to (time, lat, lon), collapse expver, optional lon→[-180,180]."""
    if "latitude" in ds.coords:  ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})
    if "valid_time" in ds.coords: ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        try:    ds = ds.sel(expver=ds["expver"].max())
        except: ds = ds.isel(expver=-1)
        ds = ds.drop_vars("expver", errors="ignore").reset_coords(names=["expver"], drop=True)
    if to_180 and float(ds["lon"].min()) >= 0.0:
        ds = ds.assign_coords(lon=((ds["lon"] + 180.0) % 360.0) - 180.0).sortby("lon")
    return ds

def _lat_slice(lat_vals, lo=-90.0, hi=-15.0):
    """Slice for [lo,hi] regardless of ascending/descending lat."""
    lat = np.asarray(lat_vals)
    return slice(hi, lo) if lat[0] > lat[-1] else slice(lo, hi)

def _pick_first_var(ds, candidates):
    for n in candidates:
        if n in ds.data_vars: return ds[n]
    raise KeyError(f"None of {candidates} in {list(ds.data_vars)}")

def _freq_label_from_time_da(da):
    t = da["time"].values
    if t.size < 2: return "unknown"
    dt_h = float((t[1] - t[0]) / np.timedelta64(1, "h"))
    m = {1.0: "hourly", 3.0: "3hourly", 6.0: "6hourly", 24.0: "daily"}
    return m.get(round(dt_h), f"{dt_h:g}hourly")

def _yyyymmdd(t64):  # numpy datetime64 -> YYYYMMDD
    return np.datetime_as_string(t64, unit="D").replace("-", "")

# ============================================================
# Geometry: earth radius & cell area, toroidal labeling
# ============================================================
def earth_radius(lat_deg):
    a = 6378137.0; b = 6356752.3142
    e2 = 1.0 - (b**2 / a**2)
    lat = np.deg2rad(lat_deg)
    lat_gc = np.arctan((1.0 - e2) * np.tan(lat))
    return (a * np.sqrt(1.0 - e2)) / np.sqrt(1.0 - e2 * np.cos(lat_gc)**2)

def gridcell_area(lat_1d, lon_1d):
    """Area per grid cell on regular lat/lon grid [m^2]."""
    ylat, xlon = np.meshgrid(lat_1d, lon_1d, indexing='ij')
    R = earth_radius(ylat)
    # respect actual lat spacing & ordering
    dlat = np.empty_like(lat_1d, dtype=float)
    dlat[1:-1] = (lat_1d[2:] - lat_1d[:-2]) / 2.0
    dlat[0]    =  lat_1d[1] - lat_1d[0]
    dlat[-1]   =  lat_1d[-1] - lat_1d[-2]
    dlat2d = np.deg2rad(dlat)[:, None]
    dlon = np.diff(lon_1d).mean()
    dlon2d = np.deg2rad(dlon)
    dy = dlat2d * R
    dx = dlon2d  * R * np.cos(np.deg2rad(ylat))
    return np.abs(dy * dx)

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

def label_torus_lon(binary2d_bool):
    """8-connected components with periodic connectivity in longitude."""
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
    remap = np.zeros(nlab+1, dtype=int)
    for k in range(1, nlab+1): remap[k] = new_id[roots[k]]
    out = np.zeros_like(lab, dtype=int); nz = lab > 0
    out[nz] = remap[lab[nz]]
    return out, len(uniq)

def area_filter_2d(mask2d_bool, area_m2, min_area_km2):
    lab, nlab = label_torus_lon(mask2d_bool)
    if nlab == 0:
        return np.zeros_like(mask2d_bool, dtype=np.int8)
    sums = np.bincount(lab.ravel(), weights=area_m2.ravel(), minlength=nlab+1)
    keep_ids = np.where((sums/1e6) >= float(min_area_km2))[0]
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0:
        return np.zeros_like(mask2d_bool, dtype=np.int8)
    return np.isin(lab, keep_ids).astype(np.int8)

# ============================================================
# Label matching across time
# ============================================================
def _match_labels(prev_lab, cur_lab):
    """
    Map each current label -> previous label with max overlap (0 if none).
    Returns (cur2prev, prev2cur).
    """
    if prev_lab is None or prev_lab.max() == 0 or cur_lab.max() == 0:
        return {}, {}
    ncur = cur_lab.max()
    key = prev_lab.ravel().astype(np.int64) * (ncur + 1) + cur_lab.ravel().astype(np.int64)
    cnt = np.bincount(key)
    cur2prev, prev2cur = {}, {}
    for c in range(1, ncur+1):
        vals = cnt[c::(ncur+1)]
        if vals.size == 0: continue
        p = int(np.argmax(vals))
        if vals[p] == 0:   continue
        cur2prev[c] = p
        prev2cur.setdefault(p, []).append(c)
    return cur2prev, prev2cur

# ============================================================
# Parent “born outside Antarctica” criterion
# ============================================================
def _born_outside_ok(ar_mask_bool, lsm_bool, lat_vals, outside_criterion="either", outside_lat=-65.0):
    """
    outside_criterion in {"lsm","lat","either","both"}.
      "lsm"   : at least one parent pixel off Antarctica (¬LSM)
      "lat"   : at least one parent pixel with lat >= outside_lat (e.g., -65)
      "either": lsm OR lat
      "both"  : lsm AND lat
    """
    if not ar_mask_bool.any():
        return False
    cond_lsm = np.logical_not(lsm_bool[ar_mask_bool]).any()
    lat2d = np.broadcast_to(lat_vals[:, None], ar_mask_bool.shape)
    cond_lat = (lat2d >= outside_lat)[ar_mask_bool].any()
    if outside_criterion == "lsm":   return bool(cond_lsm)
    if outside_criterion == "lat":   return bool(cond_lat)
    if outside_criterion == "both":  return bool(cond_lsm and cond_lat)
    return bool(cond_lsm or cond_lat)

# ============================================================
# Open ERA5 u/v → IVT (cropped), and build yearly exceed masks
# ============================================================
def open_era5_ivt_year_cropped(
    year, u_pattern, v_pattern, lat_band=(-90,-15), time_chunks=96, lon_domain="keep",
    u_names=("viwve","vertical_integral_of_eastward_water_vapour_flux","uIVT"),
    v_names=("viwvn","vertical_integral_of_northward_water_vapour_flux","vIVT"),
):
    """Return Dataset with IVT(time,lat,lon), cropped on load; lon domain consistent with lon_domain."""
    uf = u_pattern.format(year=year); vf = v_pattern.format(year=year)
    if not os.path.exists(uf): raise FileNotFoundError(uf)
    if not os.path.exists(vf): raise FileNotFoundError(vf)
    to_180 = (lon_domain == "-180")
    du = xr.open_dataset(uf, chunks={"time": time_chunks})
    dv = xr.open_dataset(vf, chunks={"time": time_chunks})
    du = _normalize_era5_coords(du, to_180=to_180)
    dv = _normalize_era5_coords(dv, to_180=to_180)
    sl = _lat_slice(du["lat"].values, *lat_band)
    du = du.sel(lat=sl); dv = dv.sel(lat=sl)
    du, dv = xr.align(du, dv, join="inner")
    u = _pick_first_var(du, u_names)
    v = _pick_first_var(dv, v_names)
    ivt = xr.apply_ufunc(np.hypot, u, v, dask="allowed", output_dtypes=[u.dtype]).rename("IVT")
    return xr.Dataset({"IVT": ivt})

def _align_pctl_to_ivt_grid(pctl_da, ivt_da, lon_domain="keep"):
    p = pctl_da
    if "latitude" in p.dims: p = p.rename({"latitude":"lat","longitude":"lon"})
    if lon_domain == "-180" and float(p["lon"].min()) >= 0.0:
        p = p.assign_coords(lon=((p["lon"] + 180.0) % 360.0) - 180.0).sortby("lon")
    return p.sel(lat=ivt_da["lat"], lon=ivt_da["lon"], drop=True)

def write_exceed_binary_year_lazy_era5(
    year, u_pattern, v_pattern, pctl_path, out_dir,
    time_chunks=96, lat_band=(-90,-15), lon_domain="keep",
    u_names=("viwve","vertical_integral_of_eastward_water_vapour_flux","uIVT"),
    v_names=("viwvn","vertical_integral_of_northward_water_vapour_flux","vIVT"),
):
    """
    Build and write exceed_binary := 1{hypot(viwve,viwvn) > ivt_pctl(DOY)} for ERA5.
    Crops latitude at open(); preserves chosen lon_domain ("keep" 0..360 or "-180").
    """
    os.makedirs(out_dir, exist_ok=True)

    ds_ivt = open_era5_ivt_year_cropped(
        year, u_pattern, v_pattern, lat_band=lat_band, time_chunks=time_chunks,
        lon_domain=lon_domain, u_names=u_names, v_names=v_names
    )
    pctl = xr.open_dataset(pctl_path)["ivt_pctl"]
    pctl = _align_pctl_to_ivt_grid(pctl, ds_ivt["IVT"], lon_domain=lon_domain)

    doy = ds_ivt["time"].dt.dayofyear
    is_leap = (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))
    if not is_leap:
        doy = xr.where(doy >= 60, doy + 1, doy)

    thr = pctl.sel(dayofyear=doy)
    exceed = (ds_ivt["IVT"] > thr).astype("i1").rename("exceed_binary")

    ds_out = xr.Dataset({"exceed_binary": exceed})
    ds_out["lat"].attrs.update(dict(standard_name="latitude", units="degrees_north", axis="Y"))
    ds_out["lon"].attrs.update(dict(standard_name="longitude", units="degrees_east", axis="X"))
    ds_out["time"].attrs.update(dict(standard_name="time", axis="T"))
    tag_domain = "LONm180" if lon_domain == "-180" else "LON0360"
    ds_out.attrs.update(dict(Conventions="CF-1.6",
                             note=f"ERA5 exceed (IVT from viwve/viwvn). Antarctic crop at open(); lon_domain={tag_domain}"))
    enc = {"exceed_binary": {"zlib": True, "shuffle": True, "complevel": 1, "dtype": "i1"}}
    out_path = os.path.join(out_dir, f"exceed_binary.ERA5.IVT.{tag_domain}.ANT_-90_to_-15.{year}.nc")
    ds_out.to_netcdf(out_path, engine="netcdf4", format="NETCDF4", encoding=enc, unlimited_dims="time")
    return out_path

# ============================================================
# Multi-file open (exceed, AR tags, LSM) with safe preprocess
# ============================================================
def _filter_files_with_var(paths, var_candidates):
    keep = []
    for p in sorted(paths):
        try:
            with NC4(p, "r") as nc:
                vset = set(nc.variables.keys())
            if any(v in vset for v in var_candidates):
                keep.append(p)
            else:
                print(f"[skip] {os.path.basename(p)} – missing {var_candidates}; has {sorted(list(vset))[:8]}")
        except Exception as e:
            print(f"[skip] {os.path.basename(p)} – cannot open ({e})")
    if not keep:
        raise FileNotFoundError(f"No files with {var_candidates}")
    return keep

def _preprocess_exceed_era5(ds, lat_band):
    ds = _normalize_era5_coords(ds)
    ds = ds.sel(lat=_lat_slice(ds["lat"].values, *lat_band))
    var = "exceed_binary"
    if var not in ds.data_vars:
        for cand in ("exceed", "exceed_mask"):
            if cand in ds.data_vars: var = cand; break
    ds = ds[[var]]
    # drop stray coords (e.g., dayofyear, quantile)
    keep = {"time","lat","lon"}
    drop = [c for c in ds.coords if c not in keep]
    if drop: ds = ds.drop_vars(drop, errors="ignore")
    ds[var] = ds[var].astype("i1")
    return ds

def open_exceed_mf_era5(files_or_glob, lat_band, chunks_time):
    files = glob(files_or_glob) if isinstance(files_or_glob, str) else list(files_or_glob)
    files = _filter_files_with_var(files, ("exceed_binary","exceed","exceed_mask"))
    ex = xr.open_mfdataset(
        files, combine="by_coords",
        preprocess=lambda d: _preprocess_exceed_era5(d, lat_band),
        data_vars="minimal", coords="minimal", join="outer",
        chunks=None, parallel=False, engine="netcdf4",
    )
    da = _pick_first_var(ex, ("exceed_binary","exceed","exceed_mask"))
    if da.name != "exceed_binary": da = da.rename("exceed_binary")
    return da.chunk({"time": chunks_time})

def _preprocess_ar_era5(ds, lat_band):
    ds = _normalize_era5_coords(ds)
    ds = ds.sel(lat=_lat_slice(ds["lat"].values, *lat_band))
    var = "ar_binary_tag"
    for cand in ("ar_binary_mask","ar_tag"):
        if var not in ds.data_vars and cand in ds.data_vars: var = cand
    ds = ds[[var]]
    keep = {"time","lat","lon"}
    drop = [c for c in ds.coords if c not in keep]
    if drop: ds = ds.drop_vars(drop, errors="ignore")
    ds[var] = ds[var].astype("i1")
    return ds

def open_artag_mf_era5(files_or_glob, lat_band, chunks_time, tag_candidates=("ar_binary_tag","ar_binary_mask","ar_tag")):
    files = glob(files_or_glob) if isinstance(files_or_glob, str) else list(files_or_glob)
    files = _filter_files_with_var(files, tag_candidates)
    ar = xr.open_mfdataset(
        files, combine="by_coords",
        preprocess=lambda d: _preprocess_ar_era5(d, lat_band),
        data_vars="minimal", coords="minimal", join="outer",
        chunks=None, parallel=False, engine="netcdf4",
    )
    da = _pick_first_var(ar, tag_candidates)
    if da.name != "ar_binary_tag": da = da.rename("ar_binary_tag")
    return da.chunk({"time": chunks_time})

def open_lsm_cropped_era5(lsm_path, lat_band, positive_threshold=0.0):
    """Return boolean Antarctic mask (True=land/ice) cropped at load."""
    ds = xr.open_dataset(lsm_path)
    ds = _normalize_era5_coords(ds)
    ds = ds.sel(lat=_lat_slice(ds["lat"].values, *lat_band))
    da = _pick_first_var(ds, ("lsm","LSM","sftlf","FRLAND","LANDFRAC","LANDMASK"))
    return (da > positive_threshold)

# ============================================================
# Prepare one output file (per year) with 2 variables
# ============================================================
def _prepare_year_nc_twovars(out_path, time_vals, lat_vals, lon_vals, comp_level=1):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nc = NC4(out_path, "w")
    nc.createDimension("time", None)
    nc.createDimension("lat",  len(lat_vals))
    nc.createDimension("lon",  len(lon_vals))
    vtime = nc.createVariable("time", "f8", ("time",))
    vlat  = nc.createVariable("lat",  "f8", ("lat",))
    vlon  = nc.createVariable("lon",  "f8", ("lon",))
    vbin  = nc.createVariable("ar_binary_tag", "i1", ("time","lat","lon"),
                              zlib=True, shuffle=True, complevel=comp_level)
    vtype = nc.createVariable("ar_type", "i1", ("time","lat","lon"),
                              zlib=True, shuffle=True, complevel=comp_level)
    vlat.standard_name="latitude"; vlat.units="degrees_north"; vlat.axis="Y"
    vlon.standard_name="longitude"; vlon.units="degrees_east";  vlon.axis="X"
    vtime.standard_name="time"; vtime.units="hours since 1900-01-01 00:00:00"; vtime.calendar="standard"; vtime.axis="T"
    vlat[:] = lat_vals; vlon[:] = lon_vals
    hours_since0 = (time_vals - np.datetime64("1900-01-01T00:00:00")) / np.timedelta64(1, "h")
    vtime[:] = hours_since0.astype("f8")
    nc.Conventions = "CF-1.6"
    nc.description = "ERA5 AR parents+children (per year), time-continuous tracking across all opened years"
    return nc

# ============================================================
# Children builder (open ALL years, write PER YEAR) – ERA5
# ============================================================
def build_ar_children_catalog_per_year_era5(
    exceed_files,             # list or glob of yearly exceed files (built by write_exceed_binary_year_lazy_era5)
    ar_files,                 # list or glob of ERA5 parent AR files (your tilted AR catalog)
    lsm_path,                 # Antarctic land/ice mask
    out_dir,                  # directory where one file per year will be written
    dataset_tag="ERA5",
    algo_children_tag="Buffet_children",
    lat_band=(-90.0, -15.0),
    time_chunks=96,
    min_area_km2=200000.0,
    apply_area_filter=True,
    require_parent_born_outside=False,
    outside_criterion="either",   # "lsm" | "lat" | "either" | "both"
    outside_lat=-65.0,
    lsm_positive_threshold=0.0
):
    # Open all years together (continuous tracking)
    ex  = open_exceed_mf_era5(exceed_files, lat_band, chunks_time=time_chunks).astype("i1")
    ar  = open_artag_mf_era5(ar_files,     lat_band, chunks_time=time_chunks).astype("i1")
    lsm = open_lsm_cropped_era5(lsm_path,  lat_band, positive_threshold=lsm_positive_threshold)

    # Grid/time checks
    if not np.array_equal(ex["lat"].values, ar["lat"].values) or not np.array_equal(ex["lon"].values, ar["lon"].values):
        raise ValueError("Exceed and AR grids (lat/lon) differ after cropping.")
    if not np.array_equal(ex["lat"].values, lsm["lat"].values) or not np.array_equal(ex["lon"].values, lsm["lon"].values):
        raise ValueError("LSM grid does not match exceed/AR grids after cropping.")
    if not np.array_equal(ex["time"].values, ar["time"].values):
        raise ValueError("Exceed and AR time coordinates differ; align them first.")

    lat_vals = ex["lat"].values
    lon_vals = ex["lon"].values
    area_m2  = gridcell_area(lat_vals, lon_vals)
    lsm_bool = lsm.values.astype(bool)

    # Time bookkeeping for per-year outputs
    tvals_all  = ex["time"].values
    years_arr  = ex["time"].dt.year.values
    years_uniq = np.unique(years_arr)
    freq_label = _freq_label_from_time_da(ex)

    pos_in_year = np.empty_like(years_arr, dtype=np.int64)
    year_to_times = {}
    for y in years_uniq:
        idxs = np.where(years_arr == y)[0]
        pos_in_year[idxs] = np.arange(len(idxs), dtype=np.int64)
        year_to_times[y]  = tvals_all[idxs]

    # Seam-safe persistent state
    next_track_id = 1
    prev_lab = None
    prev_label_to_track = {}
    had_ar_overlap = {}
    parent_born_ok = {}
    touches_antarct = {}

    # Streaming over continuous time; switch output file at year boundaries
    current_year = None
    nc = None
    vbin = vtype = None

    for it in range(ex.sizes["time"]):
        y = int(years_arr[it])

        if current_year != y:
            if nc is not None:
                nc.close()
            year_times = year_to_times[y]
            start_s = _yyyymmdd(year_times[0]); end_s = _yyyymmdd(year_times[-1])
            out_name = f"{dataset_tag}.ar_children_tag.{algo_children_tag}.{freq_label}.{start_s}-{end_s}.nc"
            out_path = os.path.join(out_dir, out_name)
            nc = _prepare_year_nc_twovars(out_path, year_times, lat_vals, lon_vals, comp_level=1)
            vbin  = nc.variables["ar_binary_tag"]
            vtype = nc.variables["ar_type"]
            current_year = y

        # Current step
        ex_t = ex.isel(time=it).values.astype(bool)
        if apply_area_filter:
            ex_t = area_filter_2d(ex_t, area_m2, min_area_km2=min_area_km2).astype(bool)

        cur_lab, ncur = label_torus_lon(ex_t)
        ar_t = ar.isel(time=it).values.astype(bool)

        # Associate labels to previous step
        cur2prev, _ = _match_labels(prev_lab, cur_lab)

        # Build label -> track_id map
        cur_label_to_track = {}
        for c in range(1, ncur+1):
            p = cur2prev.get(c, 0)
            if p and p in prev_label_to_track:
                tid = prev_label_to_track[p]
            else:
                tid = next_track_id; next_track_id += 1
                had_ar_overlap[tid]  = False
                parent_born_ok[tid]  = (not require_parent_born_outside)
                touches_antarct[tid] = False
            cur_label_to_track[c] = tid

        # Outputs for this step
        type_out  = np.zeros_like(ex_t, dtype=np.int8)  # 0/1/2
        child_out = np.zeros_like(ex_t, dtype=np.int8)

        # mark parents
        type_out[ar_t] = 1

        # Update states + mark children
        for c in range(1, ncur+1):
            tid = cur_label_to_track[c]
            comp_mask = (cur_lab == c)
        
            # Antarctic contact at THIS timestep only
            touches_now = (comp_mask & lsm_bool).any()
            if touches_now:
                # keep the historical flag if you still want to track it for diagnostics
                touches_antarct[tid] = True
        
            overlap_now = (comp_mask & ar_t).any()
            if overlap_now and not had_ar_overlap[tid]:
                had_ar_overlap[tid] = True
                if require_parent_born_outside:
                    parent_mask_local = (ar_t & comp_mask)
                    ok = _born_outside_ok(parent_mask_local, lsm_bool, lat_vals,
                                          outside_criterion=outside_criterion,
                                          outside_lat=outside_lat)
                    parent_born_ok[tid] = bool(ok)
        
            # Child only if it touches Antarctica NOW and is not parent NOW
            if had_ar_overlap[tid] and parent_born_ok[tid] and touches_now and (not overlap_now):
                child_out[comp_mask] = 1
                type_out[comp_mask & (~ar_t)] = 2

        # ar_binary_tag = parent OR child
        bin_out = ((ar_t) | (child_out > 0)).astype(np.int8)

        # write at the within-year index
        k = int(pos_in_year[it])
        vbin[k, :, :]  = bin_out
        vtype[k, :, :] = type_out

        # continuity
        prev_lab = cur_lab
        prev_label_to_track = {lab: cur_label_to_track[lab] for lab in range(1, ncur+1)}

    if nc is not None:
        nc.close()

# ============================================================
# —— Example interactive usage (call these from a notebook) —
# ============================================================
# 1) Build ERA5 exceed masks (once per problematic/missing year)
year = os.environ.get("year")

if year is None:
    raise ValueError("No year provided. Make sure the 'year' environment variable is set.")

print(f"Running script for year: {year}")

# Now you can use `year` as a string (or convert to int if needed)
year = int(year)

print(year)

write_exceed_binary_year_lazy_era5(
    year=year,
    u_pattern="/work/crct/vb234125/tilted_ARs/IVT/vertical_integral_of_eastward_water_vapour_flux_{year}_reanaHS.nc",
    v_pattern="/work/crct/vb234125/tilted_ARs/IVT/vertical_integral_of_northward_water_vapour_flux_{year}_reanaHS.nc",
    pctl_path="/work/crct/vb234125/tilted_ARs/IVT/percentile_98_IVT_rolling_15D_era5.nc",
    out_dir="/work/crct/vb234125/tilted_ARs/IVT",
    time_chunks=96, lat_band=(-90,-15), lon_domain="keep"
)
# 2) Build children (opens ALL exceed+AR years together; writes ONE file per year)
build_ar_children_catalog_per_year_era5(
    exceed_files="/Users/buffetv/artmip_out/era5/exceed_binary.ERA5.IVT.LON0360.ANT_-90_to_-15.*.nc",
    ar_files="/Users/buffetv/artmip_out/era5/ERA5.ar_tag.Buffet.6hourly.*.nc",
    lsm_path="/Users/buffetv/artmip_out/era5/lsm_era5.nc",
    out_dir="/Users/buffetv/artmip_out/era5",
    dataset_tag="ERA5", algo_children_tag="Buffet_children_IVT",
    lat_band=(-90,-15), time_chunks=96,
    min_area_km2=200000.0, apply_area_filter=True,
    require_parent_born_outside=False, outside_criterion="either", outside_lat=-65.0,
    lsm_positive_threshold=0.0
)
