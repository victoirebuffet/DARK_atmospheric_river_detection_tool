# DARK atmospheric-river detection tool

Configuration-driven DARK and DARK + children detection for ERA5, MERRA2,
CESM and compatible rectilinear latitude/longitude datasets.

[Quick start](#quick-start) · [Examples](#examples) · [Thresholds](#thresholds) · [Parallel execution](#parallel-execution) · [Outputs](#outputs)

---

## Quick start

~~~bash
git clone https://github.com/victoirebuffet/DARK_atmospheric_river_detection_tool.git
cd DARK_atmospheric_river_detection_tool
python -m pip install -r dark/requirements.txt

cp dark/example_merra2.yml my_dark_run.yml
python dark/dark.py my_dark_run.yml --validate-config
python dark/submit.py my_dark_run.yml --scheduler sge --dry-run
~~~

> [!TIP]
> All scientific choices are in one YAML or JSON configuration: files,
> variables, region, percentile, reference period and children. Scheduler
> choices are also configurable and never hard-coded in DARK.

## At a glance

| Capability | Supported |
| --- | --- |
| IVT input | Scalar IVT, or vector components u + v |
| Data layout | One file/year, separate component files, or multi-year files |
| Geography | Northern Hemisphere, Southern Hemisphere, global or regional |
| Longitude | 0–360°, −180–180°, including seam-crossing regional domains |
| Threshold | Rolling percentile and optional adaptive multiplier |
| Execution | Local machine, SGE, Slurm or PBS |

## Examples

Every configuration follows the same top-level layout: <code>data</code>,
<code>percentile</code>, <code>dark</code>, then <code>children</code>.
Only the IVT input declaration inside <code>data</code> changes by dataset.

<details open>
<summary><strong>Scalar IVT — MERRA2 or model output</strong></summary>

~~~yaml
dataset: MERRA2
output_dir: /work/USER/DARK/merra2_q98
years: [1980, 2023]
data:
  ivt_type: scalar
  file_pattern: /data/ARTMIP_MERRA_2D_{year}_3h.nc
  ivt_variable: IVT
  periodic_longitude: auto
  # Optional only for non-standard coordinate/dimension names:
  # coordinates: {time: time, lat: lat, lon: lon}
  # dimensions: {time: time, lat: lat, lon: lon}
  region:
    latitude: [-90, -15]
    longitude: [0, 360]
percentile:
  quantile: 0.98
  window_days: 15
  reference_years: [1980, 2023]
dark:
  min_area_km2: 200000
  min_length_km: 2000
children:
  enabled: false
~~~

~~~bash
python dark/dark.py my_dark_run.yml
~~~

</details>

<details>
<summary><strong>Vector IVT — ERA5 with DARK + children</strong></summary>

~~~yaml
dataset: ERA5
output_dir: /work/USER/DARK/era5_q98
years: [1979, 2025]
data:
  ivt_type: vector
  u_file_pattern: /data/vertical_integral_of_eastward_water_vapour_flux_{year}.nc
  v_file_pattern: /data/vertical_integral_of_northward_water_vapour_flux_{year}.nc
  u_variable: viwve
  v_variable: viwvn
  periodic_longitude: auto
  # Optional only for non-standard coordinate/dimension names:
  # coordinates: {time: time, lat: latitude, lon: longitude}
  # dimensions: {time: time, lat: latitude, lon: longitude}
  region:
    latitude: [-90, -15]
    longitude: [0, 360]
percentile:
  quantile: 0.98
  window_days: 15
dark:
  min_area_km2: 200000
  min_length_km: 2000
children:
  enabled: true
  land_mask_file: /data/land_sea_mask.nc
  land_mask_variable: lsm
  land_mask_threshold: 0
~~~

For an SGE cluster, add a site-specific launcher block:

~~~yaml
launcher:
  python: /path/to/your/environment/bin/python
  environment_setup: |
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
  submit_options:
    sge: ["-cwd", "-V", "-q", "batch"]
~~~

</details>

<details>
<summary><strong>Multi-year files — CESM-style input</strong></summary>

~~~yaml
dataset: CESM2-LE
output_dir: /work/USER/DARK/cesm2_member001
years: [1990, 2009]
data:
  ivt_type: scalar
  member: member001
  file_glob: /data/{member}.IVT.*.nc
  ivt_variable: IVT
  periodic_longitude: auto
  # Optional only for non-standard coordinate/dimension names:
  # coordinates: {time: time, lat: lat, lon: lon}
  # dimensions: {time: time, lat: lat, lon: lon}
  region:
    latitude: [20, 90]
    longitude: [-180, 180]
percentile: {quantile: 0.98, window_days: 15}
dark: {min_area_km2: 200000, min_length_km: 2000}
children: {enabled: false}
~~~

</details>

<details>
<summary><strong>Children land mask — required for every dataset when enabled</strong></summary>

Whether the IVT comes from ERA5, MERRA2, CESM or another model, enabling
children requires a land mask. It may have a different resolution: it is
aligned to the IVT grid with nearest-neighbour interpolation.

~~~yaml
children:
  enabled: true
  land_mask_file: /data/land_mask.nc
  land_mask_variable: lsm
  land_mask_threshold: 0
  # Optional only if this file uses different coordinate/dimension names:
  land_mask_data:
    coordinates: {lat: latitude, lon: longitude}
    dimensions: {lat: latitude, lon: longitude}
~~~

</details>

<details>
<summary><strong>Non-standard coordinate/dimension names</strong></summary>

~~~yaml
data:
  coordinates: {time: valid_time, lat: latitude, lon: longitude}
  dimensions: {time: valid_time, lat: y, lon: x}
~~~

</details>

Ready-to-copy files: [ERA5](dark/example_era5.yml),
[MERRA2](dark/example_merra2.yml), and [CESM2](dark/example_cesm2.yml).

## Thresholds

<details open>
<summary><strong>Reference period</strong></summary>

By default all detection years are used to calculate percentiles. To tag
2000–2025 against a 1981–2010 baseline:

~~~yaml
years: [2000, 2025]
percentile:
  quantile: 0.98
  window_days: 15
  reference_years: [1981, 2010]
~~~

</details>

<details>
<summary><strong>Adaptive threshold</strong></summary>

The annual mean driver is smoothed and its ratio to the reference-period mean
multiplies the percentile threshold:

~~~yaml
adaptive_threshold:
  enabled: true
  smoothing_sigma_years: 2
  data:
    file_pattern: /data/total_column_water_vapor_{year}.nc
    variable: tcwv
  region:
    latitude: [-90, -60]
    longitude: [0, 360]
~~~

The output file <code>adaptive_threshold_factors.nc</code> stores the annual
mean, smoothed mean and applied factor.

</details>

## Geographic domain

| Option | Behaviour |
| --- | --- |
| <code>periodic_longitude: auto</code> | Default. Wrap only a truly global grid. |
| <code>periodic_longitude: false</code> | Never join across the left/right domain edge. |
| Regional domain crossing 0° | Remains non-periodic if it has a longitude gap. |
| <code>data.region.latitude: [20, 90]</code> | Northern Hemisphere example. |
| <code>data.region.latitude: [-90, -15]</code> | Southern Hemisphere example. |

The detector currently requires a 1-D rectilinear latitude/longitude grid.
MAR projected x/y grids need a dedicated projection-aware adapter and a
validation catalogue before support is added.

## Parallel execution

~~~text
percentiles
     │
     ├── DARK 2000 ──┬── children 2000
     ├── DARK 2001 ──┼── children 2001  (waits for DARK 2000 + 2001)
     └── DARK 2002 ─┴── children 2002  (waits for DARK 2001 + 2002)
~~~

All DARK years run in parallel. Children years also run in parallel: each
waits only for parent artifacts from the current and previous years.

~~~bash
python dark/submit.py my_dark_run.yml --scheduler sge --dry-run
python dark/submit.py my_dark_run.yml --scheduler sge
python dark/submit.py my_dark_run.yml --scheduler slurm
python dark/submit.py my_dark_run.yml --scheduler pbs
python dark/submit.py my_dark_run.yml --scheduler local
~~~

## Outputs

| File | Contents |
| --- | --- |
| <code>OUTPUT/percentiles.nc</code> | Rolling IVT percentile climatology |
| <code>OUTPUT/adaptive_threshold_factors.nc</code> | Optional adaptive factors |
| <code>OUTPUT/DATASET.DARK.YEAR.nc</code> | Final annual catalogue |
| <code>OUTPUT/.dark-work/YEAR.nc</code> | Temporary parent/exceedance file, only with children |
| <code>OUTPUT/.dark-submit/</code> | Generated scheduler scripts |

| Variable | Meaning |
| --- | --- |
| <code>ar_binary_tag</code> | 1 = DARK parent or child; 0 = no AR |
| <code>ar_type</code> | With children: 0 = none, 1 = parent, 2 = child |

## Validate before computing

~~~bash
python dark/dark.py my_dark_run.yml --validate-config
~~~

## Requirements

Python 3.10+ with:

~~~text
numpy  xarray  netCDF4  scipy  scikit-image  geopy  networkx  PyYAML
~~~

Install with <code>python -m pip install -r dark/requirements.txt</code>.
