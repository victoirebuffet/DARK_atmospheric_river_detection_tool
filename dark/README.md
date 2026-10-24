# DARK portable

`dark.py` replaces the dataset-specific ERA5/MERRA-2 launch scripts with one
configuration. It writes a rolling day-of-year IVT percentile file and then
per-year NetCDF DARK tags. Set `children.enabled: true` to produce `ar_type`
in the same pass: `0` is neither, `1` is a DARK parent and `2` is a child.

## Install

```bash
git clone YOUR_DARK_REPOSITORY
cd YOUR_DARK_REPOSITORY
python -m pip install -r dark/requirements.txt
```

## Threshold reference period and adaptive option

`years` is the detection period. By default it is also the percentile
reference period; set `percentile.reference_years: [START, END]` to build the
rolling climatology from only part of the available record. These years may be
outside the detection period as long as their input files are available.

An optional `adaptive_threshold` multiplies the percentile threshold by a
smoothed annual driver factor, reproducing the principle of the older tool:

```yaml
adaptive_threshold:
  enabled: true
  smoothing_sigma_years: 2
  data:
    file_pattern: /data/total_column_water_vapor_{year}.nc
    variable: tcwv
```

The driver defaults to the detection region, but `adaptive_threshold.region`
can override it. Its factor is its smoothed annual mean divided by the mean
over `percentile.reference_years`; values above one raise the threshold. The
saved `adaptive_threshold_factors.nc` records the annual means and factors.

```bash
python dark/dark.py my_run.yml
python dark/dark.py my_run.yml --build-percentiles-only
```

Set `data.ivt_type` explicitly: `scalar` means a direct IVT magnitude
(`file_pattern` + `ivt_variable`, e.g. MERRA-2 or a model); `vector` computes
the magnitude from `u` and `v`, either in one `file_pattern` or in separate
`u_file_pattern`/`v_file_pattern` files (ERA5). Coordinates `latitude`/`longitude`/`valid_time` are
normalised automatically; `data.region` crops any latitude/longitude domain.
For a quick validation without processing an entire year, add
`data.time_slice: ["2000-01-01", "2000-01-05T23:59:59"]`.
For CESM and other inputs split into multi-year chunks, use `data.files` with
an explicit list: the relevant year is selected from the combined files and
CF/cftime calendars are preserved. See `example_cesm2.yml`.
The coordinate variables and dimensions are configurable for rectilinear
latitude/longitude model output:

```yaml
data:
  coordinates: {time: valid_time, lat: latitude, lon: longitude}
  dimensions: {time: valid_time, lat: latitude, lon: longitude}
  periodic_longitude: auto       # default; inferred from global longitude coverage
```

With `auto` (the default), periodicity is enabled only for grids spanning
approximately 0–360° or −180–180°; it is off for a limited-area domain.
Set it explicitly to `true` or `false` only to override that inference.
Components then do not join across the left and right edges of a regional
grid. The detector works in either hemisphere; choose the
latitude interval in `data.region` (or omit it for the full globe).

At present `data.grid: latlon` is deliberately required: the centreline and
cell-area calculations require one-dimensional geographic latitude/longitude.
MAR-style projected x/y grids need a separate adapter based on their map
projection or 2-D geographic auxiliary coordinates. That adapter should be
validated against a MAR reference catalogue before it is advertised as a
supported input, rather than silently treating kilometres as degrees.
When children are enabled, each annual children job warms its continuity state
by replaying the preceding annual parent artifact; it does not wait for the
preceding children job.

Use the supplied `example_era5.yml` as a template. The only optional runtime
dependency is `PyYAML` for YAML; JSON works with the standard library.

`example_merra2.yml` is the direct-IVT variant and `example_cesm2.yml` shows
multi-year model files. The `compatibility_profile` setting is only for
reproducing a historical catalogue; omit it for a new scientific run.

Before allocating compute time, check only the configuration (no data are
opened):

```bash
python dark/dark.py my_run.yml --validate-config
```

## Parallel DARK, continuous children

For any scheduler, use the two portable stages below rather than embedding
scientific logic in a submission script:

```bash
# Run once, then make every dark-year job depend on this completed file.
python dark/pipeline.py build-percentiles run.yml

# Can run immediately for every year, in parallel.
python dark/pipeline.py dark-year run.yml 2001

# Wait only for DARK 2001 and DARK 2000; it replays 2000 to initialize
# temporal continuity, then writes children for 2001.
python dark/pipeline.py children-year run.yml 2001 \
  --previous-parent OUTPUT/.dark-work/2000.nc
```

`dark-year` writes `OUTPUT/.dark-work/YEAR.nc`, containing the parent mask and
the filtered IVT exceedance. Each `children-year` writes its final annual
catalog independently as soon as its current and previous artifacts exist.
The first children year has no `--previous-parent` and starts independently.
Parent artifacts remain until
the complete workflow is finished, because they can be read by the next year's
children job. This contract works unchanged for local jobs, SGE, Slurm, PBS,
or a workflow tool: only their dependency syntax differs.

## One-command launch on common systems

`submit.py` generates the small per-stage scripts and wires the appropriate
dependencies for you. It never embeds a cluster path or module name: place
machine-specific details in the optional `launcher` block of the config.

```bash
# Always inspect the generated dependency graph first.
python dark/submit.py my_run.yml --scheduler sge --dry-run

# Then submit it, or use slurm / pbs / local instead.
python dark/submit.py my_run.yml --scheduler sge
python dark/submit.py my_run.yml --scheduler slurm
python dark/submit.py my_run.yml --scheduler pbs
python dark/submit.py my_run.yml --scheduler local
```

The graph is deliberately the same everywhere: percentiles → all annual DARK
jobs in parallel → children year *Y* after DARK year *Y* and *Y − 1* (also in
parallel). Generated
scripts live under `OUTPUT/.dark-submit/`, while temporary parent/exceedance
files live under `OUTPUT/.dark-work/`.
