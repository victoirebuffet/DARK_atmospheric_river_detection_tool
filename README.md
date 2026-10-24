# DARK atmospheric-river detection tool

Portable, configuration-driven implementation of DARK and the optional
**DARK + children** classification for reanalyses and climate models.

The maintained implementation is in [`dark/`](dark/README.md). It supports:

- direct scalar IVT or vector components;
- configurable variables, coordinate/dimension names, region and percentile;
- global and limited-area rectilinear latitude/longitude grids;
- ERA5, MERRA2, CESM-style multi-year files, and other compatible inputs;
- standalone DARK or DARK + children;
- local, SGE, Slurm and PBS execution;
- configurable percentile reference periods and an optional adaptive,
  smoothed annual driver threshold.

Start with one of the configurations in `dark/example_*.yml`, validate it,
then use `dark/submit.py` to inspect or submit the workflow.

The legacy `DARK_era5.py` and `DARK_children_era5.py` files are retained only
as historical reference implementations.
