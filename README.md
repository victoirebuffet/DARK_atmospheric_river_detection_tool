# DARK Atmospheric River Catalogs (ERA5, 1979–2023)

This repository contains the code used to generate atmospheric river (AR) catalogs from the ERA5 reanalysis using the **DARK** detection algorithm and its **DARK + Children** tracking extension.

## Scripts

### `DARK_era5.py`
Generates the **DARK AR catalog** by:

- Computing a seasonally varying IVT threshold (98th percentile, 15-day rolling climatology)
- Detecting IVT exceedances
- Applying area filtering (> 200,000 km²)
- Applying length filtering (> 2,000 km)
- Producing binary AR masks

**Output:**

```text
ERA5.ar_tag.DARK.<freq>.<YYYYMMDD-YYYYMMDD>.nc
```

### `DARK_children_era5.py`
Generates the **DARK + Children catalog** by extending the DARK detections with temporal tracking.

The algorithm:

- Tracks AR objects through time using overlap between consecutive timesteps
- Identifies **parent ARs** (`ar_type = 1`)
- Identifies **child ARs** (`ar_type = 2`) that maintain strong IVT after Antarctic landfall
- Produces time-continuous AR classifications

**Output:**

```text
ERA5.ar_children_tag.DARK_children.<freq>.<YYYYMMDD-YYYYMMDD>.nc
```

## Input Data

ERA5 reanalysis fields:

- `viwve` — eastward integrated water vapor flux
- `viwvn` — northward integrated water vapor flux
- `lsm` — land-sea mask (used to identify Antarctic land/ice regions)

Temporal resolution: typically **6-hourly**.

## Output Variables

| Variable | Description |
|-----------|-------------|
| `ar_binary_tag` | 1 = atmospheric river, 0 = no AR |
| `ar_type` | 0 = none, 1 = parent, 2 = child (DARK + Children only) |

## Coverage

- **Spatial:** Global ERA5 grid (0.25° × 0.25°)
- **Temporal:** 1979–2023
- **Primary application:** Antarctic and polar atmospheric rivers

## Default Parameters

| Parameter | Value |
|-----------|-------|
| Percentile threshold | 98th |
| Rolling climatology window | 15 days |
| Minimum area | 200,000 km² |
| Minimum length | 2,000 km |
| Temporal resolution | 6-hourly |

## Requirements

- Python ≥ 3.11

Main packages:

```text
xarray
numpy
scipy
scikit-image
networkx
geopy
matplotlib
netCDF4
```

## Authors

- Victoire Buffet
- Vincent Favier
- Benjamin Pohl

Institut des Géosciences de l'Environnement (Grenoble, France)  
Biogéosciences (Dijon, France)

## Keywords

Atmospheric Rivers, ERA5, DARK, DARK_children, ARTMIP, IVT, Antarctica, Polar Meteorology, Reanalysis, Climate Diagnostics.
