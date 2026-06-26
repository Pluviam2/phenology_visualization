# Phenology Visualization

Tools for visualizing satellite-derived phytoplankton phenology in lakes, including chlorophyll-a (chl-a) and phycocyanin (PC) time series from NetCDF datasets.

## Structure

```
scripts/
  visualization.py        # PhenologyVisualization class — main plotting interface
  functions_plotting.py   # Helper functions (spline fitting, metrics, map utilities)
notebooks/
  cubic_spline_timeseries.ipynb   # Time series plots with cubic spline fits
  comparison_plots.ipynb          # Cross-dataset and cross-pixel comparisons
  interactive_maps.ipynb          # Interactive lake maps
```

## Usage

```python
from visualization import PhenologyVisualization

PhenologyVisualization.set_shapefile_path("path/to/lakescci_shapefile.shp")

chla = PhenologyVisualization(extract_path, phenology_path)

# Single pixel time series with cubic spline
chla.single_plot(lat_idx, lon_idx, ax, start=2003, end=2022)

# Split plot (e.g. pre/post gap)
chla.split_plot(ax0, ax1, lat_idx, lon_idx, end0=2012, start1=2016)

# Year-by-year panel
chla.single_years_plot(lat_idx, lon_idx, years=[...], ncol=5, nrow=5)

# All years overlaid
chla.yearly_cubic_spline(ax, lat_idx, lon_idx)
```

## Dependencies

`numpy`, `pandas`, `matplotlib`, `scipy`, `csaps`, `netCDF4`, `scikit-learn`, `geopandas`, `shapely`, `colorcet`, `pyproj`

## Data

Expects NetCDF files organized as:
```
data/
  v3.1/
    phenology/{variable}/{lake_id}.nc
    extract/{variable}/{lake_id}.nc
```
