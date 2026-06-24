
import netCDF4
import numpy as np
from csaps import csaps
from datetime import datetime, timezone
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error, r2_score



def datenum_to_datetime(arr):
    return np.array([datetime.fromordinal(int(dn) - 366).replace(tzinfo=timezone.utc) for dn in np.array(arr)])

def unix_to_datetime(arr):
    return np.array([datetime.fromtimestamp(int(ts), tz=timezone.utc) for ts in np.array(arr)])

def unix_to_datenum(arr):
    return np.array([datetime.utcfromtimestamp(int(ts)).toordinal() + 366 for ts in np.array(arr)])

def remove_nan(arr):
    arr = np.array(arr)
    return arr[~np.isnan(arr)]

# define year range for plotting 
def define_year_range(start, end, years):
                return(years.min() if start == 0 else start, 
                       years.max() if end == 9999 else end)


def bivariate_legend( ax, color_set):

    labels = ["0", "1", "2", "2+"]
    cell_size = 1

    for row in range(4):
        for col in range(4):
            idx = col * 4 + row
            ax.add_patch(
                Rectangle(
                    (col, row),          # x,y
                    cell_size,           # width
                    cell_size,           # height
                    facecolor=color_set[idx],
                    edgecolor="black"
                )
            )

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)

    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(labels)

    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(labels)

    ax.set_xlabel("# Troughs")
    ax.set_ylabel("# Peaks")

    ax.set_aspect("equal")

# find closest factor for nice plotting
def close_factors(number):
    ''' 
    find the closest pair of factors for a given number
    '''
    factor1 = 0
    factor2 = number
    while factor1 +1 <= factor2:
        factor1 += 1
        if number % factor1 == 0:
            factor2 = number // factor1
        
    return factor1, factor2

    
def to_frac_month(dates):
    result = []
    for d in dates:
        days_in_month = (pd.Timestamp(d.year, d.month % 12 + 1, 1) - pd.Timedelta(days=1)).day if d.month < 12 else 31
        result.append(d.month + (d.day - 1) / days_in_month)
    return np.array(result)


def calculate_spline(whole_timeframe, masked_values, masked_time, smoothing_parameter ):
    if len(masked_values) > 1:
        smooth_x = np.arange(whole_timeframe.min(), whole_timeframe.max() + 1, 1)
        smooth_y = csaps(masked_time, masked_values, smooth_x, smooth=smoothing_parameter)
        return smooth_x, smooth_y
    else:
        warnings.warn("not enough data to plot")





def plot_lake_outline(geometry, ax):
    if geometry.geom_type == "Polygon":
        x, y = geometry.exterior.xy
        ax.plot(x, y, color="black", linewidth=1, label="Lake Outline")
    elif geometry.geom_type == "MultiPolygon":
        for i, poly in enumerate(geometry.geoms):
            x, y = poly.exterior.xy
            ax.plot(x, y, color="black", linewidth=1, label="Lake Outline" if i == 0 else None)

def grab_metrics(e_path, metric_scores, buffered_geom_prep):
    with netCDF4.Dataset(e_path) as nc:
        summary = np.array(nc.variables["summary"][:, :])
        lats = nc.variables["lat"][:]
        lons = nc.variables["lon"][:]
        map_data = np.full(summary.shape, np.nan)
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]
        for (i, j), r2 in metric_scores.items():
            lon = lons[j]
            lat = lats[i]
            if buffered_geom_prep.contains(Point(lon, lat)):
                map_data[i, j] = r2
        return map_data, extent

def plot_map_data(colormap, map_data, extent, ax, cmap_extent=None):
    cmap = colormap if colormap else "RdYlBu"
    if cmap_extent is not None:
        im = ax.imshow(map_data, cmap=cmap, aspect="auto", origin="lower",
                       vmin=cmap_extent[0], vmax=cmap_extent[1], extent=extent)
    else:
        im = ax.imshow(map_data, cmap=cmap, aspect="auto", origin="lower", extent=extent)
    return im

def set_labels(ax, fig, im, title, colorbar_label, colorbar_ticks=None):
    ax.set_xlabel("Lon index")
    ax.set_ylabel("Lat index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, label=colorbar_label)
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
    ax.legend()

def grab_time_data(e_path, p_path, valid_coords, buffered_geom_prep,
                   var_x, var_y, year, use_max, DOY_start = 160, DOY_end = 250):
    with netCDF4.Dataset(e_path) as nc:
        summary = np.array(nc.variables["summary"][:, :])
        lats = nc.variables["lat"][:]
        lons = nc.variables["lon"][:]

    map_data = np.full(summary.shape, np.nan)
    neg_warned = False
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    with netCDF4.Dataset(p_path) as nc_p:
        x_full = nc_p.variables[var_x][:, :, :]
        y_full = nc_p.variables[var_y][:, :, :]

    for (i, j) in valid_coords:
        lon, lat = lons[j], lats[i]
        if not buffered_geom_prep.contains(Point(lon, lat)):
            continue
        x_arr = np.array(unix_to_datetime(remove_nan(x_full[i, j, :])))
        y_arr = np.array(remove_nan(y_full[i, j, :]))
        if len(x_arr) == 0:
            continue
        year_arr = np.array([d.year for d in x_arr])
        doys = np.array([d.timetuple().tm_yday for d in x_arr])
        mask = (year_arr == year) & (doys >= DOY_start) & (doys <= DOY_end)
        x_sub, y_sub = x_arr[mask], y_arr[mask]
        if len(x_sub) == 0:
            continue
        if (y_sub < 0).any() and not neg_warned:
            warnings.warn(f"Negative values in {year}", Warning)
            neg_warned = True
        if use_max:
            map_data[i, j] = float(x_sub[int(np.argmax(y_sub))].timetuple().tm_yday)
        else:
            map_data[i, j] = float(x_sub[0].timetuple().tm_yday)

    return map_data, extent


def grab_plotting_variables(start, end, pixel_data, variables = None):  
    """
    valid variables = ['pks', 'trgs',
                'midUP', 'midDOWN', 
                'onsetUP', 'onsetDOWN',
                'advUP', 'advDOWN']
    """
    if variables is None:
        variables = ["pks", "trgs", "midUP", "midDOWN"]
    result = {}
    for var in variables:
        var_x = f"{var}_x"
        var_y = f"{var}_y"
        
        if var_x not in pixel_data or var_y not in pixel_data:
            continue

        date_mask = np.array([(d.year <= end) & (d.year >= start) for d in pixel_data[var_x]])
        var_x_sub    = pixel_data[var_x][date_mask]
        var_y_sub    = pixel_data[var_y][date_mask]
        if var in ["pks","trgs"]:
            var_qa = str(var) + "_qa"
            if var_qa not in pixel_data:
                continue
            var_qa_sub  = pixel_data[var_qa][date_mask]

            result[var] = [var_x_sub, var_y_sub, var_qa_sub]
        else:
            result[var] = [var_x_sub, var_y_sub]
    return result


def prep_dimark_data(insitu_df,start=0, end=9999,  insitu_date_col="datetime", insitu_value_col="chlorophyll_a", insitu_station_col=None, station_id=None, max_depth = 5):
    insitu = insitu_df.copy()
    insitu[insitu_date_col] = pd.to_datetime(
            insitu[insitu_date_col], format = "mixed", dayfirst = True)

    # optional station filtering
    if (insitu_station_col is not None) and (station_id is not None):
            insitu = insitu[
            insitu[insitu_station_col] == station_id
            ]

    # time filtering
    insitu = insitu[
        (insitu[insitu_date_col].dt.year >= start) &
        (insitu[insitu_date_col].dt.year <= end)
    ]

    # remove invalid values
    insitu = insitu[np.isfinite(insitu[insitu_value_col])]
    insitu[insitu_date_col] = pd.to_datetime(insitu["datetime"]).dt.date
    insitu = insitu[insitu["depth"]< max_depth]
    insitu_mean = (insitu.groupby("datetime", as_index=False)[insitu_value_col].mean())
    return insitu_mean

def create_empty_heatmap(nrows=4, ncols=4):

    data = np.full((nrows, ncols), np.nan)
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(0, 3)
    ax.set_ylim(int(2001), int(2024))

    # Draw grid lines
    ax.set_xticks(np.arange(0, 5, 1), minor=True)
    ax.grid(which='minor')
    ax.set_yticks(np.arange(2001, 2024), minor = True)


    ax.set_xticks(np.arange(0.5, 4, 1))
    ax.set_yticks(np.arange(2001.5, 2024.5, 1))
    months = ["Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"]
    years = [str(yr) for yr in range(2002,2025)]
    ax.set_xticklabels(months, horizontalalignment = "center")
    ax.set_yticklabels(years, horizontalalignment = "center")

    # Hide tick marks
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0, pad = 20)

    return fig, ax, data

def bivariate_continuous_legend(ax, color_set, n=64):
    """Render a smooth 2D gradient legend for a continuous bivariate colormap.

    Bilinearly interpolates over the 4×4 color_set grid to produce a continuous
    gradient image. X-axis = peaks fraction, Y-axis = troughs fraction (both 0–1).
    """
    grid = np.zeros((n, n, 3))
    for row in range(n):
        for col in range(n):
            # normalize pixel coordinates to fractions in [0, 1]
            trg_frac = col / (n - 1)
            pk_frac = row / (n - 1)

            # bilinearly interpolate between the four colors
            grid[row, col] = interpolate_from_color_set(pk_frac, trg_frac, color_set=color_set)
    ax.imshow(grid, origin='lower', extent=[0, 1, 0, 1], aspect='equal')
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Troughs fraction")
    ax.set_ylabel("Peaks fraction")


def interpolate_from_color_set(pks_fraction, trgs_fraction, color_set):
    """Bilinearly interpolate an RGB color from a 4×4 color_set grid.

    Maps pk_frac and trg_frac (both in [0, 1]) onto the [0, 3] grid axes and
    interpolates between the four surrounding grid-point colors.
    """

    # map fractions to positions in the 4×4 color grid (indices 0–3)
    pk_pos = float(np.clip(pks_fraction * 3, 0, 3))
    trg_pos = float(np.clip(trgs_fraction * 3, 0, 3))

    # find the neighboring grid cell indices
    pk0 = int(np.floor(pk_pos))
    pk1 = min(pk0 + 1, 3)
    trg0 = int(np.floor(trg_pos))
    trg1 = min(trg0 + 1, 3)

    # compute interpolation weights within the grid cell
    t_pk = pk_pos - pk0
    t_trg = trg_pos - trg0
    c00 = np.array(mcolors.to_rgb(color_set[trg0 * 4 + pk0]))
    c10 = np.array(mcolors.to_rgb(color_set[trg0 * 4 + pk1]))
    c01 = np.array(mcolors.to_rgb(color_set[trg1 * 4 + pk0]))
    c11 = np.array(mcolors.to_rgb(color_set[trg1 * 4 + pk1]))
    return (c00 * (1 - t_pk) * (1 - t_trg) + c10 * t_pk * (1 - t_trg) +
           c01 * (1 - t_pk) * t_trg + c11 * t_pk * t_trg)



def calculate_metrics_to_plot(start, end, masked_values, masked_time, smoothing_parameter):
    limits = sorted(datenum_to_datetime(masked_time))
    if start == 0:
        function_start = min(limits).year
    else:
        function_start = start
    if end == 9999:
        function_end= max(limits).year
    else:
        function_end = end
    y_pred =csaps(masked_time, masked_values, masked_time, smooth=smoothing_parameter)
    y_true = masked_values

    time_slice = np.array(datenum_to_datetime(masked_time))

    mask_sub = np.array([(d.year <=function_end) & (d.year >= function_start) for d in time_slice])

    if mask_sub.sum()>2:
        rmse_sub = np.sqrt(mean_squared_error(y_true[mask_sub], y_pred[mask_sub]))
        r2_sub = r2_score(y_true[mask_sub], y_pred[mask_sub])
        mad_sub = np.median(np.abs(y_true[mask_sub]-y_pred[mask_sub]))

        rmse_tot = np.sqrt(mean_squared_error(y_true, y_pred))
        r2_tot = r2_score(y_true, y_pred)
        mad_tot = np.median(np.abs(y_true-y_pred))

        return {"rmse": [rmse_sub, rmse_tot],
                "r2": [r2_sub, r2_tot],
                "mad":[mad_sub, mad_tot]}, [function_start, function_end]
    else:
        warnings.warn("Not enough data to plot or compute metrics for chosen time interval")
        return None, [function_start, function_end]


def plot_variables(ax, plotting_data, spline_x, spline_y, time_frame, variables = None):

    if variables is None:
        variables = ["pks", "trgs", "midUP", "midDOWN"]
        
    neg_values_sub =[]
    neg_label_before = False
    start = time_frame[0]
    end = time_frame[1]
    
    ax.plot(datenum_to_datetime(spline_x), spline_y, color="black", linewidth=1, label="Spline")
    qa_colors = {0: "blue", 1: "orange", 2: "red"}
    qa_labels = {0: "Good", 1: "Fair", 2: "Poor"}
    for qa in (0, 1, 2):
        pm = plotting_data["pks"][2] == qa if "pks" in variables else None
        tm = plotting_data["trgs"][2] == qa if "trgs" in variables else None
        if pm is not None and pm.any():
            ax.scatter(plotting_data["pks"][0][pm], plotting_data["pks"][1][pm], color=qa_colors[qa], s=50,
                            marker="o", edgecolors="black", linewidths=0.5,
                            zorder=4, label=qa_labels[qa])
        if tm is not None and tm.any():
            ax.scatter(plotting_data["trgs"][0][tm], plotting_data["trgs"][1][tm], color=qa_colors[qa], s=50,
                            marker="o", edgecolors="black", linewidths=0.5,
                            zorder=4, label=qa_labels[qa] if (pm is not None and pm.any()) else None)
    if "pks" in variables:
        if (plotting_data["pks"][1] < 0).any():
            mask =  plotting_data["pks"][1]<0
            pks_x_neg_before = plotting_data["pks"][0][mask]
            pks_y_neg_before = plotting_data["pks"][0][mask]
            label = "Negative Value" if not neg_label_before else None
            ax.scatter(pks_x_neg_before, pks_y_neg_before, color="red", s=50, marker="x", zorder=6, label=label)
            neg_values_sub.append(len(pks_x_neg_before))
            neg_label_before = True
            warnings.warn(f"Negative Peak(s) in time period {start}-{end}", Warning)
    if "trgs" in variables:
        if (plotting_data["trgs"][1] < 0).any():
            mask =  plotting_data["trgs"][1]<0
            trgs_x_neg_before = plotting_data["trgs"][0][mask]
            trgs_y_neg_before = plotting_data["trgs"][1][mask]
            label = "Negative Value" if not neg_label_before else None
            ax.scatter(trgs_x_neg_before, trgs_y_neg_before, color="red", s=50, marker="x", zorder=6, label=label)
            neg_values_sub.append(len(trgs_x_neg_before))
            neg_label_before = True
            warnings.warn(f"Negative Troughs(s) in time period {start}-{end}", Warning)

    if "midUP" in variables:
        ax.scatter(plotting_data["midUP"][0], plotting_data["midUP"][1], color="mediumseagreen", s=30, marker="^", zorder=4, label="Mid Up")
        if (plotting_data["midUP"][1] < 0).any():
            mask =  plotting_data["midUP"][1]<0
            midUP_x_neg_before = plotting_data["midUP"][0][mask]
            midUP_y_neg_before = plotting_data["midUP"][1][mask]
            label = "Negative Value" if not neg_label_before else None
            ax.scatter(midUP_x_neg_before, midUP_y_neg_before, color="red", s=50, marker="x", zorder=6, label=label)
            neg_values_sub.append(len(midUP_x_neg_before))
            neg_label_before = True
            warnings.warn(f"Negative Mid Up(s) in time period {start}-{end}", Warning)

    if "midDOWN" in variables:
        ax.scatter(plotting_data["midDOWN"][0], plotting_data["midDOWN"][1], color="darkgreen", s=30, marker="v", zorder=4, label="Mid Down")
        if (plotting_data["midDOWN"][1] < 0).any():
            mask =  plotting_data["midDOWN"][1]<0
            midDOWN_x_neg_before = plotting_data["midDOWN"][0][mask]
            midDOWN_y_neg_before = plotting_data["midDOWN"][1][mask]
            label = "Negative Value" if not neg_label_before else None
            ax.scatter(midDOWN_x_neg_before, midDOWN_y_neg_before, color="red", s=50, marker="x", zorder=6, label=label)
            neg_values_sub.append(len(midDOWN_x_neg_before))
            neg_label_before = True
            warnings.warn(f"Negative Mid Down(s) in time period {start}-{end}", Warning)
    return neg_values_sub



def lakeID_to_name(metadata_df, id:int):
    """
    Look up the lake name corresponding to a given lake ID.

    Parameters
    ----------
    metadata_df : pandas.DataFrame
        DataFrame containing lake metadata, including at least
        columns "id" and "name".
    id : int
        Numeric lake identifier to search for.

    Returns
    -------
    str
        Name of the lake corresponding to the given ID.

    Raises
    ------
    IndexError
        If the provided ID is not found in the DataFrame.

    """
    matches = metadata_df.loc[metadata_df["id"] == id, "name"]
    if matches.empty:
        raise ValueError(f"Lake ID {id} not found")
    return matches.iloc[0]

