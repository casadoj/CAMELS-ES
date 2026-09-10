from pathlib import Path
from typing import Optional, Union
import logging
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


from ocab.plots.utils import compute_annual_timeseries, compute_monthly_climatology, compute_climatology, define_y_limits
from ocab.signatures import baseflow_index, flashiness_index, slope_fdc, budyko
import ocab.meteorology as METEO

logger = logging.getLogger(__name__)

def plot_stations_map(
    geometry,
    area: Optional[pd.Series] = None,
    save: Union[str, Path] = None,
    **kwargs
):
    """Creates a map where stations are represented as dots. The size of the dots reflects the catchment area (if provided)
    
    Parameters:
    -----------
    geometry: gpd.GeoSeries
        Geometry of the points
    area: pandas.Series (optional)
        Reservoir catchment area (km2)
    save: str or pathlib.Path (optional)
        If provided, file where the plot will be saved
    """
    
    figsize = kwargs.get('figsize', (20, 5))
    title = kwargs.get('title', None)
    alpha = kwargs.get('alpha', .7)
    size = kwargs.get('size', 12)
    color = kwargs.get('color', 'steelblue')
    marker = kwargs.get('marker', 'o')

    # set up plot
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=figsize, 
        subplot_kw={'projection': proj}
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='wheat'),#'lightgray'),
        alpha=.5,
        zorder=0
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m', edgecolor='lightslategrey', facecolor='none', linewidth=.5),
        alpha=0.7,
        zorder=1
    )
    if 'extent' in kwargs:
        ax.set_extent(kwargs['extent'], crs=proj)
    if title is not None:
        ax.text(.5, 1.125, title, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize=12)
    ax.axis('off')
    
    # plot poitns
    s = np.cbrt(area) if area is not None else size
    scatter = ax.scatter(
        geometry.x,
        geometry.y,
        s=s,
        c=color,
        marker=marker,
        alpha=alpha,
        zorder=2
    )  
    
    # legend
    # if area is not None:
    #     legend = ax.legend(
    #         *scatter.legend_elements(prop='sizes', num=4, alpha=alpha),
    #         title='catchment (km²)',
    #         # bbox_to_anchor=(1.2, 0.5),
    #         bbox_to_anchor=[1.025, .6, .06, .25],
    #         loc='center left',
    #         frameon=False
    #         )
    #     ax.add_artist(legend)    

    # save
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')


def plot_station_timeseries(
    ts: pd.DataFrame,
    # area: float,
    attrs: pd.Series,
    regime: Optional[str] = None,
    save: bool = False,
    **kwargs
    ):
    """
    Creates an interactive figure with four plots: daily and annual timeseries, the flow 
    duration curve, and the climograph.
    Interactive buttons allow to transform the specific discharge and precipitation data to 
    logarithmic or square root scales to identify issues in the low flows.
    
    Parameters:
    -----------
    ts: pandas.Series
        Discharge time series
    area: float
        Catchment area (km²)
    regime: string (optional)
        Flow regime (if provided by the Ministry)
    save: boolean
        If True, return an object with the Plotly figure. If false, show the figure
    
    Keyword arguments:
    ------------------
    alpha: float
        Transparency of the precipitation bar plots
    c_obs: string
        Color used to plot observed discharge
    c_pet: string
        Color used to plot potential evapotranspiration
    c_precip: string
        Color used to plot precipitation
    c_sim: string
        Color used to plot simulated discharge
    c_temp: string
        Color used to plot temperature
    title: string
        Figure title
    """

    # Extract keywords
    alpha = kwargs.get('alpha', 0.2)
    c_obs = kwargs.get('c_obs', 'darkslategrey')
    c_sim = kwargs.get('c_sim', 'darkorange')
    c_precip = kwargs.get('c_precip', 'lightblue')
    c_temp = kwargs.get('c_temp', 'darkred')
    c_pet = kwargs.get('c_pet', 'darkseagreen') #'olivedrab')
    title = kwargs.get('title', None)

    # detect available meteo datasets
    available_datasets = {
        name: suffix
        for name, suffix in METEO.DATASETS.items()
        if any(f'_{suffix}' in col for col in ts.columns)
    }
    if not available_datasets:
        available_datasets = {'Default': ''}

    # Default meteo dataset
    default_dataset = 'ROCIO-IBEB'
    default_suffix = available_datasets[default_dataset]

    # is the station simulated?
    has_sim = 'discharge_mm_sim' in ts.columns
    name_dis_obs = 'Discharge (obs)' if has_sim else 'Discharge'

    # Setup Grid
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"colspan": 2}, None], # row 1
            [{"colspan": 2}, None], # row 2
            [{"colspan": 2, "secondary_y": True}, None], # row 3
            [{"secondary_y": True}, {}] # row 4
            ],
        column_widths=[0.5, 0.5], 
        row_heights=[0.48, 0.02, 0.1, 0.4],
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=("Daily Time Series", "", "Annual Time Series", 'Climatology', "Budyko Diagram"),
        vertical_spacing=0.05,
        horizontal_spacing=0.15
    )

    # --- PLOT 1: DAILY TIME SERIES ---

    row, col = 1, 1

    # precipitation
    fig.add_trace(
        go.Bar(
            x=ts.index, 
            y=ts[f'precip_mm_{default_suffix}'], 
            name="Precipitation", 
            marker_color=c_precip, 
            opacity=1 - alpha,
            marker_line_width=0,
            visible=True,
            legendgroup="P",
            showlegend=False,
        ),
        row=row, col=col
    )

    # observed discharge
    fig.add_trace(
        go.Scatter(
            x=ts.index, 
            y=ts['discharge_mm'], 
            name=name_dis_obs, 
            line=dict(color=c_obs, width=1), 
            visible=True,
            legendgroup="Q",
            showlegend=False
        ),
        row=row, col=col
    )
    
    # simulated discharge
    if has_sim:
        fig.add_trace(
            go.Scatter(
                x=ts.index, 
                y=ts['discharge_mm_sim'], 
                name="Discharge (sim)", 
                line=dict(color=c_sim, width=1, dash='dot'), 
                visible='legendonly',
                legendgroup="Q_sim",
                showlegend=False
            ),
            row=row, col=col
        )
        
    # set axes properties
    fig.update_yaxes(
        title_text="mm", 
        rangemode='nonnegative',
        row=row, col=col, secondary_y=False
    )

    # --- PLOT 2: MISSING DATA ---

    row, col = 2, 1

    # find missing discharge dates
    missing_x = ts[ts['discharge_cms'].isnull()].index
    missing_y = [0] * len(missing_x)
    fig.add_trace(
        go.Scatter(
            x=missing_x,
            y=missing_y, 
            mode='markers',
            marker=dict(
                symbol='line-ns-open',
                size=10,
                color=c_obs,
                line_width=1
            ),
            legendgroup="Q",
            showlegend=False,
            hoverinfo='x'
        ),
        row=row, col=col
    )

    # set axes
    fig.update_yaxes(
        showgrid=False, 
        showticklabels=False, 
        zeroline=False, 
        fixedrange=True,
        row=row, col=col
    )
    fig.update_xaxes(
        showgrid=False, 
        showticklabels=False, 
        row=row, col=col,
        matches='x'
    )

    # --- PLOT 3: ANNUAL TIME SERIES ---

    row, col = 3, 1

    # compute annual time series
    ts_y = compute_annual_timeseries(ts, rule='YS-OCT')
    ts_y.index += pd.DateOffset(months=6) # to adapt labels in the plots

    # meteo traces
    fig.add_trace(
        go.Bar(
            x=ts_y.index, 
            y=ts_y[f'precip_mm_{default_suffix}'], 
            name="Precipitation", 
            marker_color=c_precip, 
            opacity=1 - alpha,
            visible=True,
            showlegend=True,
            legendgroup="P",
            hovertemplate="<b>Year: %{x|%Y}</b><br>P: %{y:.0f} mm<extra></extra>"
        ),
        row=row, col=col
    )
    fig.add_trace(go.Scatter(
            x=ts_y.index, 
            y=ts_y[f'temp_degC_{default_suffix}'], 
            name="Temperature",
            line=dict(color=c_temp, width=2),
            visible='legendonly',
            showlegend=True,
            legendgroup="T",
            hovertemplate="<b>Year: %{x|%Y}</b><br>T: %{y:.1f} °C<extra></extra>"
        ), 
        row=row, col=col, secondary_y=True
    )
    fig.add_trace(go.Scatter(
            x=ts_y.index, 
            y=ts_y[f'pet_mm_{default_suffix}'], 
            name="PET",
            line=dict(color=c_pet, width=2),
            visible='legendonly',
            showlegend=True,
            legendgroup="PET",
            hovertemplate="<b>Year: %{x|%Y}</b><br>PET: %{y:.0f} mm<extra></extra>"
        ), 
        row=row, col=col,
    )

    # discharge traces
    fig.add_trace(
        go.Scatter(
            x=ts_y.index, 
            y=ts_y['discharge_mm'], 
            name=name_dis_obs,
            line=dict(color=c_obs, width=2),
            visible=True,
            showlegend=True,
            legendgroup="Q",
            hovertemplate="<b>Year: %{x|%Y}</b><br>Q: %{y:.0f} mm<extra></extra>"
        ),
        row=row, col=col
    )
    if has_sim:
        fig.add_trace(
            go.Scatter(
                x=ts_y.index, 
                y=ts_y['discharge_mm_sim'], 
                name="Discharge (sim)",
                line=dict(color=c_sim, width=1.8, dash='dot'), 
                visible='legendonly',
                showlegend=True,
                legendgroup="Q_sim",
                hovertemplate="<b>Year: %{x|%Y}</b><br>Q: %{y:.0f} mm<extra></extra>"
            ),
            row=row, col=col
        )

    # set axes properties
    scale_y = 100
    round_mm_y = 500
    mm_range_y, deg_range_y = define_y_limits(ts_y, round_mm_y, scale_y)
    fig.update_yaxes(
        title_text="mm", 
        range=mm_range_y, 
        dtick=round_mm_y, 
        row=row, col=col
    )
    fig.update_yaxes(
        title_text="°C", 
        range=deg_range_y, 
        dtick=round_mm_y / scale_y, 
        row=row, col=col, secondary_y=True
    )
    fig.update_xaxes(
        showticklabels=False, 
        row=row, col=col,
        matches='x'
    )

    # --- PLOT 4: CLIMOGRAPH ---

    row, col = 4, 1

    # compute monthly climatology
    ts_m = compute_monthly_climatology(ts)

    # meteo traces
    fig.add_trace(
        go.Bar(
            x=ts_m.index, 
            y=ts_m[f'precip_mm_{default_suffix}'], 
            name="Precipitation", 
            marker_color=c_precip, 
            opacity=1 - alpha,
            visible=True,
            showlegend=False,
            legendgroup="P",
            text=[calendar.month_abbr[i] for i in ts_m.index],
            textposition='none',
            hovertemplate="<b>Month: %{text}</b><br>P: %{y:.0f} mm<extra></extra>"
        ), 
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts_m.index, 
            y=ts_m[f'temp_degC_{default_suffix}'], 
            name="Temperature", 
            line=dict(color=c_temp, width=2),
            visible='legendonly',
            showlegend=False,
            legendgroup="T",
            text=[calendar.month_abbr[i] for i in ts_m.index],
            hovertemplate="<b>Month: %{text}</b><br>Tmean: %{y:.1f} °C<extra></extra>"
        ),
        row=row, col=col, secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=ts_m.index, 
            y=ts_m[f'pet_mm_{default_suffix}'], 
            name="PET", 
            line=dict(color=c_pet, width=2),
            visible='legendonly',
            showlegend=False,
            legendgroup="PET",
            text=[calendar.month_abbr[i] for i in ts_m.index],
            hovertemplate="<b>Month: %{text}</b><br>PET: %{y:.0f} mm<extra></extra>"
        ),
        row=row, col=col
    )

    # discharge traces
    fig.add_trace(
        go.Scatter(
            x=ts_m.index, 
            y=ts_m['discharge_mm'], 
            name=name_dis_obs, 
            line=dict(color=c_obs, width=2),
            visible=True,
            showlegend=False,
            legendgroup="Q",
            text=[calendar.month_abbr[i] for i in ts_m.index],
            hovertemplate="<b>Month: %{text}</b><br>Q: %{y:.0f} mm<extra></extra>"
        ), 
        row=row, col=col
    )
    if has_sim:
        fig.add_trace(
            go.Scatter(
                x=ts_m.index, 
                y=ts_m['discharge_mm_sim'], 
                name="Discharge (sim)", 
                line=dict(color=c_sim, width=1.8, dash='dot'), 
                visible='legendonly',
                legendgroup="Q_sim",
                showlegend=False
            ), 
            row=row, col=col
        )

    # set axes properties
    scale_m = 2
    round_mm_m = 20
    mm_range_m, deg_range_m = define_y_limits(ts_m, round_mm_m, scale_m)
    fig.update_yaxes(
        title_text="mm", 
        range=mm_range_m, 
        dtick=round_mm_m, 
        row=row, col=col
    )
    fig.update_yaxes(
        title_text="°C", 
        range=deg_range_m, 
        dtick=round_mm_m / scale_m, 
        row=row, col=col, secondary_y=True
    )
    fig.update_xaxes(
        tickvals=list(range(1, 13)), 
        ticktext=['J','F','M','A','M','J','J','A','S','O','N','D'], 
        row=row, col=col
    )

    # # --- PLOT 5: BUDYKO DIAGRAM ---

    row, col = 4, 2

    # Compute indices
    pet_y = ts_y[f'pet_mm_{default_suffix}']
    precip_y = ts_y[f'precip_mm_{default_suffix}']
    aridity = pet_y / precip_y 
    evaporativity = (precip_y - ts_y['discharge_mm']) / precip_y

    # plot limits
    round = 0.5
    xlim = [0, max(1.01, np.ceil(aridity.max() / round) * round)]
    ylim = [min(0, np.floor(aridity.min() / round) * round), 1.0]

    # Theoretical lines
    line_props = dict(color='grey', width=1)
    # Water Limit
    fig.add_trace(
        go.Scatter(
            x=np.linspace(*xlim, 100), 
            y=[1] * 100, 
            mode='lines', 
            name='Water Limit',
            line=line_props, 
            showlegend=False,
            hovertemplate=" "
        ),
        row=row, col=col
    )
    # Energy Limit
    fig.add_trace(
        go.Scatter(
            x=np.linspace(*xlim, 100), 
            y=np.linspace(*xlim, 100), 
            mode='lines', 
            name='Energy Limit',
            line=line_props, 
            showlegend=False,
            hovertemplate=" "
        ),
        row=row, col=col
    )
    # Budyko Curve
    aridity_idx = np.linspace(0.001, xlim[1], 100)
    fig.add_trace(
        go.Scatter(
            x=aridity_idx, 
            y=budyko(aridity_idx, n=2), 
            mode='lines', 
            name='Budyko Limit',
            line=line_props,
            showlegend=False,
            hovertemplate=" "
        ),
        row=row, col=col
    )

    # Observed discharge
    fig.add_trace(
        go.Scatter(
            x=aridity, 
            y=evaporativity,
            mode='markers',
            name='Budyko',
            marker=dict(color=c_obs, size=8, line=dict(width=1, color='white')),
            customdata=aridity.index.year,
            hovertemplate="<b>Year: %{customdata}</b><br>Arid. Index: %{x:.2f}<br>Evap. Index: %{y:.2f}<extra></extra>",
            legendgroup="Q",
            showlegend=False
        ),
        row=row, col=col
    )

    # Simulated discharge
    if has_sim:
        evaporativity_sim = (precip_y - ts_y['discharge_mm_sim']) / precip_y
        fig.add_trace(
            go.Scatter(
                x=aridity, 
                y=evaporativity_sim,
                mode='markers',
                name='Budyko',
                marker=dict(color=c_sim, size=8, line=dict(width=1, color='white')),
                customdata=aridity.index.year,
                hovertemplate="<b>Year: %{customdata}</b><br>Arid. Index: %{x:.2f}<br>Evap. Index: %{y:.2f}<extra></extra>",
                visible='legendonly',
                legendgroup="Q_sim",
                showlegend=False
            ),
            row=row, col=col
        )

    # set axes
    fig.update_xaxes(
        title_text="Aridity Index: PET/P", 
        range=xlim,
        row=row, col=col)
    fig.update_yaxes(
        title_text="Evaporative Index: (P-Q)/P", 
        range=ylim,
        row=row, col=col
    )

    ### --- SIGNATURES & CLIMATOLOGY ---

    def get_signature_text(climatology: pd.Series, attrs: pd.Series, suffix: str) -> str:
        text = (
            "<b>Catchment Properties</b><br>"
            f"Area: {attrs['catch_skm']:.0f} km²<br>"
            f"Regime: {regime}<br>"
            "<br><b>Climatology</b><br>"
            f"Discharge: {climatology['discharge_mm']:.0f} mm/year<br>"
            f"Precipitation: {climatology[f'precip_mm_{suffix}']:.0f} mm/year<br>"
            f"PET: {climatology[f'pet_mm_{suffix}']:.0f} mm/year<br>"
            f"Temperature: {climatology[f'temp_degC_{suffix}']:.1f} °C<br>"
            "<br><b>Hydrological Signatures</b><br>"
            f"Baseflow Index: {bfi:.2f}<br>"
            f"Flashiness Index: {fi:.2f}<br>"
            f"Slope FDC: {slope:.2f}<br>"
        )
        if 'KGE' in attrs.index or 'NSE' in attrs.index:
            performance_text = "<br><b>Model performance</b><br>"
            if 'NSE' in attrs.index:
                performance_text += f"NSE: {attrs['NSE']:.2f}<br>"
            if 'KGE' in attrs.index:
                performance_text += f"KGE: {attrs['KGE']:.2f}<br>"
            if 'Beta-KGE' in attrs.index:
                performance_text += f"Bias: {attrs['Beta-KGE']:.2f}<br>"
            if 'Alpha-NSE' in attrs.index:
                performance_text += f"Variability: {attrs['Alpha-NSE']:.2f}<br>"
            if 'Pearson-r' in attrs.index:
                performance_text += f"Correlation: {attrs['Pearson-r']:.2f}<br>"
            text += performance_text
        return text

    # Compute hydrological signatures
    _, bfi = baseflow_index(ts['discharge_cms'])
    fi = flashiness_index(ts['discharge_cms'])
    _, slope = slope_fdc(ts['discharge_mm'])

    # Calculate climatology values
    climatology = compute_climatology(ts)

    # add signature annotation
    signature_text = get_signature_text(climatology, attrs, default_suffix)
    fig.add_annotation(
        name='signature_annotation',
        text=signature_text, 
        x=1, xanchor="left", xref="paper", 
        y=0.5, yanchor="top", yref="paper", 
        showarrow=False, 
        align="left", 
        bgcolor="rgba(0, 0, 0, 0)"
    )

    # locate the index of the signature annotation
    sig_annot_idx = next(
        i for i, annot in enumerate(fig.layout.annotations)
        if getattr(annot, 'name', None) == 'signature_annotation'
    )

    # --- BUTTONS SETUP ---

    # ------------------------------------------------------------------
    # Meteorology buttons
    # ------------------------------------------------------------------

    # Map target trace indices to restyle on meteo change
    base_annual_idx = 4 if has_sim else 3
    base_climo_idx = 9 if has_sim else 7
    base_budyko_idx = 17 if has_sim else 14
    meteo_target_indices = [
        0,
        base_annual_idx, base_annual_idx + 1, base_annual_idx + 2,
        base_climo_idx, base_climo_idx + 1, base_climo_idx + 2,
        base_budyko_idx
    ]
    if has_sim:
        meteo_target_indices += [base_budyko_idx + 1]

    meteo_buttons = []
    for label, suffix in available_datasets.items():
        # define time series
        daily_p = ts[f'precip_mm_{suffix}']
        annual_p = ts_y[f'precip_mm_{suffix}']
        annual_t = ts_y[f'temp_degC_{suffix}']
        annual_pet = ts_y[f'pet_mm_{suffix}']
        climo_p = ts_m[f'precip_mm_{suffix}']
        climo_t = ts_m[f'temp_degC_{suffix}']
        climo_pet = ts_m[f'pet_mm_{suffix}']
        budyko_arid = ts_y[f'pet_mm_{suffix}'] / ts_y[f'precip_mm_{suffix}'] 
        budyko_evap = (ts_y[f'precip_mm_{suffix}'] - ts_y['discharge_mm']) / ts_y[f'precip_mm_{suffix}']

        x = [ts.index, ts_y.index, ts_y.index, ts_y.index, ts_m.index, ts_m.index, ts_m.index, budyko_arid]
        y = [daily_p, annual_p, annual_t, annual_pet, climo_p, climo_t, climo_pet, budyko_evap]

        if has_sim:
            budyko_evap_sim = (ts_y[f'precip_mm_{suffix}'] - ts_y['discharge_mm_sim']) / ts_y[f'precip_mm_{suffix}']
            x += [budyko_arid]
            y += [budyko_evap_sim]

        # signature text
        current_signature_text = get_signature_text(climatology, attrs, suffix)

        meteo_buttons.append(
            dict(
                label=label,
                method="update",
                args=[{
                        "x": x,
                        "y": y
                    },
                    {
                        f"annotations[{sig_annot_idx}].text": current_signature_text
                    },
                    meteo_target_indices
                ]
            )
        )

    # ------------------------------------------------------------------
    # Scale buttons
    # ------------------------------------------------------------------

    # Update Layout & Scale Buttons
    y_raw = [ts[f'precip_mm_{default_suffix}'], ts['discharge_mm']]
    scale_target_indices = [0, 1]
    if has_sim:
        y_raw += [ts['discharge_mm_sim']]
        scale_target_indices += [2]
    y_sqrt = [trace**0.5 for trace in y_raw]

    scale_buttons = [
        dict(
            label="Linear", 
            method="update", 
            args=[
                {"y": y_raw},
                {"yaxis.type": "linear", "yaxis.title.text": "mm"},
                scale_target_indices
            ]
        ),
        dict(
            label="Square root", 
            method="update",
            args=[
                {"y": y_sqrt},
                {"yaxis.type": "linear", "yaxis.title.text": "sqrt mm"},
                scale_target_indices
            ]
        ),
        dict(
            label="Logarithmic", 
            method="update",
            args=[
                {"y": y_raw},
                {"yaxis.type": "log", "yaxis.title.text": "log mm"},
                scale_target_indices
            ]
        ),
    ]

    fig.update_layout(
        meta=dict(
            initial_scale="linear",
            initial_meteo=default_dataset
        )
    )

    # add titles and buttons
    fig.update_layout(
        title_text=f"<b>{title if title else ''}</b>",
        title_x=0.01,
        title_xanchor='left',
        title_y=0.99,
        title_yanchor='top',
        margin=dict(l=50, r=200, t=50, b=50),
        template="plotly_white",
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.075,
            xanchor="center",
            x=0.45
        ),
        bargap=0.08, 
        barmode='overlay',
        updatemenus=[
            # meteo dataset selection
            dict(
                type="buttons",
                direction="down",
                active=0,
                x=1, xanchor="left",
                y=0.975, yanchor="top",
                showactive=True,
                buttons=meteo_buttons,
            ),
            # axis scale selection
            dict(
                type="buttons", 
                direction="down", 
                active=0, 
                x=1, xanchor="left", 
                y=0.8, yanchor='top', 
                showactive=True,
                buttons=scale_buttons,
            )
        ]
    )

    # add titles to the buttons
    fig.add_annotation(
        text="<b>Meteorology</b>",
        x=1, xanchor="left", xref="paper",
        y=0.975, yanchor="bottom", yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        text="<b>Y-axis Scale</b>",
        x=1, xanchor="left", xref="paper",
        y=0.8, yanchor="bottom", yref="paper",
        showarrow=False,
    )

    fig.update_xaxes(
        range=[ts.index.min(), ts.index.max()],
        row=1, col=1
    )

    if save:
        return fig
    else:
        fig.show()


# def create_station_html(
#     fig, 
#     path: str,
#     start: str,
#     end: str
# ):
#     """Wraps a Plotly figure into the full validation HTML template.
    
#     Parameters:
#     -----------
#     fig:
#         Result of `plot_reservoir_timeseries()`
#     path: string
#         Name of the HTML file where the figure wil be saved
#     start: string
#         Start date of the time series. Format YYYY-mm-dd
#     end: string
#         End date of the time series. Format YYYY-mm-dd
#     """
        
#     fig.update_layout(height=None, width=None, autosize=True)

#     # Convert figure to HTML div string
#     plotly_html = pio.to_html(
#         fig, 
#         full_html=False, 
#         include_plotlyjs='cdn',
#         include_mathjax=False,
#         config={'responsive': True, 'displaylogo': False}
#     )

#     title = Path(path).stem

#     full_page_html = f"""
#         <html>
#         <head>
#             <!-- Google tag (gtag.js) -->
#             <script async src="https://www.googletagmanager.com/gtag/js?id=G-E565MB2DE7"></script>
#             <script>
#                 window.dataLayer = window.dataLayer || [];
#                 function gtag(){{dataLayer.push(arguments);}}
#                 gtag('js', new Date());
#                 gtag('config', 'G-E565MB2DE7');
#             </script>

#             <meta charset="utf-8" />
#             <title>camelses_{title}</title>
#             <style>
#                 body {{ 
#                     margin: 0; padding: 0; height: 100vh; width: 100vw;
#                     display: flex; 
#                     flex-direction: row; 
#                     font-family: sans-serif;
#                     overflow: hidden;
#                 }}
#                 .hydrograph {{ 
#                     flex: 7; 
#                     height: 100vh; 
#                     min-width: 0;
#                     position: relative; /* Added to anchor the button */
#                 }}
#                 .google-form {{ 
#                     flex: 3; 
#                     height: 100vh; 
#                     border-left: 1px solid #ddd;
#                     display: flex;
#                 }}
#                 .hydrograph > .plotly-graph-div {{ height: 100% !important; width: 100% !important; }}
                
#                 iframe {{ width: 100%; height: 100%; border: none; }}
                
#                 /* Updated: Positioned at bottom right of the plot panel */
#                 .back-nav {{ 
#                     position: absolute; 
#                     bottom: 20px; 
#                     right: 20px; 
#                     z-index: 9999; 
#                 }}
#                 .back-btn {{
#                     text-decoration: none; 
#                     color: white; 
#                     font-size: 14px; 
#                     font-weight: bold; 
#                     background-color: steelblue; /* Solid color looks better at bottom */
#                     padding: 10px 16px; 
#                     border-radius: 5px; 
#                     box-shadow: 0 2px 5px rgba(0,0,0,0.2);
#                     transition: background-color 0.3s;
#                 }}
#                 .back-btn:hover {{
#                     background-color: #2e5d86;
#                 }}
#             </style>
#         </head>
#         <body>
#             <div class="hydrograph">
#                 # <div class="back-nav">
#                 #     <a href="../../../index.html" class="back-btn">← Back to map</a>
#                 # </div>
#                 {plotly_html}
#             </div>
            
#             <div class="google-form" id="form-container"></div>

#             <script>
#                 (function() {{
#                     const baseUrl = "https://docs.google.com/forms/d/e/1FAIpQLSdMDms_IAwkhPwOVqLvU0oVghG9Xti1LMhTsPoujm2w2uXF9A/viewform?embedded=true";
#                     const emailEntryId = "2010975770";
#                     const stationEntryId = "1301492004";
#                     const startEntry1Id = "598747998";
#                     const endEntry1Id = "2001326573";
#                     const startEntry2Id = "1136531299";
#                     const endEntry2Id = "530564783";
#                     const startEntry3Id = "1646852203";
#                     const endEntry3Id = "2002260210";

#                     const userEmail = localStorage.getItem('userEmail') || "";
#                     const filename = window.location.pathname.split('/').pop();
#                     const stationId = filename.replace('.html', '');
#                     const start = "{start}";
#                     const end = "{end}";

#                     const finalUrl = baseUrl + 
#                         "&entry." + stationEntryId + "=" + stationId + 
#                         "&entry." + emailEntryId + "=" + encodeURIComponent(userEmail) + 
#                         "&entry." + startEntry1Id + "=" + start + 
#                         "&entry." + endEntry1Id + "=" + end +
#                         "&entry." + startEntry2Id + "=" + start + 
#                         "&entry." + endEntry2Id + "=" + end +
#                         "&entry." + startEntry3Id + "=" + start + 
#                         "&entry." + endEntry3Id + "=" + end;

#                     document.getElementById('form-container').innerHTML = 
#                         '<iframe src="' + finalUrl + '" frameborder="0">Loading form…</iframe>';
#                 }})();

#                 window.addEventListener('load', function() {{
#                     setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 200); 
#                 }});
#             </script>
#         </body>
#         </html>
#         """
    
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(full_page_html)


def create_station_html(
    fig, 
    path: str,
    start: str,
    end: str
):
    """Wraps a Plotly figure into the full validation HTML template."""
        
    fig.update_layout(height=None, width=None, autosize=True)

    plotly_html = pio.to_html(
        fig, 
        full_html=False, 
        include_plotlyjs='cdn',
        include_mathjax=False,
        config={'responsive': True, 'displaylogo': False}
    )

    title = Path(path).stem

    full_page_html = f"""
        <html>
        <head>
            <!-- Google tag (gtag.js) -->
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-E565MB2DE7"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());
                gtag('config', 'G-E565MB2DE7');
            </script>
            
            <meta charset="utf-8" />
            <title>camelses_{title}</title>
            <style>
                body {{ 
                    margin: 0; padding: 0; height: 100vh; width: 100vw;
                    display: flex; 
                    font-family: sans-serif;
                    overflow: hidden;
                }}
                
                /* Hydrograph panel expands to fill space */
                .hydrograph {{ 
                    flex: 1; 
                    height: 100vh; 
                    min-width: 0;
                    position: relative;
                    transition: flex 0.3s ease-in-out;
                }}
                
                /* Sidebar Drawer in document flow */
                .google-form-panel {{ 
                    width: 0;
                    height: 100vh; 
                    background: #fff;
                    border-left: 0px solid #ddd;
                    box-shadow: -4px 0 15px rgba(0,0,0,0.15);
                    z-index: 10000;
                    position: relative;
                    transition: width 0.3s ease-in-out;
                    display: flex;
                }}
                
                /* Open State applied via body class */
                body.form-open .google-form-panel {{
                    width: 450px;
                    border-left: 1px solid #ddd;
                }}

                /* Toggle Button placed 5% from top */
                .toggle-btn {{
                    position: absolute;
                    left: -42px;
                    top: 5%;
                    transform: translateY(0);
                    background-color: steelblue;
                    color: white;
                    border: none;
                    padding: 12px 10px;
                    cursor: pointer;
                    border-radius: 6px 0 0 6px;
                    box-shadow: -2px 2px 5px rgba(0,0,0,0.2);
                    font-weight: bold;
                    writing-mode: vertical-rl;
                    text-orientation: mixed;
                    font-size: 13px;
                    letter-spacing: 1px;
                    z-index: 10001;
                }}
                .toggle-btn:hover {{
                    background-color: #2e5d86;
                }}

                .hydrograph > .plotly-graph-div {{ height: 100% !important; width: 100% !important; }}
                
                iframe {{ width: 100%; height: 100%; border: none; }}
                
                .back-nav {{ 
                    position: absolute; 
                    bottom: 20px; 
                    right: 20px; 
                    z-index: 9999; 
                }}
                .back-btn {{
                    text-decoration: none; 
                    color: white; 
                    font-size: 14px; 
                    font-weight: bold; 
                    background-color: steelblue; 
                    padding: 10px 16px; 
                    border-radius: 5px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    transition: background-color 0.3s;
                }}
                .back-btn:hover {{
                    background-color: #2e5d86;
                }}
            </style>
        </head>
        <body>
            <div class="hydrograph">
                {plotly_html}
            </div>
            
            <div class="google-form-panel" id="form-panel">
                <button class="toggle-btn" id="toggle-btn" onclick="toggleForm()">📋 Open Form</button>
                <div style="width: 100%; height: 100%; min-width: 450px;" id="form-container"></div>
            </div>

            <script>
                function toggleForm() {{
                    const body = document.body;
                    const btn = document.getElementById('toggle-btn');
                    
                    body.classList.toggle('form-open');
                    
                    if (body.classList.contains('form-open')) {{
                        btn.innerHTML = '✕ Close Form';
                    }} else {{
                        btn.innerHTML = '📋 Open Form';
                    }}
                    
                    let startTime = Date.now();
                    let resizeInterval = setInterval(() => {{
                        window.dispatchEvent(new Event('resize'));
                        if (Date.now() - startTime > 350) {{
                            clearInterval(resizeInterval);
                        }}
                    }}, 30);
                }}

                (function() {{
                    const baseUrl = "https://docs.google.com/forms/d/e/1FAIpQLSdMDms_IAwkhPwOVqLvU0oVghG9Xti1LMhTsPoujm2w2uXF9A/viewform?embedded=true";
                    const emailEntryId = "2010975770";
                    const stationEntryId = "1301492004";
                    const startEntry1Id = "598747998";
                    const endEntry1Id = "2001326573";
                    const startEntry2Id = "1136531299";
                    const endEntry2Id = "530564783";
                    const startEntry3Id = "1646852203";
                    const endEntry3Id = "2002260210";

                    const userEmail = localStorage.getItem('userEmail') || "";
                    const filename = window.location.pathname.split('/').pop();
                    const stationId = filename.replace('.html', '');
                    const start = "{start}";
                    const end = "{end}";

                    const finalUrl = baseUrl + 
                        "&entry." + stationEntryId + "=" + stationId + 
                        "&entry." + emailEntryId + "=" + encodeURIComponent(userEmail) + 
                        "&entry." + startEntry1Id + "=" + start + 
                        "&entry." + endEntry1Id + "=" + end +
                        "&entry." + startEntry2Id + "=" + start + 
                        "&entry." + endEntry2Id + "=" + end +
                        "&entry." + startEntry3Id + "=" + start + 
                        "&entry." + endEntry3Id + "=" + end;

                    document.getElementById('form-container').innerHTML = 
                        '<iframe src="' + finalUrl + '" frameborder="0">Loading form…</iframe>';
                }})();

                window.addEventListener('load', function() {{
                    setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 200); 
                }});
            </script>
        </body>
        </html>
        """
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(full_page_html)