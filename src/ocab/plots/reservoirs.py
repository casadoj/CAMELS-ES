from pathlib import Path
# from typing import Optional
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


from ocab.plots.utils import compute_annual_timeseries, compute_monthly_climatology, compute_climatology, define_y_limits
from ocab.signatures import baseflow_index, flashiness_index, slope_fdc, budyko
import ocab.meteorology as METEO

logger = logging.getLogger(__name__)


def plot_reservoir_timeseries(
    ts: pd.DataFrame,
    attrs: pd.Series,
    save: bool = False,
    **kwargs
    ):
    """
    Creates an interactive figure with two plots: the time series and the Budyko diagram.
    Interactive buttons allow to transform the specific discharge and precipitation data to 
    logarithmic or square root scales to identify issues in the low flows.
    
    Parameters:
    -----------
    ts: pandas.Series
        Discharge time series
    attributes: pd.Series
        Reservoir and dam attributes
    save: boolean
        If True, return an object with the Plotly figure. If false, show the figure
    
    Keyword arguments:
    ------------------
    alpha: float
        Transparency of the precipitation bar plot
    c_fill: string
        Color used to plot reservoir filling
    c_in: string
        Color used to plot inflow
    c_out: string
        Color used to plot outflow
    c_precip: string
        Color used to plot precipitation
    title: string
        Figure title
    """

    # Extract keywords
    alpha = kwargs.get('alpha', 0.2)
    c_fill = kwargs.get('c_fill', 'darkslategrey')
    c_in = kwargs.get('c_line', 'seagreen')
    c_out = kwargs.get('c_line', 'darkred')
    c_precip = kwargs.get('c_bar', 'lightblue')
    title = kwargs.get('title', None)

    # extract time series
    variables = ['filling', 'inflow_cms', 'inflow_mm', 'outflow_cms', 'outflow_mm'] #, 'precip_mm', 'temp_degC', 'pet_mm']
    missing_vars = list(set(variables).difference(ts.columns))
    if len(missing_vars) > 0:
        ts[missing_vars] = np.nan

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

    # TODO: once a simulation exists
    # # is the station simulated?
    # has_sim = 'outflow_mm_sim' in ts.columns
    # name_dis_obs = 'Discharge (obs)' if has_sim else 'Discharge'

    # Setup grid
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"colspan": 2, "secondary_y": True}, None], # row 1
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

    # add traces
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
            showlegend=False
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts.index, 
            y=ts['inflow_mm'], 
            name="Inflow", 
            line=dict(color=c_in, width=1), 
            visible='legendonly',
            legendgroup="I",
            showlegend=False
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts.index, 
            y=ts['outflow_mm'], 
            name="Outflow", 
            line=dict(color=c_out, width=1), 
            visible=True,
            legendgroup="O", 
            showlegend=False
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts.index, 
            y=ts['filling'], 
            name="Filling", 
            line=dict(color=c_fill, width=1), 
            visible=True,
            legendgroup="F", 
            showlegend=False
        ), 
        row=row, col=col, secondary_y=True
    )

    # set axes properties
    round_mm_d = 20
    scale_d = 100
    mm_range_d, fill_range_d = define_y_limits(
        ts, 
        round_mm_d, 
        scale_d,
        cols_primary=('precip_mm', 'inflow_mm', 'outflow_mm'),
        cols_secondary=('filling')
    )
    fig.update_yaxes(
        title_text="mm", 
        # rangemode='nonnegative',
        range=mm_range_d,
        dtick=round_mm_d,
        row=row, col=col, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Filling (-)",
        range=fill_range_d,
        dtick=round_mm_d / scale_d,
        row=row, col=col, secondary_y=True,
    )

    # --- PLOT 2: MISSING DATA ---

    row, col = 2, 1
    size = 2.5

    # find missing outflow dates
    missing_x_in = ts[ts['inflow_mm'].isnull()].index
    missing_y_in = [0] * len(missing_x_in)
    fig.add_trace(
        go.Scatter(
            x=missing_x_in,
            y=missing_y_in, 
            mode='markers',
            marker=dict(
                symbol='line-ns-open',
                size=size,
                color=c_in,
                line_width=1
            ),
            visible='legendonly',
            legendgroup="I",
            showlegend=False,
            hoverinfo='x'
        ),
        row=row, col=col
    )
    # find missing outflow dates
    missing_x_out = ts[ts['outflow_mm'].isnull()].index
    missing_y_out = [10] * len(missing_x_out)
    fig.add_trace(
        go.Scatter(
            x=missing_x_out,
            y=missing_y_out, 
            mode='markers',
            marker=dict(
                symbol='line-ns-open',
                size=size,
                color=c_out,
                line_width=1
            ),
            legendgroup="O",
            showlegend=False,
            hoverinfo='x'
        ),
        row=row, col=col
    )
    # find missing filling dates
    missing_x_fill = ts[ts['filling'].isnull()].index
    missing_y_fill = [20] * len(missing_x_fill)
    fig.add_trace(
        go.Scatter(
            x=missing_x_fill,
            y=missing_y_fill, 
            mode='markers',
            marker=dict(
                symbol='line-ns-open',
                size=size,
                color=c_fill,
                line_width=1
            ),
            legendgroup="F",
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
    ts_y = compute_annual_timeseries(ts)
    ts_y.index += pd.DateOffset(months=6) # to adapt labels in the plots

    # add traces
    fig.add_trace(
        go.Bar(
            x=ts_y.index, 
            y=ts_y[f'precip_mm_{default_suffix}'], 
            name="Precipitation", 
            marker_color=c_precip, 
            opacity=1 - alpha,
            legendgroup="P",
            showlegend=True,
            hovertemplate="<b>Year: %{x|%Y}</b><br>P: %{y:.0f} mm<extra></extra>"
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts_y.index, 
            y=ts_y['inflow_mm'], 
            name="Inflow",
            line=dict(color=c_in, width=2),
            legendgroup="I",
            visible='legendonly',
            showlegend=True,
            hovertemplate="<b>Year: %{x|%Y}</b><br>I: %{y:.0f} mm<extra></extra>"
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts_y.index, 
            y=ts_y['outflow_mm'], 
            name="Outflow",
            line=dict(color=c_out, width=2),
            legendgroup="O",
            showlegend=True,
            hovertemplate="<b>Year: %{x|%Y}</b><br>O: %{y:.0f} mm<extra></extra>"
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts_y.index, 
            y=ts_y['filling'], 
            name="Filling",
            line=dict(color=c_fill, width=2),
            legendgroup="F",
            showlegend=True,
            hovertemplate="<b>Year: %{x|%Y}</b><br>F: %{y:.2f}<extra></extra>"
        ),
        row=row, col=col, secondary_y=True
    )

    # set axes properties
    round_mm_y = 500
    scale_y = 1000
    mm_range_y, fill_range_y = define_y_limits(
        ts_y, 
        round_mm_y, 
        scale_y,
        cols_primary=('precip_mm', 'inflow_mm', 'outflow_mm'),
        cols_secondary=('filling')
    )
    fig.update_yaxes(
        title_text="mm", 
        range=mm_range_y, 
        dtick=round_mm_y, 
        row=row, col=col
    )
    fig.update_yaxes(
        title_text="Filling (-)",
        range=fill_range_y,
        dtick=round_mm_y / scale_y,
        row=row, col=col, secondary_y=True,
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

    # add traces
    fig.add_trace(
        go.Bar(
            x=ts_m.index, 
            y=ts_m[f'precip_mm_{default_suffix}'], 
            name="Precipitation", 
            marker_color=c_precip, 
            opacity=1 - alpha,
            legendgroup="P",
            showlegend=False
        ), 
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts_m.index, 
            y=ts_m['inflow_mm'], 
            name="Inflow", 
            line=dict(color=c_in, width=2),
            visible='legendonly',
            legendgroup="I",
            showlegend=False
        ), 
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts_m.index, 
            y=ts_m['outflow_mm'], 
            name="Outflow", 
            line=dict(color=c_out, width=2),
            legendgroup="O",
            showlegend=False
        ), 
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=ts_m.index, 
            y=ts_m['filling'], 
            name="Filling", 
            line=dict(color=c_fill, width=2),
            legendgroup="F",
            showlegend=False
        ),
        row=row, col=col, secondary_y=True
    )

    # set axes properties
    round_mm_m = 20
    scale_m = 100
    mm_range_m, fill_range_m = define_y_limits(
        ts_m, 
        round_mm_m,
        scale_m,
        cols_primary=('precip_mm', 'inflow_mm', 'outflow_mm'),
        cols_secondary=('filling')
    )
    fig.update_yaxes(
        title_text="mm", 
        range=mm_range_m, 
        dtick=round_mm_m, 
        row=row, col=col
    )
    fig.update_yaxes(
        title_text="Filling (-)",
        range=fill_range_m,
        dtick=round_mm_m / scale_m,
        row=row, col=col, secondary_y=True,
    )
    fig.update_xaxes(
        tickvals=list(range(1, 13)), 
        ticktext=['J','F','M','A','M','J','J','A','S','O','N','D'], 
        row=row, col=col
    )

    # --- PLOT 5: BUDYKO DIAGRAM ---

    row, col = 4, 2

    # calculate indices
    pet_y = ts_y[f'pet_mm_{default_suffix}']
    precip_y = ts_y[f'precip_mm_{default_suffix}']
    aridity = pet_y / precip_y 
    evaporativity = (precip_y - ts_y['outflow_mm']) / precip_y

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

    # Observed outflow
    fig.add_trace(
        go.Scatter(
            x=aridity, 
            y=evaporativity,
            mode='markers',
            name='Budyko',
            marker=dict(color=c_out, size=8, line=dict(width=1, color='white')),
            customdata=aridity.index.year,
            hovertemplate="<b>Year: %{customdata}</b><br>Arid. Index: %{x:.2f}<br>Evap. Index: %{y:.2f}<extra></extra>",
            legendgroup="Outflow",
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
            "<b>Properties</b><br>"
            f"Capacity: {attrs.cap_mcm:.1f} hm³<br>"
            f"Surface: {attrs.area_skm:.1f} km²<br>"
            f"Catchment: {attrs.catch_skm:.1f} km²<br>"
            f"Deg. Regulation: {attrs.dor_d:.0f} days<br>"
            f"Deg. Disruptivity: {attrs.dod_m * 1000:.0f} mm<br>"
            f"Use: {attrs.main_use}<br>"
            "<br><b>Climatology</b><br>"
            f"Filling: {climatology.filling:.2f}<br>"
            f"Inflow: {climatology.inflow_mm:.0f} mm/year<br>"
            f"Outflow: {climatology.outflow_mm:.0f} mm/year<br>"
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
    _, bfi = baseflow_index(ts['outflow_cms'])
    fi = flashiness_index(ts['outflow_cms'])
    _, slope = slope_fdc(ts['outflow_mm'])

    # Calculate climatology values
    climatology = compute_climatology(ts)

    # signature_text = (
    #     "<b>Properties</b><br>"
    #     f"Capacity: {attributes.cap_mcm:.1f} hm³<br>"
    #     f"Surface: {attributes.area_skm:.1f} km²<br>"
    #     f"Catchment: {attributes.catch_skm:.1f} km²<br>"
    #     f"Deg. Regulation: {attributes.dor_d:.0f} days<br>"
    #     f"Deg. Disruptivity: {attributes.dod_m * 1000:.0f} mm<br>"
    #     f"Use: {attributes.main_use}<br>"
    #     "<br><b>Climatology</b><br>"
    #     f"Filling: {climatology.filling:.2f}<br>"
    #     f"Inflow: {climatology.inflow_mm:.0f} mm/year<br>"
    #     f"Outflow: {climatology.outflow_mm:.0f} mm/year<br>"
    #     f"Precipitation: {climatology.precip_mm:.0f} mm/year<br>"
    #     f"PET: {climatology.pet_mm:.0f} mm/year<br>"
    #     f"Temperature: {climatology.temp_degC:.1f} °C<br>"
    #     "<br><b>Hydrological Signatures</b>"
    #     f"Baseflow Index: {bfi:.2f}<br>"
    #     f"Flashiness Index: {fi:.2f}<br>"
    #     f"Slope FDC: {slope:.2f}<br>"
    # )

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
    base_annual_idx = 7 #if has_sim else 7
    base_climo_idx = 11 #if has_sim else 7
    base_budyko_idx = 18 # if has_sim else 14
    meteo_target_indices = [
        0,
        base_annual_idx,
        base_climo_idx,
        base_budyko_idx
    ]
    # if has_sim:
    #     meteo_target_indices += [base_budyko_idx + 1]

    meteo_buttons = []
    for label, suffix in available_datasets.items():
        # define time series
        daily_p = ts[f'precip_mm_{suffix}']
        annual_p = ts_y[f'precip_mm_{suffix}']
        climo_p = ts_m[f'precip_mm_{suffix}']
        budyko_arid = ts_y[f'pet_mm_{suffix}'] / ts_y[f'precip_mm_{suffix}'] 
        budyko_evap = (ts_y[f'precip_mm_{suffix}'] - ts_y['outflow_mm']) / ts_y[f'precip_mm_{suffix}']

        x = [ts.index, ts_y.index, ts_m.index, budyko_arid]
        y = [daily_p, annual_p, climo_p, budyko_evap]
        # if has_sim:
        #     budyko_evap_sim = (ts_y[f'precip_mm_{suffix}'] - ts_y['outflow_mm_sim']) / ts_y[f'precip_mm_{suffix}']
        #     x += [budyko_arid]
        #     y += [budyko_evap_sim]

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
    y_raw = [ts[f'precip_mm_{default_suffix}'], ts['inflow_mm'], ts['outflow_mm']]
    scale_target_indices = [0, 1, 2]
    # if has_sim:
    #     y_raw += [ts['outflow_mm_sim']]
    #     scale_target_indices += [3]
    y_sqrt = [trace**.5 for trace in y_raw]

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


# def create_reservoir_html(
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
#             <title>beaverses_{title}</title>
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
#                 <div class="back-nav">
#                     <a href="../../../index.html" class="back-btn">← Back to map</a>
#                 </div>
#                 {plotly_html}
#             </div>

#             <div class="google-form" id="form-container"></div>

#             <script>
#                 (function() {{
#                     const baseUrl = "https://docs.google.com/forms/d/e/1FAIpQLSdcqEbw2bhJXLUqHFCQLBz0LEbCBUesnxeT32L5l5XkNAfMyg/viewform?embedded=true";
#                     const emailEntryId = "204970380";
#                     const stationEntryId = "889543648";
#                     const startEntry1Id = "1625252996";
#                     const endEntry1Id = "314932285";
#                     const startEntry2Id = "1590382732";
#                     const endEntry2Id = "1770881539";
#                     const startEntry3Id = "208904126";
#                     const endEntry3Id = "1937793816";

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
#                     setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 100); 
#                 }});
#             </script>
#         </body>
#         </html>
#         """
    
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(full_page_html)


def create_reservoir_html(
    fig, 
    path: str,
    start: str,
    end: str
):
    """Wraps a Plotly figure into the full validation HTML template.
    
    Parameters:
    -----------
    fig:
        Result of `plot_reservoir_timeseries()`
    path: string
        Name of the HTML file where the figure wil be saved
    start: string
        Start date of the time series. Format YYYY-mm-dd
    end: string
        End date of the time series. Format YYYY-mm-dd
    """

    fig.update_layout(height=None, width=None, autosize=True)

    # Convert figure to HTML div string
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
            <title>beaverses_{title}</title>
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
                    const baseUrl = "https://docs.google.com/forms/d/e/1FAIpQLSdcqEbw2bhJXLUqHFCQLBz0LEbCBUesnxeT32L5l5XkNAfMyg/viewform?embedded=true";
                    const emailEntryId = "204970380";
                    const stationEntryId = "889543648";
                    const startEntry1Id = "1625252996";
                    const endEntry1Id = "314932285";
                    const startEntry2Id = "1590382732";
                    const endEntry2Id = "1770881539";
                    const startEntry3Id = "208904126";
                    const endEntry3Id = "1937793816";

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