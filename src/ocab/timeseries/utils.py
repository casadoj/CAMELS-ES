import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def resample_daily(
        df: pd.DataFrame, 
        decimals: int = 3
    ) -> pd.DataFrame:
    """Resample hourly time series to daily"""

    df = df.resample('D').mean().round(decimals)
    df.index = df.index.normalize()
    df.index.name = 'date'

    return df


def compute_filling(ts: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Computes reservoir filling out of storage and total capacity.
    
    Parameters
    ----------
    ts: pandas.DataFrame
        Time series that includes a column representing reservoir storage: "storage" or 
        "storage_mcm".
    capacity: float
        Maximum storage capacity. The units must be the same as those in the time series.

    Returns
    -------
    pandas.DataFrame
        Same as the input timeseries, but with a new column called 'filling'.
    """

    # Find storage column
    storage_col = next((col for col in ['storage', 'storage_mcm'] if col in ts.columns), None)
    if storage_col is None:
        raise ValueError('The input DataFrame must contain either "storage" or "storage_mcm" column.')

    # compute filling
    ts = ts.copy()
    ts['filling'] = ts[storage_col] / capacity

    return ts


def define_observed_period(
        timeseries: Dict[int, pd.DataFrame], 
        last_year: Optional[int] = None
    ) -> (pd.DataFrame):
    """Defines start and end year of the time series, and whether the station is active or not

    Parameters:
    -----------
    timeseries: dictionary
        A dictionary where keys are station IDs and values the observed time series
    last_year: integer
        Last recorded year. Used to define whether the station is active or not
    """

    if last_year is None:
        last_year = datetime.now().year
    df = pd.DataFrame(index=timeseries.keys(), columns=['start', 'end', 'active'], dtype='Int64')
    for ID, ts in timeseries.items():
        start, end = ts.index.min(), ts.index.max()
        df.loc[ID, 'start'] = start.year
        df.loc[ID, 'end'] = end.year if end.year < last_year else np.nan
        df.loc[ID, 'active'] = 1 if end.year == last_year else 0

    return df


def clean_discharge(
    ts: pd.DataFrame,
    q: str = 'discharge_mm',
    P: str = 'precip_mm',
    factor: float = 2.0
) -> pd.DataFrame:
    """Cleans the discharge timeseries (both in m³/s and mm) based on lower and upper 
    bound thresholds:
        1. Discharge can't be negative.
        2. Specific discharge can't exceed "factor" times the maximum in the indicated
           precipitation time series.

                0 <= q <= factor * max(P)

    Parameters
    ----------
    ts: pandas.DataFrame
        Time series to be corrected. It contains, at least, both discharge records 
        ("discharge_cms", "discharge_mm") and precipitation records.
    q: string
        Name of the column in "ts" that contains specific discharge.
    P: string
        Name of the column in "ts" that contains precipitation. It can be used to select
        multiple precipitation time series by using only the start of the column name. 
        For instance, "precip_mm" will select the columns "precip_mm_rocio", "precip_mm_cerra",
        ...
    factor: float
        Defines the upper threshold.

    Returns
    -------
    pandas.DataFrame
        A table similar to the input, but filled with NaN whenever the discharge thresholds are
        not met.
    """

    ts = ts.copy()

    # mask negative values
    mask_low = ts[q] < 0

    # mask values that exceed "factor" times the maximum precipitation
    cols_precip = ts.columns[ts.columns.str.startswith(P)]
    if len(cols_precip) == 0:
        raise ValueError(f'No precipitation columns starting with {P} found in "ts".')
    mask_high = ts[q] > (factor * ts[cols_precip].max().max())

    # remove values exceeded the previous thresholds
    cols_discharge = [col for col in ['discharge_cms', 'discharge_mm'] if col in ts.columns]
    ts.loc[mask_low | mask_high, cols_discharge] = np.nan

    return ts


def clean_storage(
    ts: pd.DataFrame,
    filling: str = 'filling',
    storage: str | None = 'storage',
    max_fill: float = 2.0,
    max_rate: float | None = None,
    window: int = 7
    ) -> pd.DataFrame:
    """Removes invalid values from a reservoir filling time series.

    Values are set to NaN if filling values exceed bounds [0, max_fill] or if absolute 
    deviations from a centered moving median exceed `max_rate`.

    Parameters
    ----------
    ts : pandas.DataFrame
        DataFrame containing reservoir storage and filling time series.
    filling : str, default 'filling'
        Column name for filling values (expected as a ratio/percentage time series).
    storage : str, default 'storage'
        Column name for storage values.
    max_fill : float, default 2.0
        Maximum allowable filling ratio. Values below 0 or above this limit are set to NaN.
    max_rate : float or None, default None
        Maximum acceptable absolute deviation from the rolling median (e.g., 0.1 for 10% change). 
        Time steps with a deviation above this threshold are converted to NaN. If None, this filter is skipped.
    window : int, default 7
        Window width for the centered rolling median.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with invalid values in `storage` and `filling` set to NaN.
    """

    if filling not in ts.columns:
        raise ValueError(f'"{filling}" is not a column in the input DataFrame.')

    cols = [filling]
    if storage is not None:
        if storage in ts.columns:
            cols.append(storage)
        else:
            raise ValueError(f'"{storage}" is not a column in the input DataFrame.')

    ts = ts.copy()

    # remove values exceeding the thresholds
    mask_thr = (ts[filling] < 0) | (ts[filling] > max_fill)
    if mask_thr.sum() > 0:
        logger.info(f'{mask_thr.sum()} filling values exceed the thresholds.')
        ts.loc[mask_thr, cols] = np.nan

    # relative error compared with the rolling median
    if max_rate is not None:
        median = ts[filling].rolling(window, center=True, min_periods=int(np.floor(window / 2))).median()
        rate = (ts[filling] - median)
        mask_rate = rate.abs() > max_rate
        if mask_rate.sum() > 0:
            logger.info(f'{mask_rate.sum()} filling values exceed the maximum rate {max_rate}.')
            ts.loc[mask_rate, cols] = np.nan
    
    return ts