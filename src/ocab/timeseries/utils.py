import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple


def resample_daily(
        df: pd.DataFrame, 
        decimals: int = 3
    ) -> pd.DataFrame:
    """Resample hourly time series to daily"""

    df = df.resample('D').mean().round(decimals)
    df.index = df.index.normalize()
    df.index.name = 'date'

    return df


def compute_filling(df: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Computes reservoir filling out of storage and total capacity"""
    
    if 'storage' in df.columns:
        df['filling'] = df['storage'] / capacity

    return df


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