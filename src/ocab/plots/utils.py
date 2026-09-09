import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

AGGREGATION = {
    'discharge_mm': 'sum',
    # 'discharge_mm_sim': 'sum',
    'filling': 'mean',
    'inflow_mm': 'sum',
    'inflow': 'mean',
    'outflow_mm': 'sum',
    'outflow': 'mean',
    'pet_mm': 'sum',
    'precip_mm': 'sum',
    'storage_mcm': 'mean',
    'storage': 'mean',
    'temp_degC': 'mean',
}

ROUNDING = {
    'discharge_mm': 1,
    # 'discharge_mm_sim': 1,
    'filling': 3,
    'inflow_cms': 3,
    'inflow_mm': 1,
    'inflow': 3,
    'outflow_cms': 3,
    'outflow_mm': 1,
    'outflow': 3,
    'pet_mm': 1,
    'precip_mm': 1,
    'temp_degC': 1,
    'storage_mcm': 3,
    'storage': 3
}


def _define_mapping(timeseries: pd.DataFrame, mapping: Dict) -> Dict:
    """
    Finds the mapping between the columns of a time series with some action to be applied on
    it, such as aggregation, rounding, etc

    Parameters:
    -----------
    timeseries: pandas.DataFrame
        The mapping will be applied to its columns
    mapping: dictionary
        It contains the simplified variable names (e.g. variable_unit) and the value to be mapped

    Returns:
    --------
    dictionary
        A dictionary that maps the action to be applied upon the columns in 'timeseries' whose name 
        starts with one of the values in 'mapping'
    """
    dct = {}
    for key, value in mapping.items():
        cols = timeseries.columns[timeseries.columns.str.startswith(key)]
        if len(cols) > 0:
            dct.update({col: value for col in cols}),
    return dct


def compute_monthly_climatology(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Creates a monthly climatology by resampling the daily data and then averaging across months."""

    agg_logic = _define_mapping(timeseries, AGGREGATION)
    if len(agg_logic) == 0:
        raise ValueError("No variables in the timeseries match the aggregation logic.")
    round_logic = _define_mapping(timeseries, ROUNDING)
    
    # resample to monthly time series
    ts_monthly = timeseries.resample('MS').agg(agg_logic)
    # compute monthly averages
    ts_monthly = ts_monthly.groupby(ts_monthly.index.month).mean()

    return ts_monthly.round(round_logic)


def compute_annual_timeseries(
        timeseries: pd.DataFrame, 
        rule: str = 'YS-OCT'
    ) -> pd.DataFrame:
    """Creates the annual time series by resampling the daily data.
    
    Parameters
    ----------
    timeseries: pd.DataFrame
        Daily time series
    rule: string
        How to aggregate the years. By default, it uses hydrological years (start in October)
    """

    agg_logic = _define_mapping(timeseries, AGGREGATION)
    if len(agg_logic) == 0:
        raise ValueError("No variables in the timeseries match the aggregation logic.")
    round_logic = _define_mapping(timeseries, ROUNDING)
    
    # resample to yearly time series
    ts_year = timeseries.resample(rule).agg(agg_logic)
    ts_year.replace(0, np.nan, inplace=True)

    return ts_year.round(round_logic)


def compute_climatology(
        timeseries: pd.DataFrame, 
        rule: str = 'YS-OCT'
    ) -> pd.DataFrame:
    """Computes climatological values.
    
    Parameters
    ----------
    timeseries: pd.DataFrame
        Daily time series
    rule: string
        How to aggregate the years. By default, it uses hydrological years (start in October)
    """

    timeseries_year = compute_annual_timeseries(timeseries, rule)
    climatology = pd.DataFrame(timeseries_year.mean(skipna=True, axis=0)).transpose()

    round_logic = _define_mapping(climatology, ROUNDING)
    return climatology.round(round_logic).squeeze()


def compute_annual_timeseries(
        timeseries: pd.DataFrame, 
        rule: str = 'YS-OCT'
    ) -> pd.DataFrame:
    """Creates the annual time series by resampling the daily data.
    
    Parameters
    ----------
    timeseries: pd.DataFrame
        Daily time series
    rule: string
        How to aggregate the years. By default, it uses hydrological years (start in October)
    """

    def scaled_sum(series):
        """Custom function to compute the sum with missing values"""
        if series.isna().all():
            return np.nan
        return series.mean() * len(series)

    agg_logic = _define_mapping(timeseries, AGGREGATION)
    agg_logic = {col: scaled_sum if func == 'sum' else func for col, func in agg_logic.items()}
    # agg_logic = {}
    # for var, func in AGGREGATION.items():
    #     if var not in timeseries.columns:
    #         continue
    #     if func == 'sum':
    #         agg_logic[var] = scaled_sum
    #     else:
    #         agg_logic[var] = func
    if len(agg_logic) == 0:
        raise ValueError("No variables in the timeseries match the aggregation logic.")
    round_logic = _define_mapping(timeseries, ROUNDING)
    
    # 1. Resample and aggregate
    resampled = timeseries.resample(rule)
    ts_year = resampled.agg(agg_logic)
    
    # 2. Safety Check: Mask years with too much missing data
    n_data = resampled.count()
    n_days = resampled.size()
    for col in ts_year.columns:
        mask = n_data[col] < (n_days * 0.90)
        ts_year.loc[mask, col] = np.nan

    ts_year.replace(0, np.nan, inplace=True)

    return ts_year.round(round_logic)


def define_y_limits(
        df: pd.DataFrame,
        round_primary: int,
        scale: float = 1,
        cols_primary: Tuple = ('precip_mm', 'discharge_mm', 'pet_mm'),
        cols_secondary: Optional[Tuple] = ('temp_degC')
):
    """
    """
    
    eps = round_primary / 10
    
    # 1. Define mm limits

    # extract values
    cols_primary = df.columns[df.columns.str.startswith(cols_primary)]
    pri_vals = df[cols_primary].values.flatten()
    pri_vals = pri_vals[~np.isnan(pri_vals)]   

    # define min/max values
    pri_min = 0.0
    pri_max = np.nanmax(pri_vals) if pri_vals.size > 0 else 100
    pri_max = np.ceil(pri_max / round_primary) * round_primary

    # 2. Define °C limits
    if cols_secondary is not None:
        round_sec = round_primary / scale

        # extract values
        cols_secondary = df.columns[df.columns.str.startswith(cols_secondary)]
        sec_vals = df[cols_secondary].values.flatten()
        sec_vals = sec_vals[~np.isnan(sec_vals)]

        # define min/max values
        sec_min = np.nanmin(sec_vals) if sec_vals.size > 0 else 0
        sec_min = np.floor(sec_min / round_sec) * round_sec
        sec_max = np.nanmax(sec_vals) if sec_vals.size > 0 else 30
        sec_max = np.ceil(sec_max / round_sec) * round_sec
        
        # Sync the scales
        pri_min = min(pri_min, sec_min * scale) #- eps
        sec_min = pri_min / scale
        pri_max = max(pri_max, sec_max * scale) + eps
        sec_max = pri_max / scale
    else:
        # pri_min -= eps
        pri_max += eps
        sec_min, sec_max = None, None

    return [pri_min, pri_max], [sec_min, sec_max]