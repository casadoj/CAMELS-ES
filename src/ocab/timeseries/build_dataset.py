from typing import Tuple

import numpy as np
import pandas as pd


def combine_periods(row: pd.Series, days: int = 365) -> pd.Series:
    """Combines the study periods defined in the questionnaire.
    
    Parameters
    ----------
    row: pandas.Series
        Contains indices such as "start_1", "end_1", etc
    days: int
        Number of days used to predict dicharge in the LTSM model. It is used to extend
        to the past the study period.

    Returns
    -------
    pd.Series
        Contains two indices ("start_dates" and "end_dates") with the start and end of each study period
    """

    starts = [row['start_1']]
    ends = [row['end_1']]
    if row['second_period'] == 'Yes':
        starts.append(row['start_2'])
        ends.append(row['end_2'])
        if row['third_period'] == 'Yes':
            starts.append(row['start_3'])
            ends.append(row['end_3'])
    starts = pd.to_datetime(starts, dayfirst=True) - pd.Timedelta(days=days)
    ends = pd.to_datetime(ends, dayfirst=True)

    return pd.Series({"start_dates": starts, "end_dates": ends})


def valid_timeseries(
        answers: pd.DataFrame, 
        column: str ='incorrect_ts', 
        inplace: bool = False
    ) -> pd.DataFrame | None:
    """Creates a DataFrame with fields of valid time series, based on the answers to the 
    online questionnaire.
    
    Parameters
    ----------
    answers: pd.DataFrame
        The raw answers to the online questionnaire.
    columnn: str
        Name of the column indicating the incorrect time series.
    inplace: bool
        If True, the original DataFrame is modified in place. If False, a new DataFrame is 
        returned with the valid time series fields.

    Returns
    -------
    pd.DataFrame or None
        * pd.DataFrame : Binary indicator DataFrame if `inplace=False`.
        * None : If `inplace=True`.
    """

    # split list of variables
    incorrect_ts = answers[column].str.split(', ').copy()

    # extract possible variables and create a rename mapping
    old_vars = incorrect_ts.explode().dropna().unique().tolist()
    rename_vars = {var: var.split()[0].lower() for var in old_vars}
    new_vars = list(rename_vars.values())

    # apply rename mapping
    incorrect_ts = incorrect_ts.apply(
        lambda lst: [rename_vars[x] for x in lst if x in rename_vars]
        if isinstance(lst, list) else []
    )

    # create fields of valid time series
    valid_ts = pd.DataFrame(1, index=answers.index, columns=new_vars, dtype=int)
    for ID, lst in incorrect_ts.items():
        if len(lst) > 0:
            valid_ts.loc[ID, lst] = 0

    if inplace:
        answers.drop(columns=[column], inplace=True)
        for col in valid_ts.columns:
            answers[col] = valid_ts[col]
    else:
        return valid_ts


def time_encoding(
    time: np.ndarray,
    period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms time feature values in an xarray.DataArray to sine and cosine components.

    Parameters:
    -----------
    time: xarray.DataArray
        An xarray.DataArray with time feature values (e.g., month, day of year).
    period: integer
        The period of the time feature (e.g., 12 for months, 7 for days of the week).

    Returns:
    --------
    sin_da, cos_da (tuple of xarray.DataArray):
        Sine and cosine transformations of the time feature values.
    """
    
    # Normalize time feature values to [0, 2π]
    if time.min() == 1:
        norm_da = (time - 1) * 2 * np.pi / period
    elif time.min() == 0:
        norm_da = time * 2 * np.pi / period
    else:
        norm_da = (time - 1) * 2 * np.pi / period
        
    # correct leap years, if necessary
    norm_da = norm_da.where(norm_da <= np.pi * 2, np.pi * 2)
    
    return np.round(np.sin(norm_da), 8), np.round(np.cos(norm_da), 8)