import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_discharge(
        path: Path,
        id: Union[int, List[int]] = None, 
        start: Union[str, pd.Timestamp] = None, 
        end: Union[str, pd.Timestamp] = None
    ) -> pd.DataFrame:
    """
    Extracts daily discharge time series from the 'afliq.csv' file of the Anuario de Aforos. 
    The series can be clipped to the specific period and stations of interest.

    Parameters:
    -----------
    path:      Path
        Path to the 'afliq.csv' file containing the discharge series.
    id:        int or list, optional
        List of ROEA (Official Network of Gauging Stations) IDs whose discharge 
        series are to be extracted.
    start:     str or datetime.date, optional
        Start date of the study period.
    end:       str or datetime.date, optional
        End date of the study period.

    Returns:
    -------
    discharge:    pandas.DataFrame
        DataFrame containing the daily discharge time series.
    """

    if isinstance(id, int):
        id = [id]
    elif isinstance(id, list):
        id = [int(x) for x in id]
    elif id is not None:
        logger.error('"id" must be either an integer, a list or None.')
        return dict()

    # load time series
    file = path / 'afliq.csv'
    if not file.is_file():
        raise FileNotFoundError(f"Could not find required file: {file}")
    data = pd.read_csv(file, sep=';', index_col='indroea')

    # rename columns
    rename_cols = {
        'indroea': 'id_roea',
        'fecha': 'date',
        'altura': 'stage',
        'caudal': 'discharge'
    }
    data = data[rename_cols.keys()].rename(columns=rename_cols, errors='ignore')

    # set index
    data.set_index('id_roea', inplace=True)
    data.index = data.index.astype(int)

    # define dates
    data.date = pd.to_datetime(data.date, dayfirst=True)

    # crop to study period
    if start is not None:
        data = data.loc[data.date >= start, :]
    if end is not None:
        data = data.loc[data.date <= end, :]

    # reformat time series
    IDs = data.index.unique() if id is None else data.index.unique().intersection(id)
    discharge = pd.DataFrame(
        index=pd.date_range(data.date.min(), data.date.max(), freq='D'),
        columns=IDs,
        dtype=float
        )
    for stn in discharge.columns:
        data_stn = data.loc[stn].set_index('date', drop=True)
        discharge[stn] = data_stn['discharge']

    # eliminar estaciones sin ningún dato
    discharge.dropna(axis=1, how='all', inplace=True)
    discharge.columns = discharge.columns.astype(int)

    return discharge


def get_reservoir_series(
        path: Path, 
        id: Optional[Union[int, List[int]]] = None, 
        start: Optional[Union[str, pd.Timestamp]] = None, 
        end: Optional[Union[str, pd.Timestamp]] = None,
        inflow: bool = False,
        fill_value: float = np.nan
    ) -> Dict[int, pd.DataFrame]:
    """Extracts daily reservoir time series from the 'afliqe.csv' file of the 'Anuario de Aforos'. 
    The series can be clipped to the period and stations of interest.

    Parameters:
    -----------
    path: str
        Path to the file 'afliqe.csv' containing the reservoir time series.
    id: int or list
        List of ROEA (Official Network of Gauging Stations) reservoir IDs 
        whose series are to be extracted.
    start: str or datetime.date
        Start date of the study period.
    end: str or datetime.date
        End date of the study period.
    inflow: bool
        Whether to compute the inflow time series (True) or not (False, default).
    fill_value: float, optional
        Value used to replace negative calculated inflows. Defaults to 0.

    Returns:
    --------
    timeseries: Dict[str, pandas.DataFrame]
        Dictionary containing the dataframes for each reservoir's series.
    """
    
    # variables required to compute inflows
    required_cols = {'storage', 'outflow', 'type'}

    # adapt "id" type
    if isinstance(id, str):
        id = [int(id)]
    elif isinstance(id, list):
        id = [int(x) for x in id]
    elif id is not None:
        logger.error('"id" must be either an integer, a list or None.')
        return dict()
    
    # load time series
    file = path / 'afliqe.csv'
    if not file.is_file():
        raise FileNotFoundError(f"Could not find required file: {file}")
    data = pd.read_csv(file, sep=';')

    # rename columns
    rename_cols = {
        'ref_ceh': 'id_ceh',
        'fecha': 'date', 
        'reserva': 'storage', 
        'salida': 'outflow', 
        'tipo': 'type'
    }
    data = data[rename_cols.keys()].rename(columns=rename_cols, errors='ignore')

    # set index
    data.set_index('id_ceh', inplace=True)
    data.index = data.index.astype(int)

    # define dates
    data.date = pd.to_datetime(data.date, dayfirst=True)

    # trim to the study period
    if start is not None:
        data = data.loc[data.date >= start, :]
    if end is not None:
        data = data.loc[data.date <= end, :]

    # reformat time series
    timeseries = {}
    IDs = data.index.unique() if id is None else data.index.unique().intersection(id)
    for ID in IDs:
        # extract ID time series
        ts = data.loc[ID, :]

        # handle time stamps
        ts.set_index('date', drop=True, inplace=True)
        ts = ts.asfreq('D')
        start = ts.drop(columns=['type']).apply(pd.Series.first_valid_index).min()
        end = ts.drop(columns=['type']).apply(pd.Series.last_valid_index).max()
        ts =  ts.loc[start:end]

        # remove empty variables
        ts.dropna(axis=1, how='all', inplace=True)

        # compute inflows
        # WARNING: @casadoj computing inflow before cleaning the time series is dangerous!
        if inflow & required_cols.issubset(ts):
            ts = compute_inflow(ts, fill_value=fill_value)

        # save
        timeseries[ID] = ts.round(3)

    return timeseries


def compute_inflow(
        data: pd.DataFrame, 
        fill_value: float = np.nan,
    ) -> pd.DataFrame:
    """Computes reservoir inflow time series based on storage and outflow data.

    The function handles two mass-balance calculation types:
    - Type 1 (Backward): Inflow depends on current outflow and storage change.
    - Type 2 (Forward): Inflow depends on current outflow and next-day storage change.

    Parameters:
    -----------
    data: pandas.DataFrame
        Input time series containing columns 'storage' (million m³), 'outflow' (m³/s), 
        and 'type' (1 or 2).
    fill_value: float, optional
        Value used to replace negative calculated inflows. Defaults to 0.

    Returns:
    --------
    data: pandas.DataFrame
        DataFrame containing the calculated 'inflow' (m³/s) and a 'quality' 
        flag (1 for valid, 0 for corrected negative values).
    """

    required_cols = {'storage', 'outflow', 'type'}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f'The input DataFrame is missing required column(s): {missing}')
    
    # change in storage (m3/s)
    delta_t = data.index.to_series().diff().dt.total_seconds()
    delta_v = data['storage'].diff() / delta_t * 1e6

    # compute two potential inflows
    # type 1: Inflow[t] = ΔV[t] + Outflow[t]
    inflow_type_1 = delta_v + data['outflow']
    # type 2: Inflow[t] = ΔV[t+1] + Outflow[t]
    inflow_type_2 = delta_v.shift(-1) + data['outflow']

    # assign inflow
    data['inflow'] = np.where(data['type'] == 1, inflow_type_1, inflow_type_2)
    
    # Handle quality flags and negative values
    mask_neg = data['inflow'] < 0
    data['quality'] = np.where(mask_neg, 0, 1)
    data.loc[mask_neg, 'inflow'] = fill_value
        
    return data


def get_river_series(
        path: Path, 
        id: Union[int, List[int]] = None, 
        start: Union[str, pd.Timestamp] = None, 
        end: Union[str, pd.Timestamp] = None
        ) -> Dict[int, pd.DataFrame]:
    """
    Extracts daily discharge time series from the 'afliq.csv' file of the Anuario de Aforos. 
    The series can be clipped to the specific period and stations of interest.

    Parameters:
    -----------
    path:      Path
        Path to the 'afliq.csv' file containing the discharge series.
    id:        int or list, optional
        List of ROEA (Official Network of Gauging Stations) IDs whose discharge 
        series are to be extracted.
    start:     str or datetime.date, optional
        Start date of the study period.
    end:       str or datetime.date, optional
        End date of the study period.

    Returns:
    -------
    timeseries: Dict[str, pandas.DataFrame]
        Dictionary containing the dataframes for each gauge's series.
    """

    if isinstance(id, int):
        id = [id]
    elif isinstance(id, list):
        id = [int(x) for x in id]
    elif id is not None:
        logger.error('"id" must be either an integer, a list or None.')
        return dict()

    # load time series
    file = path / 'afliq.csv'
    if not file.is_file():
        raise FileNotFoundError(f"Could not find required file: {file}")
    data = pd.read_csv(file, sep=';')

    # rename columns
    rename_cols = {
        'indroea': 'id',
        'fecha': 'date',
        'altura': 'stage',
        'caudal': 'discharge'
    }
    data = data[rename_cols.keys()].rename(columns=rename_cols, errors='ignore')

    # set index
    data.set_index('id', inplace=True)
    data.index = data.index.astype(int)

    # define dates
    data.date = pd.to_datetime(data.date, dayfirst=True)

    # crop to study period
    if start is not None:
        data = data.loc[data.date >= start, :]
    if end is not None:
        data = data.loc[data.date <= end, :]

    # reformat time series
    IDs = data.index.unique() if id is None else data.index.unique().intersection(id)
    timeseries = {}
    for ID in IDs:
        # extract ID time series
        ts = data.loc[ID, :]
        
        # handle time stamps
        ts.set_index('date', drop=True, inplace=True)
        ts = ts.asfreq('D')
        start = ts.apply(pd.Series.first_valid_index).min()
        end = ts.apply(pd.Series.last_valid_index).max()
        ts =  ts.loc[start:end]

        # remove empty variables
        ts.dropna(axis=1, how='all', inplace=True)

        # save
        timeseries[ID] = ts

    return timeseries


def get_meteo_series(
        path: Path, 
        id: Optional[Union[int, List[int]]] = None, 
        start: Optional[Union[str, pd.Timestamp]] = None, 
        end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Dict[int, pd.DataFrame]:
    """
    Extracts monthly meteorological time series from the 'evap.csv' file of the 'Anuario de Aforos'. 
    The series can be clipped to the period and stations of interest.

    Parameters:
    -----------
    path: str
        Path to the 'evap.csv' file that contains the monthly meteorological time series.
    id: int or list
        List of ROEA (Official Network of Gauging Stations) evaporimeter IDs 
        whose series are to be extracted.
    start: str or datetime.date
        Start date of the study period.
    end: str or datetime.date
        End date of the study period.

    Returns:
    --------
    timeseries: Dict[str, pandas.DataFrame]
        Dictionary containing the monthly time series for each evaporimeter.
    """

    if isinstance(id, int):
        id = [id]
    elif isinstance(id, list):
        id = [int(x) for x in id]
    elif id is not None:
        logger.error('"id" must be either an integer, a list or None.')
        return dict()

    # load time series
    file = path / 'evap.csv'
    if not file.is_file():
        raise FileNotFoundError(f"Could not find required file: {file}")
    data = pd.read_csv(file, sep=';')

    # rename columns
    rename_cols = {
        'ref_evap': 'id_evap',
        'hora1': 'hour1',
        'hora2': 'hour2',
        'hora3': 'hour3',
        'evapiche': 'evap_piche',
        'evatanque': 'evap_pan',
        'temedmax': 'temp_max_avg',
        'temedmin': 'temp_min_avg',
        'temedsec1': 'temp_dry_1',
        'temedsec2': 'temp_dry_2',
        'temedsec3': 'temp_dry_3',
        'temedhum1': 'temp_wet_1',
        'temedhum2': 'temp_wet_2',
        'temedhum3': 'temp_wet_3',
        'velviento': 'wind_speed',
        'prectotmes': 'precip_total',
        'diasprec': 'precip_days',
        'insol': 'sunshine_hours'
    }
    data = data[rename_cols.keys()].rename(columns=rename_cols, errors='ignore')

    # set index
    data.set_index('id_evap', inplace=True)
    data.index = data.index.astype(int)

    # define dates
    data['date'] = pd.to_datetime(data.anomes, format='%Y%m')
    data.drop(['anomes'], axis=1, inplace=True)

    # crop to the study period
    if start is not None:
        data = data.loc[data.date >= start, :]
    if end is not None:
        data = data.loc[data.date <= end, :]
        
    # reformat time series
    IDs = data.index.unique() if id is None else data.index.unique().intersection(id)
    timeseries = {}
    for ID in IDs:
        # extract ID time series
        ts = data.loc[ID, :]

        # handle time stamps
        ts.set_index('date', drop=True, inplace=True)
        ts.sort_index(axis=0, inplace=True)
        ts = ts.asfreq('MS')

        # remove empty variables
        ts = ts.dropna(axis=1, how='all')

        # save
        timeseries[ID] = ts

    return timeseries


def select_study_period(
        timeseries: pd.DataFrame, 
        ID: Union[str, int],
        availability: float = 0.9, 
        years: int = 8,
        mode: Literal['all', 'any'] = 'all'
    ) -> pd.Series:
    """
    Finds the start and end dates of the best study period for a single station 
    considering multiple variables (columns) simultaneously.

    Parameters:
    -----------
    timeseries: pd.DataFrame
        DataFrame where columns are variables (e.g., 'storage', 'inflow') 
        and the index is daily DatetimeIndex.
    ID: string or integer
        Code of the station 'timeseries' refers to.
    availability: float
        Minimum fraction of days per year required for a year to be valid.
    years: int
        Minimum number of consecutive valid years.
    mode: str, {'all', 'any'}
        'all': Every variable must meet the availability in a given year.
        'any': At least one variable must meet the availability in a given year.

    Returns:
    ---------
    pd.DataFrame
        DataFrame with 'start' and 'end' years for the best period found.
    """

    assert 0 < availability <= 1, '"availability" must be a float between 0 and 1.'
    assert years > 0, '"years" must be a positive integer.'

    if isinstance(timeseries, pd.Series):
        timeseries = pd.DataFrame(timeseries)

    # 1. Compute yearly availability
    yearly_groups = timeseries.groupby(timeseries.index.year)
    yearly_availability = yearly_groups.count() / yearly_groups.size().values[:, None]

    # 2. Identify "good" years 
    # good years per variable
    good_years_per_var = yearly_availability >= availability

    # Collapse across variables based on mode
    if mode == 'all':
        is_good_year = good_years_per_var.all(axis=1)
    else:
        is_good_year = good_years_per_var.any(axis=1)

    # 3. Find consecutive sequences of True
    # We convert to int to use diff() to find transitions (0->1 is start, 1->0 is end)
    year_indices = is_good_year.index.values
    status = is_good_year.astype(int)
    diff = status.diff()
    diff.iloc[0] = status.iloc[0]

    starts = year_indices[diff == 1]
    ends = year_indices[diff.shift(-1) == -1]
    
    # If the last year is part of a sequence, it won't have a -1 transition
    if status.iloc[-1] == 1:
        ends = np.append(ends, year_indices[-1])

    # 4. Select Longest, then Most Recent
    durations = ends - starts + 1
    valid_mask = durations >= years
    
    if not any(valid_mask):
        return pd.DataFrame({'start': np.nan, 'end': np.nan}, index=[ID], dtype='Int64')

    # Find the maximum duration among those that meet the minimum requirement
    max_dur = np.max(durations[valid_mask])
    
    # Get indices of all sequences matching that max duration
    best_indices = np.where((durations == max_dur) & valid_mask)[0]
    
    # Pick the last one (most recent)
    final_idx = best_indices[-1]

    return pd.DataFrame({
        'start': starts[final_idx],
        'end': ends[final_idx]
        }, 
        index=[ID],
        dtype='Int64'
    )


def manage_conflicts(timeseries: dict, conflicts: dict):
    """Resolves conflicts between stations based on the provided solutions.""
    
    Parameters:
    -----------
    timeseries : dict
        A dictionary where keys are station IDs (int/str) and values are 
        pandas.Series containing the time series data.
    conflicts : dict
        A dictionary where keys are tuples of conflicting station IDs (e.g., (id1, id2)) 
        and values are strings: 'remove {ID}', 'keep {ID}', or 'combine'.

    Returns:
    --------
    dict
        Updated dictionary with merged or removed stations as per the strategies.
    """

    def _combine_timeseries(timeseries: dict, id1: int, id2: int):
        """
        Identifies the most recent station and uses it as the priority,
        filling its missing values with data from the older station.
        """

        # Determine priority based on which station has the most recent data
        ts1 = dct.get(id1)
        ts2 = dct.get(id2)
        if ts1.last_valid_index() >= ts2.last_valid_index():
            id_keep, id_remove = id1, id2
            primary, backup = ts1, ts2
        else:
            id_keep, id_remove = id2, id1
            primary, backup = ts2, ts1

        # combine_first uses 'primary' values, and fills NaNs using 'backup'
        combined = primary.combine_first(backup).asfreq('D')
        
        dct[id_keep] = combined
        dct.pop(id_remove, None)

        return dct
    
    dct = timeseries.copy()
    for ids, solution in conflicts.items():
        if not all([ID in dct for ID in ids]):
            continue

        solution = solution.lower()

        if solution.startswith('remove'):
            try:
                id_to_remove = int(solution.split()[1])
                dct.pop(id_to_remove, None)
            except (IndexError, ValueError):
                logger.warning(f"Invalid 'remove' format for conflict {ids}: {solution}")

        elif solution.startswith('keep'):
            try:
                id_to_keep = int(solution.split()[1])
                for ID in ids:
                    if ID != id_to_keep:
                        dct.pop(ID, None)
            except (IndexError, ValueError):
                logger.warning(f"Invalid 'keep' format for conflict {ids}: {solution}")

        elif solution == 'combine':
            _combine_timeseries(dct, *ids)
            
        else:
            logger.warning('Unrecognized solution for conflict between stations {0}: {1}'.format(ids, solution))
        
    return dct


def upsample_to_daily(
        ts_monthly: pd.DataFrame,
):
    """
    Upsamples a monthly time series to daily resolution.
    
    Parameters:
    -----------
    ts_monthly: pd.DataFrame
        Monthly time series.

    Returns:
    --------
    pd.DataFrame
        Daily time series.
    """
    # resample with forward fill
    ts_daily = ts_monthly.resample('D').ffill()

    # extend to the end of the last month
    last_day = ts_monthly.index.max() + pd.offsets.MonthEnd(0)
    ts_daily = ts_daily.reindex(
        pd.date_range(ts_daily.index.min(), last_day, freq='D'),
        method='ffill'
    )

    return ts_daily