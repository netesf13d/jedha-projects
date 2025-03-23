# -*- coding: utf-8 -*-
"""
Functions for the analysis of rental delays 
"""

import numpy as np
import pandas as pd


def prepare_dataset(filename: str)-> pd.DataFrame:
    """
    Prepare the dataset for rental delay analysis.
    
    The preparation steps are:
    - Load the data from xlsl file.
    - Join pairs of successive rentals
    - Binarize the rental `state` -> `is_canceled`
    - Compute the waiting time
    - Discard now irrelevant columns ('rental_id', etc)
    """
    # load
    raw_df = pd.read_excel(filename)
    # join pairs of subsequent rentals
    prev_rental_cols = ['rental_id', 'delay_at_checkout_in_minutes']
    curr_rental_cols = ['previous_ended_rental_id', 'checkin_type', 'state',
                        'time_delta_with_previous_rental_in_minutes']
    df = pd.merge(raw_df.loc[:, prev_rental_cols], raw_df.loc[:, curr_rental_cols],
                  how='inner', left_on='rental_id',
                  right_on='previous_ended_rental_id')
    df = df.assign(is_canceled=(df['state'] == 'canceled')) \
           .drop('rental_id', axis=1) \
           .drop('previous_ended_rental_id', axis=1) \
           .drop('state', axis=1)
    # compute waiting time
    waiting_time = (df['delay_at_checkout_in_minutes']
                    - df['time_delta_with_previous_rental_in_minutes'])
    df = df.assign(waiting_time=waiting_time) \
           .drop('delay_at_checkout_in_minutes', axis=1) \
           .drop('time_delta_with_previous_rental_in_minutes', axis=1)
    return df


def cancel_prob(dataset: pd.DataFrame,
                time_bins: np.ndarray)-> dict[str, np.ndarray]:
    """
    Estimate cancellation probability in `time_bins` from the `dataset`.
    
    The dataset must contain columns 'checkin_type', 'is_canceled',
    and 'waiting_time'.
    
    Returns
    -------
    p_cancel : dict[str, np.ndarray]
        The estimated cancellation probability in `time_bins` for each checkin
        method and also the combined ('total') methods.
    """
    p_cancel = {}
    ## Distinct checkout methods
    for (checkin,), df in dataset.groupby(['checkin_type']):
        df = df.loc[~df['waiting_time'].isna()]
        # estimation in bins
        prob = np.zeros(len(time_bins)+1, dtype=float)
        idx = np.digitize(df['waiting_time'], time_bins)
        for i in range(len(time_bins)+1):
            prob[i] = df.iloc[idx == i]['is_canceled'].mean()
        p_cancel[checkin]  = prob

    ## Combined checkout methods
    df = dataset.loc[~dataset['waiting_time'].isna()]
    prob = np.zeros(len(time_bins)+1, dtype=float)
    idx = np.digitize(df['waiting_time'], time_bins)
    for i in range(len(time_bins)+1):
        prob[i] = df.iloc[idx == i]['is_canceled'].mean()
    p_cancel['total']  = prob

    return p_cancel


def _cancel_rate(df: pd.DataFrame,
                 p_cancel: np.ndarray,
                 time_bins: np.ndarray,
                 time_delta: float,):
    """
    TODO doc

    Args:
        df (pd.DataFrame): _description_
        time_delta (float): _description_
        p_cancel (dict[str, np.ndarray]): _description_
        time_bins (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    idx = np.digitize(df['waiting_time'], time_bins + time_delta)
    return np.sum(p_cancel[idx]) / len(df)


def cancellation_rates(dataset: pd.DataFrame,
                      p_cancel: dict[str, np.ndarray],
                      time_bins: np.ndarray,
                      rental_delays: np.ndarray
                      )-> dict[str, float | np.ndarray]:
    """
    TODO doc

    Args:
        dataset (pd.DataFrame): _description_
        rental_delays (np.ndarray): _description_

    Returns:
        dict[str, np.ndarray]: _description_
    """
    cancellation_rate = {}
    for (checkin,), df in dataset.groupby(['checkin_type']):
        df = df.loc[~df['waiting_time'].isna()]
        cancel_rate = np.zeros_like(rental_delays, dtype=float)
        for i, dt in enumerate(rental_delays):
            cancel_rate[i] = _cancel_rate(df, p_cancel[checkin], time_bins, dt)
        cancellation_rate[checkin] = cancel_rate
        # cancellation_rate[checkin + '_ntot'] = len(df)
    return cancellation_rate


def delay_info_df(dataset: pd.DataFrame)-> pd.DataFrame:
    """
    TODO doc
    Estimate the impact of the introduction of a rental delay `time_delta` on
    the number of rental cancellations, for both checkin methods.
    
    The dataset must contain columns 'checkin_type' and 'waiting_time'.
    
    The computation requires an estimate of the cancellation probability in
    contiguous time intervals. `time_delta` is expressed in minutes.
    
    The function returns a dict[str, float] with entries
    - '<checkin>_n_rentals': the total number of rentals using the given <checkin>
      method that are potentially affected by the rental delay.
    - '<checkin>_cancel_frac': the estimated fraction of rental cancellations
      for the given <checkin> method. This estimate includes the baseline
      cancellation rate. It should be multiplied by the estimated cancel
      fraction to get an estimate of the actual number of cancellations.
    """
    info_df = pd.DataFrame(
        np.zeros((7, 3), dtype=float),
        columns=['connect', 'mobile', 'total'],
        index=['Nb rentals',
               'Baseline cancel rate', 'Cancel rate', 'Cancel rate diff',
               'Baseline cancel nb', 'Cancel nb',  'Cancel nb diff']
    )
    
    ## Distinct checkout methods
    for (checkin,), df in dataset.groupby(['checkin_type']):
        df = df.loc[~df['waiting_time'].isna()]
        info_df.loc['Nb rentals', checkin] = len(df)
        info_df.loc['Baseline cancel rate', checkin] = df['is_canceled'].mean()
        info_df.loc['Baseline cancel nb', checkin] = df['is_canceled'].sum()
    
    ## Total
    df = dataset.loc[~dataset['waiting_time'].isna()]
    info_df.loc['Nb rentals', 'total'] = len(dataset)
    info_df.loc['Baseline cancel rate', 'total'] = df['is_canceled'].mean()
    info_df.loc['Baseline cancel nb', 'total'] = df['is_canceled'].sum()

    return info_df


def update_delay_info(dataset: pd.DataFrame,
                      p_cancel: dict[str, np.ndarray],
                      time_bins: np.ndarray,
                      info_df: pd.DataFrame,
                      time_delta: float )-> pd.DataFrame:
    """
    TODO doc
    Estimate the impact of the introduction of a rental delay `time_delta` on
    the number of rental cancellations, for both checkin methods.
    
    The dataset must contain columns 'checkin_type' and 'waiting_time'.
    
    The computation requires an estimate of the cancellation probability in
    contiguous time intervals. `time_delta` is expressed in minutes.
    
    The function returns a dict[str, float] with entries
    - '<checkin>_n_rentals': the total number of rentals using the given <checkin>
      method that are potentially affected by the rental delay.
    - '<checkin>_cancel_frac': the estimated fraction of rental cancellations
      for the given <checkin> method. This estimate includes the baseline
      cancellation rate. It should be multiplied by the estimated cancel
      fraction to get an estimate of the actual number of cancellations.
    """
    ## Distinct checkout methods
    for (checkin,), df in dataset.groupby(['checkin_type']):
        df = df.loc[~df['waiting_time'].isna()]
        cancel_frac = _cancel_rate(df, p_cancel[checkin], time_bins, time_delta)
        info_df.loc['Cancel rate', checkin] = cancel_frac
        info_df.loc['Cancel nb', checkin] = len(df) * cancel_frac
    
    ## Total
    df = dataset.loc[~dataset['waiting_time'].isna()]
    cancel_frac = _cancel_rate(df, p_cancel[checkin], time_bins, time_delta)
    info_df.loc['Cancel rate', 'total'] = cancel_frac
    info_df.loc['Cancel nb', 'total'] = len(df) * cancel_frac
    
    ## Differences
    info_df.loc['Cancel nb diff'] = (info_df.loc['Cancel nb']
                                     - info_df.loc['Baseline cancel nb'])
    info_df.loc['Cancel rate diff'] = (info_df.loc['Cancel rate']
                                     - info_df.loc['Baseline cancel rate'])

    return info_df