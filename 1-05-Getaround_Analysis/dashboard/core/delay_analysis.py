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
                 time_delta: float)-> float:
    """
    Estimate the rental cancellation rate at a given rental delay `time_delta`.
    The function uses `p_cancel`, an estimation of the cancellation probability
    in the given `time_bins`.

    The dataset `df` must contain the column 'waiting_time'.
    """
    idx = np.digitize(df['waiting_time'], time_bins + time_delta)
    return np.sum(p_cancel[idx]) / len(df)


def cancellation_rates(dataset: pd.DataFrame,
                      p_cancel: dict[str, np.ndarray],
                      time_bins: np.ndarray,
                      rental_delays: np.ndarray
                      )-> dict[str, np.ndarray]:
    """
    Estimate the cancellation rates for both checkin methods at the selected
    `rental_delays`.
    """
    cancellation_rate = {}
    for (checkin,), df in dataset.groupby(['checkin_type']):
        df = df.loc[~df['waiting_time'].isna()]
        cancel_rate = np.zeros_like(rental_delays, dtype=float)
        for i, dt in enumerate(rental_delays):
            cancel_rate[i] = _cancel_rate(df, p_cancel[checkin], time_bins, dt)
        cancellation_rate[checkin] = cancel_rate
    return cancellation_rate


def delay_info_df(dataset: pd.DataFrame)-> pd.DataFrame:
    """
    Setup the dataframe holding info relative to the rental delay feature.
    Compute the baselines values which are static.
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
    Update the rental delay info dataframe with estimates at the selected
    rental delay `time_delta`.
    """
    ## Distinct checkout methods
    for (checkin,), df in dataset.groupby(['checkin_type']):
        df = df.loc[~df['waiting_time'].isna()]
        cancel_frac = _cancel_rate(df, p_cancel[checkin], time_bins, time_delta)
        info_df.loc['Cancel rate', checkin] = cancel_frac
        info_df.loc['Cancel nb', checkin] = len(df) * cancel_frac

    ## Total
    df = dataset.loc[~dataset['waiting_time'].isna()]
    cancel_frac = _cancel_rate(df, p_cancel['total'], time_bins, time_delta)
    info_df.loc['Cancel rate', 'total'] = cancel_frac
    info_df.loc['Cancel nb', 'total'] = (info_df.loc['Cancel nb', 'mobile']
                                         +info_df.loc['Cancel nb', 'connect'])

    ## Differences
    info_df.loc['Cancel nb diff'] = (info_df.loc['Cancel nb']
                                     - info_df.loc['Baseline cancel nb'])
    info_df.loc['Cancel rate diff'] = (info_df.loc['Cancel rate']
                                     - info_df.loc['Baseline cancel rate'])

    return info_df