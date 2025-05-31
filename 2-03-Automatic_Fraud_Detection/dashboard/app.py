# -*- coding: utf-8 -*-
"""

"""

import os
from collections import defaultdict

import numpy as np
import httpx
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import (probe_api, get_pricing_models, get_categories, get_pricing,
                  prepare_dataset, cancel_prob, cancellation_rates,
                  delay_info_df, update_delay_info)


API_URL = os.environ['API_URL'].strip('/')

DATASET_FILENAME = './data/get_around_delay_analysis.xlsx'
TIME_BINS = np.linspace(-15, 225, 5)
RENTAL_DELAYS = np.linspace(-120, 300, 22)


# =============================================================================
#
# =============================================================================

def rental_delay_chart(rental_delays: np.ndarray,
                       cancel_rates: np.ndarray,
                       p0: float, n0: float,
                       y0: float, y1: float)-> go.Figure:
    """
    Produce a plotly Figure displaying the cancellation rate data.

    Parameters
    ----------
    rental_delays, cancel_rates: np.ndarray
        Cancellation rate data (x, y).
    y0, y1 : float
        y-axis initial range.
    p0, n0 : float
        Baseline cancellation probability and number of rental cancellations.
    """

    layout = go.Layout(
        width=700, height=400,
        margin=dict(l=60, r=40, t=50, b=40),
        showlegend=False,
        xaxis={'title': 'Rental delay (h)',
               'range': rental_delays[[0, -1]]/60,
               'gridcolor': '#ccc',
               'gridwidth': 0.2},
        yaxis={'title': 'Rental cancellation fraction',
               'range': [y0, y1],
               'gridcolor': '#ccc',
               'gridwidth': 0.2},
        yaxis2={'title': 'Rental cancellations',
                'range': [y0*n0/p0, y1*n0/p0],
                'showgrid': False,
                'side': 'right'},
        )

    hoverdata = np.array([100*cancel_rates, 100*(cancel_rates-p0),
                          n0/p0*cancel_rates, n0/p0*cancel_rates-n0]).T
    data_trace = go.Scatter(
        x=rental_delays/60, y=cancel_rates,
        customdata=hoverdata,
        hovertemplate=(
            'Cancel rate: %{customdata[0]:.2f} %<br>'
            'Rate difference: %{customdata[1]:.2f} %<br>'
            'Cancel number: %{customdata[2]:.2f}<br>'
            'Cancel diff: %{customdata[3]:.2f}<br>'
            '<extra></extra>'
        )
    )

    baseline_trace = go.Scatter(x=[-1000, 1000], y=[n0, n0], yaxis='y2',
                                line={'dash': '5, 4', 'color': 'black'})

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.update_layout(layout)
    fig.add_trace(data_trace)
    fig.add_trace(baseline_trace)
    return fig


# =============================================================================
# App initialization
# =============================================================================

########## Setup delay analysis session state ##########
if not 'dataset' in st.session_state:
    df = prepare_dataset(DATASET_FILENAME)
    st.session_state['dataset'] = df
    st.session_state['time_bins'] = TIME_BINS

    p_cancel = cancel_prob(df, TIME_BINS)
    st.session_state['p_cancel'] = p_cancel

    cancel_rates = cancellation_rates(df, p_cancel, TIME_BINS, RENTAL_DELAYS)
    st.session_state['rental_delays'] = RENTAL_DELAYS
    st.session_state['cancellation_rates'] = cancel_rates

    st.session_state['delay_info_df'] = delay_info_df(df)


########## Setup pricing optimization session state ##########
## Pricing optimization API available ?
if 'api_available' not in st.session_state:
    try:
        probe_api(API_URL)
    except httpx.HTTPError:
        st.session_state['api_available'] = False
    else:
        st.session_state['api_available'] = True
api_available = st.session_state['api_available']

## If the API is available, get available models
if 'pricing_models' not in st.session_state:
    st.session_state['pricing_models'] = defaultdict(list)
if api_available:
    st.session_state['pricing_models'] = get_pricing_models(API_URL)

## If the API is available, get categorical variables info
if 'categories' not in st.session_state:
    st.session_state['categories'] = defaultdict(list)
if api_available:
    st.session_state['categories'] = get_categories(API_URL)

## If the API is available, track the state of the current car attributes input
if api_available and not 'curr_data' in st.session_state:
    st.session_state['curr_model'] = ''
    st.session_state['curr_data'] = {}


## Delay analysis data
df = st.session_state['dataset']
time_bins = st.session_state['time_bins']
p_cancel = st.session_state['p_cancel']
rental_delays = st.session_state['rental_delays']
cancel_rates = st.session_state['cancellation_rates']
delay_info_df = st.session_state['delay_info_df']

## Pricing optimization
pricing_models = st.session_state['pricing_models']
categories = st.session_state['categories']


# =============================================================================
# Application structure
# =============================================================================

########## App configuration ##########
st.set_page_config(
    page_title='Getaround analysis dashboard',
    page_icon='\U0001f697',
    layout='wide'
)


########## App Title and Description ##########
st.title('Getaround dashboard \U0001f697')
st.markdown(
"""
\U0001f44b Hello there! Welcome to this Getaround car rental dashboard.
The dashboard provides:
- Visualizations and information related to the implementation of
  a rental delay.
- A connection to the pricing API to price a car by manually entering
  its characteristics
""")
st.caption('Data provided by [Jedha](https://jedha.co)')


# st.write(cancellation_rates) # test



col_plot, col_table = st.columns([1, 0.5], gap='medium')

with col_plot:

    ## Mobile checkin plot
    n0 = delay_info_df.loc['Baseline cancel nb', 'mobile']
    p0 = delay_info_df.loc['Baseline cancel rate', 'mobile']
    mobile_fig = rental_delay_chart(
        rental_delays, cancel_rates['mobile'], p0, n0, 0.08, 0.12)
    mobile_fig.update_layout(
        title={'text': 'Mobile checkin', 'y':0.94, 'x':0.5},)
    st.plotly_chart(mobile_fig, key='mobile_chart', theme=None,
                    use_container_width=True)

    ## Getaround connect checkin plot
    n0 = delay_info_df.loc['Baseline cancel nb', 'connect']
    p0 = delay_info_df.loc['Baseline cancel rate', 'connect']
    getaround_fig = rental_delay_chart(
        rental_delays, cancel_rates['connect'], p0, n0, 0.14, 0.18)
    getaround_fig.update_layout(
        title={'text': 'Getaround connect checkin', 'y':0.94, 'x':0.5},)
    st.plotly_chart(getaround_fig, key='getaround_chart', theme=None,
                    use_container_width=True)


with col_table:
    # Rental delay slider
    rental_delay = st.slider('Rental delay (minutes)',
                             min_value=-400, max_value=400, value=0, step=2)

    # Rental delay summary table
    delay_info_container = st.empty()





# =============================================================================
# Dynamically changing content
# =============================================================================

with delay_info_container:
    delay_info_df = update_delay_info(df, p_cancel, time_bins,
                                      delay_info_df, rental_delay)
    delay_info_table = st.table(delay_info_df)
