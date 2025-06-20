# -*- coding: utf-8 -*-
"""
Main dashboard script. Contains 2 components:
- A tab proposing analytics related to the implementation of a rental delay.
- A tab that proposes a car rental pricing service
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
if 'dataset' not in st.session_state:
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
if api_available and 'curr_data' not in st.session_state:
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

tab_delay, tab_pricing = st.tabs(['Rental delay analysis', 'Car pricing'])


########## Rental delay tab ##########
with tab_delay:
    st.markdown('#### Car rental delay analysis')

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



########## Car pricing tab ##########
with tab_pricing:
    st.markdown('#### Car rental pricing')

    # Categorical variables
    cols_categ = st.columns(4, gap='medium')
    with cols_categ[0]:
        model_key = st.selectbox("Car model", categories['model_key'],
                                 key='car_model', disabled=not api_available)
    with cols_categ[1]:
        car_type = st.selectbox("Car type", categories['car_type'],
                                key='car_type', disabled=not api_available)
    with cols_categ[2]:
        fuel = st.selectbox("Fuel", categories['fuel'], key='fuel',
                            disabled=not api_available)
    with cols_categ[3]:
        paint_color = st.selectbox("Paint color", categories['paint_color'],
                                   key='paint', disabled=not api_available)

    # Boolean variables
    cols_bool = st.columns(7, gap='medium')
    with cols_bool[0]:
        private_parking = st.checkbox('Private parking', value=False,
                                     disabled=not api_available)
    with cols_bool[1]:
        gps = st.checkbox('GPS', value=False, disabled=not api_available)
    with cols_bool[2]:
        air_conditioning = st.checkbox('Air conditioning', value=False,
                                       disabled=not api_available)
    with cols_bool[3]:
        automatic_car = st.checkbox('Automatic car', value=False,
                                    disabled=not api_available)
    with cols_bool[4]:
        getaround_connect = st.checkbox('Getaround connect', value=False,
                                        disabled=not api_available)
    with cols_bool[5]:
        speed_regulator = st.checkbox('Speed regulator', value=False,
                                      disabled=not api_available)
    with cols_bool[6]:
        winter_tires = st.checkbox('Winter tires', value=False,
                                   disabled=not api_available)

    # Quantitative variables
    cols_quant = st.columns(2, gap='large')
    with cols_quant[0]:
        engine_power = st.number_input('Engine power', min_value=0, value=100,
                                       disabled=not api_available)
    with cols_quant[1]:
        mileage = st.number_input('Mileage', min_value=0, value=100_000,
                                  disabled=not api_available)

    st.divider()

    col_model, col_pricing = st.columns([0.5, 1], gap='medium')
    # Select pricing model
    with col_model:
        pricing_model = st.selectbox(
            "Pricing model", pricing_models, key='pricing_model',
            disabled=not api_available)
    # Pricing recommendation
    with col_pricing:
        rental_price_container = st.empty()


# =============================================================================
# Dynamically changing content
# =============================================================================

with delay_info_container:
    delay_info_df = update_delay_info(df, p_cancel, time_bins,
                                      delay_info_df, rental_delay)
    delay_info_table = st.table(delay_info_df)

if api_available:
    data = {"model_key": model_key,
            "mileage": mileage,
            "engine_power": engine_power,
            "fuel": fuel,
            "paint_color": paint_color,
            "car_type": car_type,
            "private_parking_available": private_parking,
            "has_gps": gps,
            "has_air_conditioning": air_conditioning,
            "automatic_car": automatic_car,
            "has_getaround_connect": getaround_connect,
            "has_speed_regulator": speed_regulator,
            "winter_tires": winter_tires}
    rental_price = 100

    ## fetch a new rental price only if necessary
    if (pricing_model != st.session_state['curr_model']
        or data != st.session_state['curr_data']):
        rental_price = get_pricing(API_URL, pricing_model, data)
        st.session_state['curr_model'] = pricing_model
        st.session_state['curr_data'] = data

    with rental_price_container:
        st.html('<p style="font-size:20px;">Recommended price : '
                f'{rental_price:.2f}</p>')

