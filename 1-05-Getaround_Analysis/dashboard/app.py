# -*- coding: utf-8 -*-
"""

"""

import numpy as np
# import pandas as pd
# import httpx
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import (probe_api, get_models, get_categories, get_pricing,
                  prepare_dataset, cancel_prob, cancellation_rates,
                  delay_info_df, update_delay_info)


API_URL = 'http://localhost:4000'

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


########## Setup delay analysis session state ##########
# ## Pricing optimization API available ?
# if 'api_available' not in st.session_state:
#     try:
#         probe_api(API_URL)
#     except httpx.ConnectError:
#         st.session_state['api_available'] = False
#     else:
#         st.session_state['api_available'] = True

# ## If the API is available, get available models
# if api_available and not 'pricing_models' in st.session_state:
#     st.session_state['pricing_models'] = get_models(API_URL)

# ## If the API is available, get categorical variables info
# if api_available and not 'categories' in st.session_state:
#     st.session_state['categories'] = get_categories(API_URL)




st.session_state['pricing_models'] = ['ridge', 'SVM']
st.session_state['categories'] = {'model_key': ['a', 'b'],
                                  'fuel': ['a', 'b'],
                                  'paint_color': ['a', 'b'],
                                  'car_type': ['a', 'b']}

## Delay analysis data
df = st.session_state['dataset']
time_bins = st.session_state['time_bins']
p_cancel = st.session_state['p_cancel']
rental_delays = st.session_state['rental_delays']
cancel_rates = st.session_state['cancellation_rates']
delay_info_df = st.session_state['delay_info_df']


## Pricing optimization
# api_available = st.session_state['api_available']
api_available = True
pricing_models = st.session_state['pricing_models']
categories = st.session_state['categories']



# =============================================================================
# Application structure
# =============================================================================

########## App configuration ##########
st.set_page_config(
    page_title='Getaround dashboard',
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
                                 min_value=-400, max_value=400, value=0)
        
        # Rental delay summary table
        delay_info_container = st.empty()

    

########## Car pricing tab tab ##########
with tab_pricing:
    st.markdown('#### Car rental pricing')
    
    # Categorical variables
    cols_categ = st.columns(4, gap='medium')
    with cols_categ[0]:
        car_model = st.selectbox("Car model", categories['model_key'],
                                 key='car_model', disabled=not api_available)
    with cols_categ[1]:
        car_type = st.selectbox("Car type", categories['car_type'],
                                key='car_type', disabled=not api_available)
    with cols_categ[2]:
        fuel = st.selectbox("Fuel", categories['fuel'], key='fuel',
                            disabled=not api_available)
    with cols_categ[3]:
        paint = st.selectbox("Paint color", categories['paint_color'],
                             key='paint', disabled=not api_available)

    # Boolean variables
    cols_bool = st.columns(6, gap='medium')
    with cols_bool[0]:
        private_parking = st.checkbox('Private parking', value=False,
                                     disabled=not api_available)
    with cols_bool[1]:
        gps = st.checkbox('GPS', value=False, disabled=not api_available)
    with cols_bool[2]:
        air_conditioning = st.checkbox('Air conditioning', value=False,
                                       disabled=not api_available)
    with cols_bool[3]:
        getaround_connect = st.checkbox('Getaround connect', value=False,
                                        disabled=not api_available)
    with cols_bool[4]:
        speed_regulator = st.checkbox('Speed regulator', value=False,
                                      disabled=not api_available)
    with cols_bool[5]:
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
# 
# =============================================================================

with delay_info_container:
    delay_info_df = update_delay_info(df, p_cancel, time_bins,
                                      delay_info_df, rental_delay)
    delay_info_table = st.table(delay_info_df)



if api_available:
    data = {}
    # rental_price = get_pricing(API_URL, data)
    with rental_price_container:
        st.html(f'<p style="font-size:20px;">Recomended price : {1}</p>')


# # while True:
# for container, text in zip(summary_containers, msummary.render_summary()):
#     with container:
#         st.markdown(text)


# with chart_container:
#     if mcharts.update_flag:
#         mcharts.update()
#         mcharts.set_flag(False)
#     mcharts.update_chart(chart, chart_options[top_chart], row=1)
#     mcharts.update_chart(chart, chart_options[bottom_chart], row=2)
    
#     if ccp_available and mcrashes.update_flag:
#         mcrashes.update(since=mcharts.t0)
#         mcrashes.set_flag(False)
#     if display_crashes:
#         mcrashes.add_events(chart)
#     if display_crash_prob:
#         mcrashes.add_prob(chart)
    
#     st.plotly_chart(chart)


# spread_str, order_book = morderbook.render_order_book()
# with table_orderbook:
#     with st.container():
#         st.html(order_book)
#         st.markdown(spread_str)
# with table_trades:
#     mtrades.update()
#     st.dataframe(mtrades.render_trades(), width=400, hide_index=True,
#                  column_config=trades_col_config)
    
