# -*- coding: utf-8 -*-
"""

"""

import json

# import numpy as np
import pandas as pd
import httpx
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from core import (probe_api, ob_template,
#                   MarketSummary, MarketOrderBook, MarketTrades,
#                   MarketCharts, MarketEvents)
#                   # MarketData, OHLCMarketData)


API_URL = 'http://localhost:4000'

# =============================================================================
# 
# =============================================================================

def probe_api(api_url: str):
    r = httpx.get(f'{api_url}/test', follow_redirects=True)
    res = r.raise_for_status().json()
    return res


def get_pricing(api_url: str,
                data: dict[str, bool | float | str])-> float:
    r = httpx.post(f'{api_url}/predict', follow_redirects=True, data=data)
    res = r.raise_for_status().json()
    return res


# =============================================================================
# 
# =============================================================================

def build_chart()-> go.Figure:
    layout = {
        'height': 700,
        'xaxis_rangeslider_visible': False,
    }
    fig = make_subplots(
        rows=2, shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    fig.update_layout(layout, overwrite=True)
    return fig


# =============================================================================
# App initialization
# =============================================================================

########## Setup initial session state ##########
## Load data
if not 'data' in st.session_state:
    df = pd.read_excel('./data/get_around_delay_analysis.xlsx')
    st.session_state['data'] = df

# ## Pricing optimization API available ?
# if 'api_available' not in st.session_state:
#     try:
#         probe_api(API_URL)
#     except httpx.ConnectError:
#         st.session_state['api_available'] = False
#     else:
#         st.session_state['api_available'] = True

# ## If API available, get categorical variables info
# if api_available and not 'car_models' in st.session_state:
#     st.session_state['car_models'] = ['a', 'b']


st.session_state['pricing_models'] = ['ridge', 'SVM']

st.session_state['car_models'] = ['a', 'b']
st.session_state['fuels'] = ['a', 'b']
st.session_state['paint_colors'] = ['a', 'b']
st.session_state['car_types'] = ['a', 'b']

# Delay analysis data
df = st.session_state['data']

# 
# api_available = st.session_state['api_available']
api_available = True


# Pricing optimization
pricing_models = st.session_state['pricing_models']
car_models = st.session_state['car_models']
fuels = st.session_state['fuels']
paint_colors = st.session_state['paint_colors']
car_types = st.session_state['car_types']


# =============================================================================
# Callbacks
# =============================================================================

def setup_subplot(fig: go.Figure, chart_type: str, row: int)-> go.Figure:
    match chart_type:
        case 'ohlc':
            d = {'x': [], 'open': [], 'close': [], 'high': [], 'low': []}
            trace = go.Candlestick(**d, yaxis='y1')
            # ytitle = ''
        case 'volume1':
            trace = go.Bar(x=[], y=[], yaxis='y1')
        case 'volume2':
            trace = go.Bar(x=[], y=[], yaxis='y1')
        case 'trades':
            trace = go.Bar(x=[], y=[], yaxis='y1')
        case 'volatility':
            trace = go.Scatter(x=[], y=[], yaxis='y1')
        # case '_prob':
        #     trace = go.Scatter(x=[], y=[])
        case _:
            raise ValueError('invalid chart data requested')
    
    if row > len(fig.data):
        fig.add_trace(trace, row=row, col=1, secondary_y=False)
        # fig.update_layout(yaxis1={'side': 'left'})
    else:
        fig.data = []
        fig.add_trace(trace, row=row, col=1, secondary_y=False)
        # fig.update_layout(yaxis={'side': 'left'}, overwrite=True)


def setup_prob_subplots(fig: go.Figure)-> go.Figure:
    trace = go.Scatter(x=[], y=[], yaxis='y2')
    for row in range(1, 3):
        fig.add_trace(trace, row=row, col=1, secondary_y=True)
        # fig.update_layout(yaxis2={'title': 'crash prob.', 'side': 'right'},
        #                   overwrite=True)



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

tab_delay, tab_pricing = st.tabs(['Rental delay analysis', 'Car pricing'])


########## Rental delay tab ##########
with tab_delay:
    st.markdown('#### Car rental delay analysis')
    
    col_plot, col_table = st.columns([1, 0.4], gap='medium')
    
    with col_plot:
        # slider
        st.slider('Rental delay (min)', min_value=0, max_value=1000, value=0)
        
        # plot
        fig = go.Figure()
        trace = go.Scatter(x=[-1, 0, 1], y=[1, 0, 1])
        fig.add_trace(trace)
        st.plotly_chart(fig)
        
        
    
    with col_table:
        # table
        delay_info = st.empty()

    
########## Car pricing tab tab ##########
with tab_pricing:
    st.markdown('#### Car rental pricing')
    
    # Categorical variables
    cols_categ = st.columns(4, gap='medium')
    with cols_categ[0]:
        car_model = st.selectbox("Car model", car_models, key='car_model',
                                 disabled=not api_available)
    with cols_categ[1]:
        car_type = st.selectbox("Car type", car_types, key='car_type',
                                disabled=not api_available)
    with cols_categ[2]:
        fuel = st.selectbox("Fuel", fuels, key='fuel',
                            disabled=not api_available)
    with cols_categ[3]:
        paint = st.selectbox("Paint color", paint_colors, key='paint',
                             disabled=not api_available)

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

with delay_info:
    delay_info_table = st.table(['a', 'b'])


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
    
