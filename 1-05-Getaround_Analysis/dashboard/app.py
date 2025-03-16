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

## Load data
# df = pd.read_excel('./data/get_around_delay_analysis.xlsx')


## Pricing optimization API available ?
try:
    probe_api(API_URL)
except httpx.ConnectError:
    api_available = False
else:
    api_available = True

api_available = True


## Setup initial session state
if not 'data' in st.session_state:
    df = pd.read_excel('./data/get_around_delay_analysis.xlsx')
    st.session_state['data'] = df
if api_available and not 'car_models' in st.session_state:
    st.session_state['car_models'] = ['a', 'b']

st.session_state['car_models'] = ['a', 'b']
st.session_state['fuels'] = ['a', 'b']
st.session_state['paint_colors'] = ['a', 'b']
st.session_state['car_types'] = ['a', 'b']

# Delay analysis data
df = st.session_state['data']

# Pricing optimization
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



# async def update_df():
#     await asyncio.sleep(1)
#     curr_data = st.session_state['data']
#     extend = rng.random((2, 4))
#     data = np.append(curr_data, extend, axis=0)
#     st.session_state['data'] = data
    
#     st.dataframe(pd.DataFrame(data[-10:]))

## async code for dynamic update
# with st.empty():
#     while True:
#         asyncio.run(update_df())
        # asyncio.sleep(1)
        # future_df = asyncio.run_coroutine_threadsafe(update_df(df), loop)
        # df = future_df.result()
        # st.dataframe(df.iloc[-10:])

# =============================================================================
# Application structure
# =============================================================================

########## App configuration ##########
st.set_page_config(
    page_title='Getaround dashboard',
    page_icon='\U0001f4b8',
    layout='wide'
)


########## App Title and Description ##########
st.title('Getaround dashboard \U0001f4b8')
st.markdown(
    '\U0001f44b Hello there! Welcome to this Getaround car rental dashboard. '
    'The dashboard provides:'
    '- Visualizations and information related to the implementation of '
      'a rental delay.'
    '- A connection to the pricing API to price a car by manually entering '
      'its characteristics')
st.caption('Data provided by [Jedha](https://jedha.co)')


# ########## Sidebar ##########
# with st.sidebar:
#     st.title("Market selection")
#     market_pair = st.selectbox(
#         "Select market", markets, index=markets.index('XBT/USD'),
#         key='market_select')
#     time_interval = st.selectbox(
#         "Select market", list(INTERVALS), index=3,
#         key='interval_select')
# ## 
# asset1, asset2 = asset_pairs[market_pair]
# chart_options = {opt.format(asset1=asset1, asset2=asset2): v
#                  for opt, v in raw_chart_options.items()}
# msummary, morderbook, mtrades, mcrashes, mcharts = get_market_data(
#     asset1, asset2, INTERVALS[time_interval])


# ########## Market summary ##########
# st.divider()
# st.markdown("#### {market_pair} at a glance (last 24h)")
# summary_cols = st.columns(3)
# summary_containers = []
# for col in summary_cols:
#     with col:
#         summary_containers.append(st.empty())


# ########## Market charts ##########
# st.divider()
# st.markdown("#### Market chart")
# col_graphselect, col_crashdisplay = st.columns(2)
# with col_graphselect:
#     top_chart = st.selectbox(
#         "Top chart", list(chart_options), key='top_chart_select')
#     bottom_chart = st.selectbox(
#         "Bottom chart", list(chart_options), index=3,key='bottom_chart_select')
# with col_crashdisplay:
#     crash_prob_select = st.selectbox(
#         "Show crash probability", crash_prob_options, key='display_crash_prob',
#         disabled=not ccp_available)
#     display_crashes = st.toggle(
#         'Display market crashes', value=False, disabled=not ccp_available)
#     if not ccp_available:
#         st.caption(':red[Crash prediction API unavailable]')
# ## setup market charts
# chart_container = st.empty()
# chart = st.session_state['market_chart']
# setup_subplot(chart, chart_options[top_chart], row=1)
# setup_subplot(chart, chart_options[bottom_chart], row=2)
# ## setup crash prob
# display_crash_prob = crash_prob_options[crash_prob_select]
# if display_crash_prob:
#     setup_prob_subplots(chart)

tab_delay, tab_pricing = st.tabs(['delay_analysis', 'car_pricing'])

########## Rental delay tab ##########
with tab_delay:
    st.markdown('#### Car rental delay analysis')
    # slider
    
    # plot
    
    # table
    table_orderbook = st.empty()

    
########## Car pricing tab tab ##########
with tab_pricing:
    st.markdown('#### Car rental pricing')
    
    # Categorical variables
    cols_categ = st.columns(4)
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
    cols_bool = st.columns(6)
    with cols_bool[0]:
        private_parking = st.toggle('Private parking', value=False,
                                    disabled=not api_available)
    with cols_bool[1]:
        gps = st.toggle('GPS', value=False, disabled=not api_available)
    with cols_bool[2]:
        air_conditioning = st.toggle('Air conditioning', value=False,
                                    disabled=not api_available)
    with cols_bool[3]:
        getaround_connect = st.toggle('Getaround connect', value=False,
                                      disabled=not api_available)
    with cols_bool[4]:
        speed_regulator = st.toggle('Speed regulator', value=False,
                                    disabled=not api_available)
    with cols_bool[5]:
        winter_tires = st.toggle('Winter tires', value=False,
                                 disabled=not api_available)
    
    # Quantitative variables
    cols_quant = st.columns(2)
    with cols_quant[0]:
        engine_power = st.toggle('Engine power', value=False,
                                    disabled=not api_available)
    with cols_quant[1]:
        mileage = st.toggle('Mileage', value=False, disabled=not api_available)
    
    # Pricing recommendation
    rental_price_container = st.empty()


# =============================================================================
# 
# =============================================================================

if api_available:
    data = {}
    # rental_price = get_pricing(API_URL, data)
    with rental_price_container:
        st.markdown(f'Recomended price {1}')


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
    
