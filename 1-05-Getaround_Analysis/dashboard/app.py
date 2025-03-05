# -*- coding: utf-8 -*-
"""
requires
- orjson
- httpx

"""

# import asyncio
import json
import time

# import numpy as np
# import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from httpx import ConnectError
from plotly.subplots import make_subplots
# import yfinance as yf

from core import (probe_api, ob_template,
                  MarketSummary, MarketOrderBook, MarketTrades,
                  MarketCharts, MarketEvents)
                  # MarketData, OHLCMarketData)


INTERVALS = {'1 min': 1,
             '5 min': 5,
             '15 min': 15,
             '30 min': 30,
             '1 hour': 60,
             '4 hours': 240,
             '24 hours': 1440,
             '7 days': 10080,
             '14 days': 21600}


TRADES_COUNT = 30 # <= 1000

ORDERS_COUNT = 50 # <= 500
ORDERS_DISPLAY = 5 # <= ORDERS_COUNT
OB_TEMPLATE_PATH = './static/ob_template.html'
OB_TEMPLATE_FORMAT = {'ap': '{{:.8}}', 'aq1': '{{:.8}}', 'aq2': '{{:.4f}}',
                      'bp': '{{:.8}}', 'bq1': '{{:.8}}', 'bq2': '{{:.4f}}',}


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

# Crash prediction API available ?
try:
    probe_api()
except ConnectError:
    ccp_available = False
else:
    ccp_available = True


with open('../data/ohlcvt_data_index.json', 'rt', encoding='utf-8') as f:
    crypto_index = json.load(f)
crypto_codes = crypto_index['codes']
assets = crypto_index['assets']
asset_pairs = {f'{a1}/{a2}': [a1, a2]
               for a1, a2_list in assets.items()
               for a2 in a2_list}
markets = list(asset_pairs)

raw_chart_options = {
    'Market chart (OHLC)': 'ohlc',
    'Volatility': 'volatility',
    'Trade count': 'trades',
    'Volume ({asset1})': 'volume1',
    'Volume ({asset2})': 'volume2',
}
crash_prob_options = {"Don't show": 0,
                      'Within one day': 1}

trades_col_config = {
    'PRICE': st.column_config.NumberColumn(
        'PRICE', disabled=True, format='%.6g'),
    'QUANTITY': st.column_config.NumberColumn(
        'QUANTITY', disabled=True, format='%.2g'),
    'TIME': st.column_config.Column('TIME', disabled=True)}


## Setup initial session state
if not 'Session_initialized_' in st.session_state:
    with open('./init_session_state.json', 'rt', encoding='utf-8') as f:
        init_sstate = json.load(f)
    for state_elt, state_val in init_sstate.items():
        if state_elt not in st.session_state:
            st.session_state[state_elt] = state_val
    
    if 'ob_template' not in st.session_state:
        style, template = ob_template(
            OB_TEMPLATE_PATH, ORDERS_DISPLAY, OB_TEMPLATE_FORMAT)
        st.session_state['ob_style'] = style
        st.session_state['ob_template'] = template
    
    if 'market_chart' not in st.session_state:
        st.session_state['market_chart'] = build_chart()



# =============================================================================
# Callbacks
# =============================================================================


def get_market_data(asset1: str, asset2: str, time_interval: int
                    )-> tuple:
    """
    

    Parameters
    ----------
    market_pair : str
        DESCRIPTION.
    time_interval : int
        DESCRIPTION.

    """
    market_pair = f'{asset1}{asset2}'
    key = (market_pair, time_interval)
    if market_pair not in st.session_state:
        st.session_state[market_pair] = {
            'summary': MarketSummary(asset1, asset2),
            'order_book': MarketOrderBook(
                asset1, asset2, ORDERS_DISPLAY,
                st.session_state['ob_style'], st.session_state['ob_template']),
            'trades': MarketTrades(asset1, asset2, TRADES_COUNT),
        }
        st.session_state[key] = {
            'charts': MarketCharts(asset1, asset2, time_interval),
            'crashes': MarketEvents(asset1, asset2, time_interval),
        }
    elif key not in st.session_state:
        for mdata in st.session_state[market_pair].values():
            mdata.set_flag(True)
        st.session_state[key] = {
            'charts': MarketCharts(asset1, asset2, time_interval),
            'crashes': MarketEvents(asset1, asset2, time_interval),
        }
    else: 
        for mdata in st.session_state[market_pair].values():
            mdata.set_flag(True)
        for mdata in st.session_state[key].values():
            mdata.set_flag(True)
    
    return (st.session_state[market_pair]['summary'],
            st.session_state[market_pair]['order_book'],
            st.session_state[market_pair]['trades'],
            st.session_state[key]['crashes'],
            st.session_state[key]['charts'])


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
    page_title="Crypto Crash Prediction",
    page_icon="\U0001f4b8",
    layout="wide"
)


########## App Title and Description ##########
st.title("Cryptocurrency Tracker \U0001f4b9")
st.markdown(
    "\U0001f44b Hello there! Welcome to this crypto crash prediction app. "
    "We visualize candlestick charts for selected cryptocurrencies.")
st.caption("Data updates dynamically using [Kraken API]"
           "(https://docs.kraken.com/api/docs/rest-api/add-order/).")


########## Sidebar ##########
with st.sidebar:
    st.title("Market selection")
    market_pair = st.selectbox(
        "Select market", markets, index=markets.index('XBT/USD'),
        key='market_select')
    time_interval = st.selectbox(
        "Select market", list(INTERVALS), index=3,
        key='interval_select')
## 
asset1, asset2 = asset_pairs[market_pair]
chart_options = {opt.format(asset1=asset1, asset2=asset2): v
                 for opt, v in raw_chart_options.items()}
msummary, morderbook, mtrades, mcrashes, mcharts = get_market_data(
    asset1, asset2, INTERVALS[time_interval])


########## Market summary ##########
st.divider()
st.markdown(f"#### {market_pair} at a glance (last 24h)")
summary_cols = st.columns(3)
summary_containers = []
for col in summary_cols:
    with col:
        summary_containers.append(st.empty())


########## Market charts ##########
st.divider()
st.markdown("#### Market chart")
col_graphselect, col_crashdisplay = st.columns(2)
with col_graphselect:
    top_chart = st.selectbox(
        "Top chart", list(chart_options), key='top_chart_select')
    bottom_chart = st.selectbox(
        "Bottom chart", list(chart_options), index=3,key='bottom_chart_select')
with col_crashdisplay:
    crash_prob_select = st.selectbox(
        "Show crash probability", crash_prob_options, key='display_crash_prob',
        disabled=not ccp_available)
    display_crashes = st.toggle(
        'Display market crashes', value=False, disabled=not ccp_available)
    if not ccp_available:
        st.caption(':red[Crash prediction API unavailable]')
## setup market charts
chart_container = st.empty()
chart = st.session_state['market_chart']
setup_subplot(chart, chart_options[top_chart], row=1)
setup_subplot(chart, chart_options[bottom_chart], row=2)
## setup crash prob
display_crash_prob = crash_prob_options[crash_prob_select]
if display_crash_prob:
    setup_prob_subplots(chart)

########## Tables tabs ##########
tab_orderbook, tab_trades = st.tabs(['Order book', 'Trades'])
with tab_orderbook:
    st.markdown('#### Order book')
    table_orderbook = st.empty()
    
with tab_trades:
    st.markdown('#### Market trades')
    table_trades = st.empty()


# =============================================================================
# 
# =============================================================================

# while True:
for container, text in zip(summary_containers, msummary.render_summary()):
    with container:
        st.markdown(text)


with chart_container:
    if mcharts.update_flag:
        mcharts.update()
        mcharts.set_flag(False)
    mcharts.update_chart(chart, chart_options[top_chart], row=1)
    mcharts.update_chart(chart, chart_options[bottom_chart], row=2)
    
    if ccp_available and mcrashes.update_flag:
        mcrashes.update(since=mcharts.t0)
        mcrashes.set_flag(False)
    if display_crashes:
        mcrashes.add_events(chart)
    if display_crash_prob:
        mcrashes.add_prob(chart)
    
    st.plotly_chart(chart)


spread_str, order_book = morderbook.render_order_book()
with table_orderbook:
    with st.container():
        st.html(order_book)
        st.markdown(spread_str)
with table_trades:
    mtrades.update()
    st.dataframe(mtrades.render_trades(), width=400, hide_index=True,
                 column_config=trades_col_config)
    
