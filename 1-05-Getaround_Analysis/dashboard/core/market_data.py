# -*- coding: utf-8 -*-
"""
Market data management
"""

import time
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler

from .api import (get_ticker, get_ohlc, get_trades, get_order_book,
                  get_market_crashes, get_crash_prob)
from .utils import numerize

# from api import get_ticker, get_ohlc, get_trades, get_order_book
# from utils import numerize

# BUFSIZE = 2000

# =============================================================================
# 
# =============================================================================

def rolling_sum(arr: np.ndarray, window: int)-> np.ndarray:
    rsum = np.cumsum(arr)
    rsum[window:] -= rsum[:-window]
    return rsum


def close_to_close(close: np.ndarray, window: int)-> np.ndarray:
    """
    Close-to-close volatility estimation. No drift.
    """
    slcr = np.log(close[1:] / close[:-1])**2 # squared log close ratio
    sq_vol = rolling_sum(slcr, window)
    sq_vol[:window-1] /= np.arange(1, window)
    sq_vol[window-1:] /= window
    return np.sqrt(sq_vol)


# =============================================================================
# 
# =============================================================================



class MarketSummary():
    
    def __init__(self, asset1: str, asset2: str):
        self.market_pair = f'{asset1}{asset2}'.upper()
        self.assets = (asset1.upper(), asset2.upper())
        
        self.reftime = 0
        self.refprice = 0
        
        self.update_flag = True
    
    
    def set_flag(self, val: bool):
        self.update_flag = val
    
    
    def get_refprice(self):
        """
        Get reference price for the computation of relative variation over 24h.
        """
        t = time.time()
        if t - self.reftime > 97*15*60:
            _, data = get_ohlc(self.market_pair, 15, since=t-97*15*60)
            self.reftime = data['time'][0]
            self.refprice = p if (p:=data['vwap'][0]) else data['close'][0]
        return self.refprice
    
    
    def summary(self)-> dict[str, list]:
        """
        TODO doc
        """
        # last trade
        _, trade = get_trades(self.market_pair, count=1)
        last_trade = [trade['price'][0],
                      'green' if trade['is_buy'][0] else 'red']
        # % change over 24h
        refprice = self.get_refprice()
        _, ohlc = get_ohlc(self.market_pair, 15, since=time.time())
        currprice = ohlc['vwap'][0] if ohlc['vwap'][0] else ohlc['close'][0]
        pctch = (currprice - refprice) / refprice * 100
        pct_change = [pctch, 'green' if pctch >= 0 else 'red']
        # volume
        ticker = get_ticker(self.market_pair)
        volume = [ticker['volume']['last_24h'],
                  ticker['volume']['last_24h'] * ticker['vwap']['last_24h']]
        
        summary = {'last_trade': last_trade,
                   'pct_change': pct_change,
                   'volume': volume}
        return summary
    
    
    def render_summary(self)-> list[str]:
        summary = self.summary()
        
        a1, a2 = self.assets
        #
        p, c = summary['last_trade']
        last_price = f"LAST PRICE  \n:{c}[**{p} {a2}**]"
        #
        p, c = summary['pct_change']
        pct_change = f"24H CHANGE  \n:{c}[**{p:.2f} %**]"
        # 
        v1, v2 = summary['volume']
        volume = (f"24H VOLUME  \n**{numerize(v1)} :grey[{a1}] "
                  f"{numerize(v2)} :grey[{a2}]**")
        return [last_price, pct_change, volume]
        
        

class MarketOrderBook():
    
    def __init__(self,
                 asset1: str, asset2: str,
                 depth: int,
                 style: str, template: str):
        
        self.market_pair = f'{asset1}{asset2}'.upper()
        self.assets = (asset1.upper(), asset2.upper())
        
        self.depth = depth
        self.style = style
        self.template = template.format(asset1=asset1, asset2=asset2)
        
        self.update_flag = True
    
    
    def set_flag(self, val: bool):
        self.update_flag = val
    
    
    def order_book(self,
                   norders: int = 50,
                   )-> dict[str, dict[str, np.ndarray]]:
        order_book = get_order_book(self.market_pair, norders)
        order_book['asks'].pop('time')
        order_book['bids'].pop('time')
        return order_book
    
    
    def render_order_book(self)-> tuple[str, str]:
        n = self.depth
        order_book = self.order_book(n)
        
        data = np.empty((2*n, 3), dtype=float)
        data[:n, 0] = order_book['asks']['price'][n-1::-1]
        data[n:, 0] = order_book['bids']['price'][:n]
        data[:n, 1] = order_book['asks']['volume'][n-1::-1]
        data[n:, 1] = order_book['bids']['volume'][:n]
        data[:, 2] = data[:, 0] * data[:, 1]
        
        template = self.template.format(*data.ravel())
        
        spread = data[n-1, 0] - data[n, 0]
        rel_spread = 200 * spread / (data[n-1, 0] + data[n, 0])
        spread_str = f'Spread: {spread:.2f} ({rel_spread:.2}%)'
        
        return spread_str, self.style + template



class MarketTrades():
    
    def __init__(self, asset1: str, asset2: str, depth: int):
        self.market_pair = f'{asset1}{asset2}'.upper()
        self.assets = (asset1.upper(), asset2.upper())
        
        self.depth = depth
        
        self.since: int | None = None
        self.trades = {}
        self.update_flag = True
        
        
    def set_flag(self, val: bool):
        self.update_flag = val
        
    
    def update(self):
        last, data = get_trades(self.market_pair, self.since, self.depth)
        if self.since is None: # no data available
            self.since = last
            data.pop('trade_id')
            data.pop('is_market')
            self.trades = data
        else: # data available to be updated
            n = len(data['trade_id'])
            if n == self.depth: # too many trades were missed, start anew
                self.since = None
                self.update()
            elif n == 1: # no update to make
                pass
            else: # partial update
                self.since = last
                for k, v in self.trades.items():
                    v[:1-n] = v[n-1:]
                    v[-n:] = data[k]
    
    
    def render_trades(self)-> Styler:
        styles = [
            {'selector': 'th, td', 'props': 'padding-left: 10px; padding-right: 20px; text-align: right;'},
            {'selector': 'th', 'props': 'font-weight: normal;'},
            {'selector': 'tbody', 'props': 'font-weight: bold;'},
            ]
        styles += [{'selector': f'.row{i}', 'props': 'background-color: green;'}
                   for i, b in enumerate(self.trades['is_buy'][::-1]) if b]
        styles += [{'selector': f'.row{i}', 'props': 'background-color: red;'}
                   for i, b in enumerate(self.trades['is_buy'][::-1]) if not b]
        
        times = self.trades['time'].astype('datetime64[s]')[::-1]
        times = [t.item().strftime('%H:%M:%S UTC') for t in times]
        df = pd.DataFrame({'PRICE': self.trades['price'][::-1],
                           'QUANTITY': self.trades['volume'][::-1],
                           'TIME': times})
        
        style = df.style.set_table_styles(styles)
        
        
        return style



class MarketCharts():
    
    def __init__(self, asset1: str, asset2: str, interval: int):
        self.market_pair = f'{asset1}{asset2}'.upper()
        self.assets = (asset1.upper(), asset2.upper())
        
        self.interval = interval
        
        self.t0: float = 0 # start of data
        self.since: int | None = None
        self.data = {
            'time': np.empty((0,), dtype=np.float64),
            'open': np.empty((0,), dtype=np.float32),
            'high': np.empty((0,), dtype=np.float32),
            'low': np.empty((0,), dtype=np.float32),
            'close': np.empty((0,), dtype=np.float32),
            'volume': np.empty((0,), dtype=np.float64),
            'vwap': np.empty((0,), dtype=np.float64),
            'trades': np.empty((0,), dtype=np.float32),
        }
        
        self.update_flag = True
        

    def set_flag(self, val: bool):
        self.update_flag = val
    
    
    def update(self)-> None:
        last, data = get_ohlc(self.market_pair, self.interval, self.since)
        self.since = last
        if len(data['time']) == 1:
            for k, v in self.data.items():
                v[-1] = data[k][0]
        else:
            self.t0 = data['time'][0]
            for k, v in self.data.items():
                self.data[k] = np.concatenate([v, data[k]])
    
    
    def chart_data(self, chart_type: str)-> dict[str, np.ndarray]:
        time = self.data['time'].astype('datetime64[s]')
        match chart_type:
            case 'ohlc':
                chart_data = {'x': time,
                              'open': self.data['open'],
                              'high': self.data['high'],
                              'low': self.data['low'],
                              'close': self.data['close']}
                return chart_data
            case 'volume1':
                chart_data = {'x': time, 'y': self.data['volume']}
                return chart_data
            case 'volume2':
                volume = self.data['volume'] * self.data['vwap']
                chart_data = {'x': time, 'y': volume}
                return chart_data
            case 'trades':
                chart_data = {'x': time, 'y': self.data['trades']}
                return chart_data
            case 'volatility':
                volatility = close_to_close(self.data['close'], 10)
                chart_data = {'x': time[1:], 'y': volatility}
                return chart_data
            case _:
                raise ValueError('invalid chart data requested')
                
    
    def update_chart(self, fig: go.Figure, chart_type: str, row: int)-> None:
        fig.data[row - 1].update(self.chart_data(chart_type))
    
    

class MarketEvents():
    
    def __init__(self, asset1: str, asset2: str, interval: int):
        self.market_pair = f'{asset1}{asset2}'.upper()
        self.assets = (asset1.upper(), asset2.upper())
        
        self.interval = interval
        
        # self.since: int | None = None
        self.crashes = {'events': np.empty((0, 2), dtype=np.float64),
                        'loss': np.empty((0,), dtype=np.float32)}
        self.surges = {'events': np.empty((0, 2), dtype=np.float64),
                        'loss': np.empty((0,), dtype=np.float32)}
        
        self.crash_prob = {'time': np.empty((0,), dtype=np.float64),
                           'prob': np.empty((0,), dtype=np.float32)}
        self.surges_prob = {'time': np.empty((0,), dtype=np.float64),
                           'prob': np.empty((0,), dtype=np.float32)}
        
        self.update_flag = True
    
    
    def set_flag(self, val: bool):
        self.update_flag = val
    
    
    def update(self, since: int | float = 0)-> None:
        self.crashes = get_market_crashes(
            self.market_pair, since, self.interval)
        
        self.crash_prob = get_crash_prob(
            self.market_pair, since, self.interval)
                
    
    def add_events(self, fig: go.Figure, event: str = 'crash')-> None:
        for start, stop in self.crashes['events'].astype('datetime64[s]'):
            fig.add_vrect(x0=start, x1=stop, fillcolor='red', opacity=0.5,
                          row=1, col=1)
            fig.add_vrect(x0=start, x1=stop, fillcolor='red', opacity=0.5,
                          row=2, col=1)
    
    
    def add_prob(self, fig: go.Figure, event: str = 'crash')-> None:
        prob = {'x': self.crash_prob['time'].astype('datetime64[s]'),
                'y': self.crash_prob['prob']}
        fig.data[2].update(prob)
        fig.data[3].update(prob)
    

    
# =============================================================================
#
# =============================================================================
    

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots



# chart = make_subplots(rows=2, shared_xaxes=True)

# mcharts = MarketCharts('ETH', 'USD', 5)
# mcharts.update()
# dtop = mcharts.chart_data('ohlc')
# chart.add_trace(go.Candlestick(**dtop, name='a'), row=1, col=1)
# dbottom = mcharts.chart_data('volatility')
# chart.add_trace(go.Scatter(**dbottom, name='b'), row=2, col=1)

# # chart.show()

# new_data = {'x': [], 'open': [], 'close': [], 'high': [], 'low': []}

# chart.data[0].update





