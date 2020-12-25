

##########################################
#
#
#
# from MetaTrader5 import *
# from datetime import date
# import pandas as pd
# import matplotlib.pyplot as plt
# import MetaTrader5 as mt5
#
# # Initializing MT5 connection
# initialize()
#
#
#
#
# # Create currency watchlist for which correlation matrix is to be plotted
# sym = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','GBPJPY']
#
# # Copying data to dataframe
# d = pd.DataFrame()
# for i in sym:
#      rates = copy_rates_from_pos(i, mt5.TIMEFRAME_M1, 0, 1000)
#      d[i] = [y[4] for y in rates]
#
#
# # Deinitializing MT5 connection
# shutdown()
#
# # Compute Percentage Change
# rets = d.pct_change()
#
# # Compute Correlation
# corr = rets.corr()
#
# # Plot correlation matrix
# plt.figure(figsize=(10, 10))
# plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto')
# plt.colorbar()
# plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
# plt.yticks(range(len(corr)), corr.columns);
# plt.suptitle('FOREX Correlations Heat Map', fontsize=15, fontweight='bold')
# plt.show()
#
#
# # Importing statmodels for cointegration test
# import statsmodels
# from statsmodels.tsa.stattools import coint
#
# x = d['GBPUSD']
# y = d['GBPJPY']
# x = (x-min(x))/(max(x)-min(x))
# y = (y-min(y))/(max(y)-min(y))
#
# score = coint(x, y)
# print('t-statistic: ', score[0], ' p-value: ', score[1])
#
#
#
# # Plotting z-score transformation
# diff_series = (x - y)
# zscore = (diff_series - diff_series.mean()) / diff_series.std()
#
# plt.plot(zscore)
# plt.axhline(2.0, color='red', linestyle='--')
# plt.axhline(-2.0, color='green', linestyle='--')
#
# plt.show()









#################################################################################
#################################################################################




#################################################################################
#################################################################################




# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
"""
Created on Thu Mar 14 16:13:03 2019

@author: dmitrievsky
"""
import MetaTrader5 as mt5
from MetaTrader5 import *
from datetime import datetime
import pandas as pd
# Initializing MT5 connection
from marcketAnalyst.trainer import Train





epochs=[1000]
batch_sizes=[64,124]
learning_ratess=[0.001]
layers=[2]
volume=[1]
train = Train()
# MyCompanies = ['EURUSD','GBPUSD','GBPCAD','CADCHF']
MyCompanies = ['EURUSD']
import matplotlib.pyplot as plt

for epoch in epochs:
    for batch_size in batch_sizes:
        for layer in layers:
            for v in volume:
                for lr in learning_ratess:
                    for currency in MyCompanies:
                         initialize()
                         rates = copy_rates_from_pos(currency, mt5.TIMEFRAME_M10, 0, 10000)
                         # x,y=train.process(rates,"train",1)
                         # plt.title(currency)
                         # plt.hist(y)
                         # plt.show()
                         train.retrain(rates,epoch,layer,batch_size,v,lr,currency)
                         shutdown()
# Copying data to pandas data frame


# Deinitializing MT5 connection




# plt.show()
# stockdata = pd.DataFrame(rates)
# stockdata['time']=pd.to_datetime(rates['time'], unit='s')
# print(stockdata)

# print(stockdata.tick_volume)


# for col in stockdata.columns:
#     print(col)
# train().retrain(candles=rates)
# pour extraire le bon model qui predit le mieux



# PREDICTION OF THE NEWST CANDLE
# initialize()
# model = load_model('volume 1 batch size 64 epochs 1000  layers 2 towLayer  0.3721009327901734  0.9105555415153503')
# for i in range(50000):
#         newest_candle = copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M10, i, 1)
#         x,y=train.process(newest_candle,"train",1)
#         Y = model.predict(x)
#         if Y[0][1] > 0.5:
#             print("prdicted and position ",Y,"  ",i)







#
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#
# trace = go.Ohlc(x=stockdata['time'],
#                 open=stockdata['open'],
#                 high=stockdata['high'],
#                 low=stockdata['low'],
#                 close=stockdata['close'])
#
# data = [trace]
# plot(data)


