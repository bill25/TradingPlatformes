import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import scipy.optimize as solver
import datetime as dt
from functools import reduce
from portfolioOptimiser.settings import settings
import MetaTrader5 as mt5

from datetime import datetime

class price_extractor:

    def __init__(self, api, companies):
        print('Initialised Price Extractor')
        self.__api = api
        self.__companies = companies
        print("hello :",companies)

        pass

    def get_prices(self,event,timeframe,fromdate,numberofcandles):

        prices = pd.DataFrame()
        symbols = self.__companies['Ticker']
        tmp={}

        for i in symbols:
            # try:


            tmp = mt5.copy_rates_from(i, timeframe, fromdate, numberofcandles)


            # if (i==symbols[0]):
            #     prices=pd.DataFrame()
            #     prices['time'] = pd.to_datetime(tmp['time'], unit='s')
            #     prices.rename({'time': 'Date'}, axis=1, inplace=True)
            #     prices.set_index('Date', inplace=True)
            # shutdown()
            # tmp = web.DataReader(i, self.__api, start_date, end_date)
            # print('Fetched prices for: '+i)
            # print(tmp.head())
            #
            # for col in tmp.columns:
            #     print(col)

            # except:
            #     print('Issue getting prices for: '+i)
            # else:
            prices[i] = tmp[event]

        return prices