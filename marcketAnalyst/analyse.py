from __future__ import division
from datetime import datetime
import matplotlib.pyplot as plt
import os
from portfolioOptimiser.vaue_at_risk import var
import pandas as pd
import fbprophet
import MetaTrader5 as mt5
from portfolioOptimiser.settings import settings
from portfolioOptimiser.main import generate_optimum_portfolio
from portfolioOptimiser.object_factory import object_factory
import pickle
import numpy as np









class marketananalys:



    def getDataFromMetatrader(currency="EURUSD",timeframe=mt5.TIMEFRAME_H1,fromdate=datetime.now(),count=70000):
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        eurusd_rates = mt5.copy_rates_from(currency, timeframe, fromdate, count)
        mt5.shutdown()
        df=pd.DataFrame(data=eurusd_rates)
        df['time']=pd.to_datetime(df['time'], unit='s')
        return df

    def readdatafile(name, path="C:\data"):
        df = pd.read_csv(path + "/" + name)
        # pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
        df['DATE'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
        return df

    # prepar the data for fbprophet
    def datapreparation(dataframe,tick):
        columns=['open', 'high', 'low','close','tick_volume', 'real_volume', 'spread']
        columns.remove(tick)
        dataframe = dataframe.drop(columns, axis=1)
        dataframe[['ds', 'y']] = dataframe[['time', tick]]
        dataframe = dataframe[['ds', 'y']]
        return dataframe

    def runandsavemodel(data):
        m = fbprophet.Prophet()
        m.fit(data)
        return m

    def find_correlated_and_uncorrelated_assets(timeframe,fromdate,numberofcandles):
        correlated_data=pd.DataFrame(columns = ['Indicator', 'Target','Corr','indicatorMov'])
        targets = []
        # instantiate the objects with the settings
        obj_factory = object_factory(settings)
        companie_extractor = obj_factory.get_companies_extractor(settings.AllAssets)
        cp = obj_factory.get_charts_plotter()
        mcs = obj_factory.get_portfolio_generator()
        fr = obj_factory.get_file_repository()
        metricCalculator = obj_factory.get_metrics_calculator()
        # print('1. Get companies')
        companies = companie_extractor.get_companies_list()
        # print("hello companies  :", companies)
        price_extractor = obj_factory.get_price_extractor(companies)
        data = price_extractor.get_prices(settings.PriceEvent,timeframe,fromdate,numberofcandles)
        print(data)
        # matrix=returns.corr()
        for i  in settings.AllAssets:
            closing_prices = data
            print("matrix after ",i," been shifted")
            returns = settings.DailyAssetsReturnsFunction(closing_prices, settings.ReturnType)
            returns[i] = returns[i].shift(-1, axis=0)
            returns = returns.drop(returns.index[-1])
            matrix=returns.corr()

            for index in matrix[i].iteritems():
                if ((index[1] >= settings.cutOfCorr or index[1] <= -1*settings.cutOfCorr ) and index[1]<1):
                    mov=0
                    if (data.iloc[-1][i]>data.iloc[-2][i]):
                        mov=1
                    correlated_data=correlated_data.append({'Indicator' :i,'Target' :index[0],'Corr':index[1],'indicatorMov':mov},ignore_index=True)
                    targets.append(index[0])
        return correlated_data,targets
                    # datatoplot=returns[[index[0],i]]
                    # cp.plot_Towassets_return_points(datatoplot)

        # cp.plot_correlation_matrix(returns)

        # Plot & Save covariance to file
        # cp.plot_correlation_matrix(returns)
        # fr.save_to_file(covariance, 'Covariances')

    def get_optimised_portfolio(assets,timeframe,fromdate,numberofcandles):
        return generate_optimum_portfolio(assets,timeframe,fromdate,numberofcandles)

    def Predict_Assets(assets,timeframe,fromdate,count,periods,type):

        stat = pd.DataFrame(columns=['ds','devise','trend','yhat_upper', 'yhat_lower', 'weekly','weekly_upper','weekly_lower','prophet_UpDown'])
        for i in assets:
            currencydata=marketananalys.getDataFromMetatrader(currency=i,timeframe=timeframe,fromdate=fromdate,count=count)
            currencydatacleaned= marketananalys.datapreparation(currencydata,type)
            model=marketananalys.runandsavemodel(currencydatacleaned)
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            filename = '../models/'+i+'/forecast_model_'+i+str(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))+'.pckl'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as fout:
                pickle.dump(model, fout)

            for x in range (1,periods+1):
                prophet_UpDown = 0
                if forecast.iloc[-x].trend > forecast.iloc[-(x+1)].trend:
                    prophet_UpDown = 1
                stat = stat.append(
                        {'ds': forecast.iloc[-x].ds,
                         'devise':i,
                         'trend': forecast.iloc[-x].trend,
                         'yhat_upper': forecast.iloc[-x].yhat_upper,
                         'yhat_lower': forecast.iloc[-x].yhat_lower,
                         'weekly': forecast.iloc[-x].weekly,
                         'weekly_upper': forecast.iloc[-x].weekly_upper,
                         'weekly_lower': forecast.iloc[-x].weekly_lower,
                         'prophet_UpDown': prophet_UpDown,
                         },ignore_index=True)
            fig=model.plot_components(forecast)
            fig.suptitle(i)
            filename = "../graphs/"+i + "/"+i+".png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, dpi=fig.dpi)


        return stat


mt5.initialize()
timeframe = mt5.TIMEFRAME_D1
numberofcandles = 10
fromdate= datetime(2020,5,26,22,0,0)
print("calculating target and correlated asstes")
df,targets=marketananalys.find_correlated_and_uncorrelated_assets(timeframe,fromdate,numberofcandles)
print(df)
targets=list(set(targets))

todelet=["EURUSD","USDJPY","USDHKD","USDTRY","GBPUSD","USDCHF"]
print("with" ,targets)
for i in todelet:
    if i in targets:
        targets.remove(i)

print("whitout",targets)
timeframe = mt5.TIMEFRAME_H1
numberofcandles = 10000

# print("calculating optimiszed portfolio")
# optimum_portfolio = marketananalys.get_optimised_portfolio(targets,timeframe,fromdate,numberofcandles)
# print(optimum_portfolio,"optimum_portfolio")
# assets=optimum_portfolio[:-3]["Symbol"].tolist()
# weights=np.array(optimum_portfolio[:-3][optimum_portfolio.columns[-1]].tolist())

initial_investment=120






# var.calcule_var(assets,timeframe,numberofcandles,fromdate,weights,initial_investment)

print("calculating prediction")
stat = marketananalys.Predict_Assets(targets, timeframe, fromdate, numberofcandles,5, type='close')


mt5.shutdown()


print(df)
print(stat)
# print("optimum_portfolio ",optimum_portfolio)

# with open(forecast_model_name, 'rb') as fin:
#     m2 = pickle.load(fin)

# print("optimum_portfolio",optimum_portfolio)
# yhat_upper = (sl lower_trend /tp uppertrend)
# yhat_lower = (sl uppertrend /tp lower_trend)
# trend = trend
# ds = ds
# weekly = weekly
# weekly_lower = weekly_lower
# weekly_upper = weekly_upper



