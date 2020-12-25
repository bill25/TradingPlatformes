import pandas as pd
from pandas_datareader import data as pdr
# import fix_yahoo_finance as yf
import numpy as np
import datetime as dt
from portfolioOptimiser.object_factory import object_factory
from portfolioOptimiser.settings import settings
from datetime import datetime
import MetaTrader5 as mt5


class var():
    def calcule_var(assets,timeframe,numberofcandles,fromdate,weights,initial_investment):



        # Set an initial investment level

        obj_factory = object_factory(settings)

        companie_extractor = obj_factory.get_companies_extractor(assets)
        companies = companie_extractor.get_companies_list()
        price_extractor = obj_factory.get_price_extractor(companies)


        closing_prices = price_extractor.get_prices(settings.PriceEvent,timeframe,fromdate,numberofcandles)
        print (closing_prices)
        returns = closing_prices.pct_change()
        cov_matrix = returns.cov()



        avg_rets = returns.mean()
        print("avrgmean ", avg_rets)

        # Calculate mean returns for portfolio overall,
        # using dot product to
        # normalize individual means against investment weights
        # https://en.wikipedia.org/wiki/Dot_product#:~:targetText=In%20mathematics%2C%20the%20dot%20product,and%20returns%20a%20single%20number.
        port_mean = avg_rets.dot(weights)
        print(port_mean,"port *man")
        # Calculate portfolio standard deviation
        port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        print('portstdev ',port_stdev)


        # Calculate mean of investment
        mean_investment = (1 + port_mean) * initial_investment

        # Calculate standard deviation of investmnet
        stdev_investment = initial_investment * port_stdev



        # Select our confidence interval (I'll choose 95% here)
        conf_level1 = 0.05

        # Using SciPy ppf method to generate values for the
        # inverse cumulative distribution function to a normal distribution
        # Plugging in the mean, standard deviation of our portfolio
        # as calculated above
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        from scipy.stats import norm
        cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)


        var_1d1 = initial_investment - cutoff1
        print("var_1d1=====>",var_1d1)