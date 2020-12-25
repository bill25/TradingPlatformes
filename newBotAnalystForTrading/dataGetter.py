import pandas as pd
from tradingview_ta import TA_Handler, Interval



class DataGetter:



    def get_data_from_tradingView(self):
        # dataframeSPY = pd.read_csv("data/SPY.csv")
        # dataframeSPY['time'] = pd.to_datetime(dataframeSPY['time'], unit='s')
        # print(dataframeSPY.tail())
        tesla = TA_Handler()
        tesla.set_symbol_as("TSLA")
        tesla.set_exchange_as_crypto_or_stock("NASDAQ")
        tesla.set_screener_as_stock("america")
        tesla.set_interval_as(Interval.INTERVAL_1_DAY)
        print(tesla.get_analysis().summary)




if __name__ == '__main__':
    # ProjectManager().create_repo("test4")
    DataGetter().get_data_from_tradingView()