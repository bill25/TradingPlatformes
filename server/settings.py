import numpy as np
import datetime as dt

class settings:

    PriceEvent = 'close'
    ReturnType = 'Geometric'
    Optimisersettings = {}
    OptimiserType = 'OLS'
    CompaniesUrl = 'https://en.wikipedia.org/wiki/NASDAQ-100'#'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    NumberOfPortfolios = 50000#00
    API = 'yahoo'
    YearsToGoBack = 3
    RiskFreeRate = 0
    CompanyFetchMode = "PreFixed" #Auto
    MyCompanies = ['EURUSD','GBPUSD','GBPCAD','CADCHF','EURCAD','EURCHF']
    PortfolioOptimisationPath = 'C:/newRep/data/PortfolioOptimisation.xlsx'

