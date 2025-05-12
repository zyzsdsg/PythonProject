import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from curl_cffi import requests
import ta
from fredapi import Fred
import pandas_datareader as web
session = requests.Session(impersonate="chrome")
# ======================================
# This file is used to download and formalize the data I need to use as the features.
# First step is creating a base class for the data downloader
# ======================================
class BaseDataDownloader(ABC):

    def __init__(self, source_name):
        self.source_name = source_name

    @abstractmethod
    def download(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

# ======================================
# This class is used to download the data using yfinance
# ======================================

class YfinanceDataDownloader(BaseDataDownloader):
    def __init__(self, source_name,ticker:dict,start_date,end_date):
        super().__init__(source_name)
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        self.data = pd.DataFrame()
        self.download()
        # self.save_data()
    def download(self):
        all_data = []
        for label, ticker in self.ticker.items():
            data = yf.download(ticker, self.start_date, self.end_date, session=session)
            data.columns = data.columns.droplevel(1)
            data = data[['Close', 'Volume']].rename(
                columns={'Close': label + '_Close', 'Volume': label + '_Volume'})
            transformed_data = self.transform(data,label)
            all_data.append(transformed_data)
        self.data = pd.concat(all_data, axis=1)
    def transform(self, data: pd.DataFrame, label: str) -> pd.DataFrame:

        data[f"{label}_log_return"] = np.log(data[f"{label}_Close"] / data[f"{label}_Close"].shift(1))
        data[f"{label}_MACD"] = ta.trend.macd(data[f"{label}_Close"])
        data[f"{label}_SMA_30"] = ta.trend.sma_indicator(data[f"{label}_Close"], 30)
        data[f"{label}_Volatility_30"] = data[f"{label}_log_return"].rolling(window=30).std() * np.sqrt(252)
        data[f"{label}_VROC"] = data[f"{label}_Volume"].pct_change(periods=30) * 100

        data = data.dropna()
        return data
    def save_data(self):
        self.data.to_csv('../data/'+self.source_name+'.csv')


start_date = datetime(2004, 11, 1)
end_date = datetime(2025, 4, 30)

# ======================================
# The class behind is used to download data from FRED using pandas data reader
# ======================================
class FredDataDownloader(BaseDataDownloader):
    def __init__(self, source_name,ticker:list,start_date,end_date):
        super().__init__(source_name)
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.download()
        self.transform()

    def download(self):

        self.data = web.get_data_fred(self.ticker, self.start_date, self.end_date)

    def transform(self, *args, **kwargs):
        pass



macro_indicators = [
    "EMRATIO", "PAYEMS", "LES1252881600Q", "USFIRE", "LNU02026620",
    "U6RATENSA", "RECPROUSM156N", "CES0500000011", "CCSA", "OPHNFB",
    "B4701C0A222NBEA", "BKFTTLA641N", "AWHAE", "LBSSA36", "LBSSA17",
    "LNU02026619", "BABATOTALNSAUS", "GDPC1", "GFDEGDQ188S", "M2V",
    "A939RX0Q048SBEA", "GDPPOT", "CP", "M1V", "NETEXP", "DPCERD3Q086SBEA",
    "PCDG", "UMCSENT", "MICH", "CSCICP03USM665S", "BSCICP03USM665S",
    "INDPRO", "IPMAN", "DGORDER", "NEWORDER", "AMTMNO", "ADXTNO",
    "ATCGNO", "A33SNO", "A36SNO", "ADEFNO", "AMNMNO", "A36ZNO",
    "AODGNO", "PPIACO"
]

tickers = {
    "apple": "AAPL",
    "google": "GOOG",
    "sp500": "^GSPC",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "alphabet_a": "GOOGL",
    "eli_lilly": "LLY",
    "jpmorgan": "JPM",
    "exxon": "XOM",
    "netflix": "NFLX",
    "costco": "COST",
    "walmart": "WMT",
    "unitedhealth": "UNH",
    "procter_gamble": "PG",
    "johnson_johnson": "JNJ",
    "home_depot": "HD",
    "coca_cola": "KO",
}




stock_downloader = YfinanceDataDownloader('yf',tickers,start_date,end_date)
stock_downloader.data.to_csv('../data/stocks.csv')
macro_downloader= FredDataDownloader('fred', macro_indicators, start_date, end_date)
macro_downloader.data.to_csv('../data/macro.csv')

