import yfinance as yf
from curl_cffi import requests
session = requests.Session(impersonate="chrome")
test = yf.download('^GSPC',session=session)