import pandas as pd

stock_data = pd.read_csv('../data/stocks.csv')
stock_data = stock_data.set_index('Date')
macro_data = pd.read_csv('../data/macro.csv')
macro_data = macro_data.rename(columns={'DATE':'Date'})
macro_data = macro_data.shift(1)
macro_data = macro_data.set_index('Date')
cpi_data = pd.read_csv('../data/dataset_2025-05-11T23_56_49.576284853Z_DEFAULT_INTEGRATION_IMF.STA_CPI_3.0.1.csv')
cpi_data = cpi_data.set_index('Date')
cpi_data = cpi_data.shift(1)
data = pd.merge(stock_data, macro_data, on='Date', how = 'outer')
data = pd.merge(data, cpi_data, on='Date', how = 'outer')
data.to_csv('../data/data.csv')

####interest rate