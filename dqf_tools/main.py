import pandas as pd
from dqf_tools.vol_calc import CalculateSigma2, FindOptions, InterestRateCalculator, VIXCalculator
import re



# 读取 Excel 数据
options = pd.read_excel('../data/testvix.xlsx')
options['date'] = pd.to_datetime(options['date'], format="%Y%m%d")
options['exdate'] = pd.to_datetime(options['exdate'], format="%Y%m%d")
options['strike_price'] = options['strike_price']/1000
test = FindOptions(options,30)
interest_file = pd.read_csv('../data/daily-treasury-rates.csv')
interest_rate = InterestRateCalculator(interest_file,test.results)
interest_rate.merged.to_excel('interest_rate.xlsx')

ceshi = VIXCalculator(options,interest_file,30)

ceshi.calculate_vix()