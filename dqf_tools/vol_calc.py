import pandas as pd
import numpy as np
import re

class CalculateSigma2:
    """
    A class to calculate the variance of VIX based on option data.

    Attributes:
        option (pd.DataFrame): The raw options data.
        call_option (pd.DataFrame): Filtered call options.
        put_option (pd.DataFrame): Filtered put options.
        atm (pd.Series): The at-the-money (ATM) strike data.
        selected_options (pd.DataFrame): The options selected for VIX calculation.
    """
    def __init__(self, option: pd.DataFrame, interest: float, tau: float):
        """
        Initializes the CalculateSigma2 class.

        Args:
            option (pd.DataFrame): DataFrame containing options data.
            interest (float): The risk-free interest rate.
            tau (float): Time to expiration in years.
        """
        self.option = option
        # self.option['date'] = pd.to_datetime(self.option['date'], format="%Y%m%d")
        # self.option['exdate'] = pd.to_datetime(self.option['exdate'], format="%Y%m%d")

        self.interest = interest
        self.tau = tau
        self.call_option = self.option[self.option['cp_flag'] == 'C'].copy()
        self.put_option = self.option[self.option['cp_flag'] == 'P'].copy()
        self.atm = self.calculate_atm()
        self.selected_options = self.find_strike()
        self.calculate_delta_k()
        self.calculate_sigma2()
    def calculate_atm(self):
        """
        Computes the ATM strike price based on the smallest price difference
        between call and put options.

        Returns:
            float: The ATM strike price.
        """
        self.call_option['average_price_call'] = (self.call_option['best_bid'] + self.call_option['best_offer']) / 2
        self.put_option['average_price_put'] = (self.put_option['best_bid'] + self.put_option['best_offer']) / 2

        merged_df = pd.merge(self.call_option, self.put_option, on='strike_price')
        merged_df['difference'] = (merged_df['average_price_call'] - merged_df['average_price_put']).abs()
        min_diff_row = merged_df.loc[merged_df['difference'].idxmin()]
        return min_diff_row

    def find_strike(self):
        """
        Finds the strike price to be used in VIX calculation.

        Returns:
            pd.DataFrame: Selected option data.
        """
        self.forward_price = self.atm['strike_price'] + np.exp(self.interest * self.tau) * (self.atm['average_price_call'] - self.atm['average_price_put'])
        valid_strikes = self.call_option[self.call_option['strike_price'] <= self.forward_price]['strike_price']
        self.strike = valid_strikes.max()
        self.call_option = self.call_option[self.call_option['strike_price'] >= self.strike]
        self.put_option = self.put_option[self.put_option['strike_price'] <= self.strike]
        call_price = self.call_option.loc[self.call_option["strike_price"] == self.strike, "average_price_call"].values
        put_price = self.put_option.loc[self.put_option["strike_price"] == self.strike, "average_price_put"].values

        average_price = (call_price + put_price) / 2
        average_row = pd.DataFrame({
            "strike_price": self.strike,
            "cp_flag": ["p/c average"],
            "average_price": average_price
        })

        self.selected_options = []
        zero_bid_count = 0
        for _, row in self.call_option.iterrows():
            if row["best_bid"] == 0:
                zero_bid_count += 1
            else:
                zero_bid_count = 0

            if zero_bid_count < 2:
                if row["best_bid"] > 0:
                    self.selected_options.append(row)
            else:
                break
        zero_bid_count = 0
        self.put_option = self.put_option.sort_values(by="strike_price", ascending=False)
        for _, row in self.put_option.iterrows():
            if row["best_bid"] == 0:
                zero_bid_count += 1
            else:
                zero_bid_count = 0

            if zero_bid_count < 2:
                if row["best_bid"] > 0:
                    self.selected_options.append(row)
            else:
                break
        self.selected_options = pd.DataFrame(self.selected_options).sort_values(by="strike_price", ascending=True)
        self.selected_options["average_price"] = self.selected_options["average_price_call"].fillna(
            self.selected_options["average_price_put"])
        self.selected_options = self.selected_options.drop(columns=["average_price_call", "average_price_put"])
        self.selected_options = self.selected_options[self.selected_options["strike_price"] != self.strike]
        self.selected_options = pd.concat([pd.DataFrame(self.selected_options), average_row], ignore_index=True)
        self.selected_options = self.selected_options.sort_values(by="strike_price", ascending=True)
        return self.selected_options

    def calculate_delta_k(self):
        """
            Computes ΔK for each option.
        """
        strikes = self.selected_options["strike_price"].sort_values(ascending=True).values
        delta_K = np.zeros(len(strikes))

        for i in range(len(strikes)):
            if i == 0:
                delta_K[i] = strikes[i + 1] - strikes[i]
            elif i == len(strikes) - 1:
                delta_K[i] = strikes[i] - strikes[i - 1]

            else:
                delta_K[i] = (strikes[i + 1] - strikes[i - 1]) / 2


        self.selected_options["delta_K"] = delta_K

    def calculate_sigma2(self):
        """
            Computes the variance (sigma^2) for VIX calculation.
        """
        self.selected_options["contribution"] = self.selected_options["delta_K"]/self.selected_options["strike_price"]**2 * \
            self.selected_options["average_price"] * np.exp(self.interest * self.tau)
        self.term_1 = self.selected_options["contribution"].sum() * 2 / self.tau
        self.term_2 = (self.forward_price/self.strike - 1) ** 2 / self.tau
        self.sigma2 = self.term_1 - self.term_2


class FindOptions:
    def __init__(self, options: pd.DataFrame, tau: float):
        """
        """
        self.options = options.copy()
        self.tau = tau

        self.options['date'] = pd.to_datetime(self.options['date'], format="%Y%m%d")
        self.options['exdate'] = pd.to_datetime(self.options['exdate'], format="%Y%m%d")


        self.options['days_to_expiration'] = (self.options['exdate'] - self.options['date']).dt.days

        self.results = self.get_results()

    def find_nearest_date(self,data):
        """
        找到**小于或等于** tau 的最大到期日
        :return: 近月到期日和实际到期天数
        """
        valid_options = data[data['days_to_expiration'] >= 7]
        near_term_expiry = valid_options[valid_options['days_to_expiration'] <= self.tau]

        if not near_term_expiry.empty:
            self.nearest_expiry = near_term_expiry.loc[near_term_expiry['days_to_expiration'].idxmax()]
            return self.nearest_expiry['exdate'], self.nearest_expiry['days_to_expiration']

    def find_next_date(self,data):
        """

        """
        valid_options = data[data['days_to_expiration'] >= 7]


        if self.nearest_expiry['days_to_expiration'] == self.tau:
            return 0, 0


        next_term_expiry = valid_options[valid_options['days_to_expiration'] > self.tau]
        if not next_term_expiry.empty:
            next_expiry = next_term_expiry.loc[next_term_expiry['days_to_expiration'].idxmin()]
            return next_expiry['exdate'], next_expiry['days_to_expiration']


    def get_results(self):
        expired_date = pd.DataFrame(columns = ['date', 'near_term_expiry', 'next_term_expiry', 'near_term_days_to_expiry', 'next_term_days_to_expiry'])
        for trade_date, group in self.options.groupby('date'):

            near_term_expiry, near_term_days_to_expiry = self.find_nearest_date(group)

            next_term_expiry, next_term_days_to_expiry = self.find_next_date(group)

            expired_date.loc[len(expired_date)] = [trade_date, near_term_expiry, next_term_expiry, near_term_days_to_expiry, next_term_days_to_expiry]

        return expired_date


class InterestRateCalculator:
    def __init__(self, interest_rate_file: pd.DataFrame, exdate: pd.DataFrame):
        self.interest_rate_file = interest_rate_file.copy()
        self.interest_rate_file[self.interest_rate_file.select_dtypes(include='number').columns] /= 100
        self.interest_rate_file.columns = interest_rate_file.columns.str.strip()
        self.interest_rate_file['Date'] = pd.to_datetime(self.interest_rate_file['Date'], format='%m/%d/%Y')
        self.exdate = exdate
        self.maturity_to_days()
        self.interpolate_interest()
        self.calculate_interest_rate()
    def maturity_to_days(self):
        def convert(label):
            label = label.strip()
            if label == 'Date':
                return label  # 不变
            match = re.match(r'(\d*\.?\d+)\s*(Mo|Yr)', label)
            if match:
                num, unit = match.groups()
                num = float(num)
                if unit == 'Mo':
                    return int(round(num * 30))
                elif unit == 'Yr':
                    return int(round(num * 365))
            return label  # 万一格式不对就原样返回

        new_columns = [convert(col) for col in self.interest_rate_file.columns]
        self.interest_rate_file.columns = new_columns

    def interpolate_interest(self):
        self.interest_rate_file = self.interest_rate_file.set_index('Date')
        self.interest_rate_file = self.interest_rate_file.interpolate(method='linear', axis=1)
        self.interest_rate_file = self.interest_rate_file.reset_index()

    def calculate_interest_rate(self):
        self.merged = pd.merge(self.interest_rate_file, self.exdate, how='right', left_on='Date', right_on='date')
        self.merged['near_term_interest'] = None
        self.merged['next_term_interest'] = None
        maturity_columns = [col for col in self.interest_rate_file.columns if isinstance(col, int)]


        for idx, row in self.merged.iterrows():
            near_days = row['near_term_days_to_expiry']
            next_days = row['next_term_days_to_expiry']
            # 用行来做插值（注意要 drop 非数值列）
            rate_row = row[maturity_columns]
            # 近月利率插值
            near_rate = np.interp(near_days, maturity_columns, rate_row.values.astype(float))
            near_rate = max(0, near_rate)  # 防止负利率
            self.merged.at[idx, 'near_term_interest'] = near_rate
            # 次月利率插值或标记为 False
            if next_days == 0:
                self.merged.at[idx, 'next_term_interest'] = False
            else:
                next_rate = np.interp(next_days, maturity_columns, rate_row.values.astype(float))
                next_rate = max(0, next_rate)  # 防止负利率
                self.merged.at[idx, 'next_term_interest'] = next_rate

class VIXCalculator:
    def __init__(self, options_df, interest_df, tau):
        self.options_df = options_df.copy()
        self.tau = tau
        self.options_df['date'] = pd.to_datetime(options_df['date'], format='%m/%d/%Y')
        self.options_df['exdate'] = pd.to_datetime(options_df['exdate'], format='%m/%d/%Y')
        self.finder = FindOptions(options_df, self.tau)
        self.interest = InterestRateCalculator(interest_df, self.finder.results)
        self.interest_and_term = self.interest.merged

    def calculate_vix(self):
        for _,row in self.interest_and_term.iterrows():
            print(row['date'])
            if row['next_term_days_to_expiry'] == 0:
                selected_options = self.options_df[
                    (self.options_df['date'] == row['date']) & (self.options_df['exdate'] == row['near_term_expiry'])]
                sigma_near = CalculateSigma2(selected_options, row['near_term_interest'],row['near_term_days_to_expiry']/365)
                vix = 100 * np.sqrt(sigma_near.sigma2)
                row['vix'] = vix
                print(row['vix'])
            else:
                selected_options_near = self.options_df[
                    (self.options_df['date'] == row['date']) & (self.options_df['exdate'] == row['near_term_expiry'])]
                selected_options_next = self.options_df[
                    (self.options_df['date'] == row['date']) & (self.options_df['exdate'] == row['next_term_expiry'])]
                sigma_near = CalculateSigma2(selected_options_near, row['near_term_interest'],row['near_term_days_to_expiry']/365)
                sigma_next = CalculateSigma2(selected_options_next, row['next_term_interest'],row['next_term_days_to_expiry']/365)
                weight1 = (row['next_term_days_to_expiry']-self.tau) / (row['next_term_days_to_expiry']-row['near_term_days_to_expiry'])
                weight2 = 1 - weight1
                T1 = row['near_term_days_to_expiry']/365
                T2 = row['next_term_days_to_expiry']/365
                vix = 100 * np.sqrt((sigma_near.sigma2 * weight1 * T1 + sigma_next.sigma2 * weight2 * T2)*(365/self.tau))
                row['vix'] = vix
                print(row['vix'])





