import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import CubicSpline


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_interest_rates(self):
        """加载并处理利率数据"""
        df = pd.read_csv(self.file_path)

        # 解析日期列
        df.rename(columns=lambda x: x.strip(), inplace=True)  # 移除列名的空格
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df.set_index('Date', inplace=True)

        # 确保列名是数值型的期限（天）
        maturity_mapping = {
            "1 Mo": 30, "2 Mo": 60, "3 Mo": 91, "4 Mo": 120, "6 Mo": 182,
            "1 Yr": 365, "2 Yr": 730, "3 Yr": 1095, "5 Yr": 1825,
            "7 Yr": 2555, "10 Yr": 3650, "20 Yr": 7300, "30 Yr": 10950
        }
        df.columns = [maturity_mapping[col] for col in df.columns]

        # 处理 NaN 值：按行进行线性插值
        df = df.apply(pd.to_numeric, errors='coerce')
        df.interpolate(method='linear', axis=1, inplace=True)  # 按行插值
        df /= 100.0
        return df

    def calculate_interest_rate(self, date: str, days_to_expiry: int):
        """使用 Cubic Spline 插值计算某一天的利率，包含 <30 天外推处理"""
        df = self.load_interest_rates()

        # 确保日期格式正确
        date = pd.to_datetime(date)
        if date not in df.index:
            raise ValueError(f"指定日期 {date} 在数据中不存在。")

        yield_curve = df.loc[date]
        maturity_days = np.array(yield_curve.index)  # 期限 (天)
        interest_rates = np.array(yield_curve.values)  # 对应利率

        # 确保 x 是严格递增的
        sorted_indices = np.argsort(maturity_days)
        maturity_days = maturity_days[sorted_indices]
        interest_rates = interest_rates[sorted_indices]

        if days_to_expiry in maturity_days:
            # 直接返回表中已有的利率
            return np.log(1 + ((1 + yield_curve[days_to_expiry] / 2) ** 2 - 1))

        if days_to_expiry < 30:
            # 处理小于 30 天的外推情况
            t1, CMT1 = maturity_days[0], interest_rates[0]
            t2_idx = np.where(maturity_days > t1)[0][0]  # 找到下一个点
            t2, CMT2 = maturity_days[t2_idx], interest_rates[t2_idx]

            if CMT2 >= CMT1:
                m_lower = (CMT2 - CMT1) / (t2 - t1)
                b_lower = CMT1 - m_lower * t1
                extrapolated_rate = m_lower * days_to_expiry + b_lower
            else:
                extrapolated_rate = CMT1  # 反转情况下固定

            return np.log(1 + ((1 + extrapolated_rate / 2) ** 2 - 1))

        # 使用三次样条插值计算 BEY
        spline = CubicSpline(maturity_days, interest_rates)
        bey_estimate = spline(days_to_expiry)

        # 计算 BEY 上下限
        idx = int(np.searchsorted(maturity_days, days_to_expiry) - 1)
        idx = max(0, min(idx, len(maturity_days) - 2))  # 确保 idx 在合理范围内

        r1, r2 = interest_rates[idx], interest_rates[idx + 1]
        lower_bound = np.min([r1, r2])
        upper_bound = np.max([r1, r2])

        # 限制 BEY 在上下限范围内
        bey_bounded = np.clip(bey_estimate, lower_bound, upper_bound)

        # 计算 APY 并转换为连续复利利率
        apy = (1 + bey_bounded / 2) ** 2 - 1
        risk_free_rate = np.log(1 + apy)

        return risk_free_rate
