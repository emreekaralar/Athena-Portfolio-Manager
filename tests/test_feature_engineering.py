# tests/test_feature_engineering.py

import unittest
import pandas as pd
import numpy as np
from src.feature_engineering.risk_indicators import calculate_returns, calculate_var, calculate_max_drawdown, calculate_sharpe_ratio

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Create sample price data
        dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
        prices = pd.DataFrame({
            'AAPL': [300, 305, 310, 315, 320],
            'MSFT': [150, 152, 154, 156, 158]
        }, index=dates)
        self.returns = calculate_returns(prices)

    def test_calculate_var(self):
        var = calculate_var(self.returns['AAPL'], confidence_level=0.95)
        self.assertIsInstance(var, float)

    def test_calculate_max_drawdown(self):
        cumulative_returns = (1 + self.returns['AAPL']).cumprod()
        max_dd = calculate_max_drawdown(cumulative_returns)
        self.assertIsInstance(max_dd, float)

    def test_calculate_sharpe_ratio(self):
        sharpe = calculate_sharpe_ratio(self.returns['AAPL'], risk_free_rate=0.04)
        self.assertIsInstance(sharpe, float)

if __name__ == '__main__':
    unittest.main()
