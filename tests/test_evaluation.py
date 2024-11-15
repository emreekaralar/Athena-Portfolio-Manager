# tests/test_evaluation.py

import unittest
import pandas as pd
from src.evaluation.backtesting import display_portfolio_allocation, calculate_risk_metrics

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.tickers = ['AAPL', 'MSFT']
        self.weights = [0.6, 0.4]
        self.returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.005, 0.015, 0.02],
            'MSFT': [0.005, 0.015, 0.01, -0.005, 0.02]
        })

    def test_display_portfolio_allocation(self):
        allocation = display_portfolio_allocation(self.tickers, self.weights)
        self.assertEqual(len(allocation), 2)
        self.assertListEqual(allocation['Ticker'].tolist(), self.tickers)
        self.assertListEqual(allocation['Weight'].tolist(), self.weights)

    def test_calculate_risk_metrics(self):
        risk_metrics = calculate_risk_metrics(self.returns, self.weights, risk_free_rate=0.04)
        self.assertIn('Value at Risk (VaR)', risk_metrics)
        self.assertIn('Max Drawdown', risk_metrics)
        self.assertIn('Sharpe Ratio', risk_metrics)

if __name__ == '__main__':
    unittest.main()
