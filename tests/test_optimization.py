# tests/test_optimization.py

import unittest
import pandas as pd
import numpy as np
from src.models.optimization import (
    optimize_portfolio_long_short,
    optimize_portfolio_multiple_methods,
    optimize_portfolio_mean_variance,
    optimize_portfolio_min_variance,
    optimize_portfolio_max_sharpe,
    optimize_portfolio_risk_parity
)

class TestOptimizationMethods(unittest.TestCase):
    def setUp(self):
        # Create mock return data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        data = np.random.normal(0.001, 0.02, size=(100, len(tickers)))
        self.returns = pd.DataFrame(data, index=dates, columns=tickers)

    def test_mean_variance_optimization(self):
        weights = optimize_portfolio_mean_variance(self.returns, risk_free_rate=0.04, risk_aversion=1.0)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

    def test_min_variance_optimization(self):
        weights = optimize_portfolio_min_variance(self.returns)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

    def test_max_sharpe_optimization(self):
        weights = optimize_portfolio_max_sharpe(self.returns, risk_free_rate=0.04)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

    def test_risk_parity_optimization(self):
        weights = optimize_portfolio_risk_parity(self.returns)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

    def test_long_short_optimization_max_sharpe(self):
        weights = optimize_portfolio_long_short(self.returns, risk_free_rate=0.04, objective="maximize_sharpe", long_short_ratio=1.0)
        self.assertAlmostEqual(np.sum(weights[weights > 0]), 1.0, places=4)
        self.assertAlmostEqual(np.sum(weights[weights < 0]), -1.0, places=4)

    def test_long_short_optimization_min_risk(self):
        weights = optimize_portfolio_long_short(self.returns, risk_free_rate=0.04, objective="minimize_risk", long_short_ratio=1.0)
        self.assertAlmostEqual(np.sum(weights[weights > 0]), 1.0, places=4)
        self.assertAlmostEqual(np.sum(weights[weights < 0]), -1.0, places=4)

    def test_optimize_portfolio_multiple_methods_mean_variance(self):
        weights = optimize_portfolio_multiple_methods(self.returns, risk_free_rate=0.04, method="mean_variance")
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

    def test_optimize_portfolio_multiple_methods_min_variance(self):
        weights = optimize_portfolio_multiple_methods(self.returns, method="minimum_variance")
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

    def test_optimize_portfolio_multiple_methods_max_sharpe(self):
        weights = optimize_portfolio_multiple_methods(self.returns, risk_free_rate=0.04, method="maximum_sharpe")
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

    def test_optimize_portfolio_multiple_methods_risk_parity(self):
        weights = optimize_portfolio_multiple_methods(self.returns, method="risk_parity")
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

    def test_optimize_portfolio_long_short_invalid_objective(self):
        with self.assertRaises(ValueError):
            optimize_portfolio_long_short(self.returns, objective="invalid_objective")

    def test_optimize_portfolio_multiple_methods_invalid_method(self):
        with self.assertRaises(ValueError):
            optimize_portfolio_multiple_methods(self.returns, method="invalid_method")

if __name__ == '__main__':
    unittest.main()
