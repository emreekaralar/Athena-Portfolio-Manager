# tests/test_models.py

import unittest
import pandas as pd
import numpy as np
from src.models.optimization import optimize_portfolio

class TestModels(unittest.TestCase):
    def setUp(self):
        # Create sample returns data
        dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.005, 0.015, 0.02],
            'MSFT': [0.005, 0.015, 0.01, -0.005, 0.02]
        }, index=dates)
        self.returns = returns

    def test_optimize_portfolio(self):
        weights = optimize_portfolio(self.returns)
        self.assertEqual(len(weights), self.returns.shape[1])
        self.assertAlmostEqual(sum(weights), 1.0, places=4)
        for w in weights:
            self.assertGreaterEqual(w, 0)
            self.assertLessEqual(w, 1)

if __name__ == '__main__':
    unittest.main()
