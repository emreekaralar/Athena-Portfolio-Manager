# tests/test_data_ingestion.py

import unittest
from src.data_ingestion.fetch_data import fetch_historical_data

class TestDataIngestion(unittest.TestCase):
    def test_fetch_historical_data(self):
        tickers = ['AAPL', 'MSFT']
        start_date = '2020-01-01'
        end_date = '2020-12-31'
        data = fetch_historical_data(tickers, start_date, end_date)
        self.assertFalse(data.empty)
        self.assertIn('AAPL', data.columns)
        self.assertIn('MSFT', data.columns)

if __name__ == '__main__':
    unittest.main()
