# tests/test_main.py

import unittest
from unittest.mock import patch
import sys
from src.main import main

class TestMain(unittest.TestCase):
    @patch('src.main.run_robo_advisor')
    @patch('src.main.start_scheduler')
    def test_run_once(self, mock_start_scheduler, mock_run_robo_advisor):
        # Simulate running without --schedule
        test_args = ['main.py']
        with patch.object(sys, 'argv', test_args):
            main()
            mock_run_robo_advisor.assert_called_once()
            mock_start_scheduler.assert_not_called()

    @patch('src.main.run_robo_advisor')
    @patch('src.main.start_scheduler')
    def test_run_with_schedule(self, mock_start_scheduler, mock_run_robo_advisor):
        # Simulate running with --schedule
        test_args = ['main.py', '--schedule']
        with patch.object(sys, 'argv', test_args):
            main()
            mock_run_robo_advisor.assert_not_called()
            mock_start_scheduler.assert_called_once()

if __name__ == '__main__':
    unittest.main()

#python -m unittest discover -s tests
