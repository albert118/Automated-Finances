import unittest
from unittest import expectedFailure, TestCase
from src.accountdata import AccountData
import pandas as pd
import os
from src.report import Report
from datetime import datetime

class TestReportGeneration(TestCase):
    def setUp(self):
        self.df      = pd.read_csv(os.path.join("test","testData.csv"), names=["Date","Tx", "Description", "Curr_Balance"])
        self.account = AccountData(account_frame=self.df)
    
    def tearDown(self):
        del self.df
        del self.account

    def test_generateReport(self):
        Report(self.account, os.getcwd()) # generates the output in the current directory with filename 'Personal Finance Report'
        filename_path = os.path.join(os.getcwd(), "Personal Finance Report.pdf")
        self.assertTrue(os.path.exists(filename_path))
