import unittest
from unittest import expectedFailure, TestCase
import src.paychecks as paychecks
from src.accountdata import AccountData
import pandas as pd
import os
from datetime import datetime

class TestPaychecks(TestCase):
    def setUp(self):
        self.basedir_path = os.path.join(os.getcwd(), "test")
        fnDate_str        = datetime.strftime(datetime.today(), "%d-%m-%Y")+".pdf"
        fnData_path       = os.path.join(self.basedir_path, fnDate_str)
        
        if not os.path.exists(fnData_path):
            for file in os.listdir(self.basedir_path):
                if file.endswith(".pdf"):
                    os.rename(file, fnDate_str)

        self.assertTrue(os.path.exists(fnData_path))
            
    def tearDown(self):
        del self.basedir_path

    def testOutput(self):
        outputA, outputB = paychecks.Payslips_getPayslips(self.basedir_path, '')
        dummy_frame = pd.DataFrame()

        self.assertEqual(type(outputA), type(outputB))
        self.assertEqual(type(outputA), type(dummy_frame))

        outputAheaders_list = ["Gross", "Taxable Income", "Post_Ta_Deds", "nan", "Tax", "NET INCOME", "Total_Hours"]
        outputBheaders_list = ["Description", "Rate", "Hours", "Value", "Description_Other", "Tax_Ind", "Value_Other"]
        
        self.assertEqual(outputAheaders_list, outputA.columns.tolist())
        self.assertEqual(outputBheaders_list, outputB.columns.tolist())
        
    @expectedFailure
    def testBadInput(self):
        outputA, outputB = paychecks.Payslips_getPayslips('', '')

