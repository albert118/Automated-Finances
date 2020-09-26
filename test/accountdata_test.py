import unittest
from unittest import expectedFailure, TestCase
from src.accountdata import AccountData
import pandas as pd
import os
from math import fabs

class TestAccountData(TestCase):
    def setUp(self):
        self.dates_list = ["31/07/2016",
                            "30/06/2016",
                            "29/06/2016",
                            "28/06/2016",
                            "27/06/2016",
                            "26/06/2015",
                            "25/06/2014",
                            "24/06/2013",
                            "23/06/2012",
                            "22/06/2011",
                            "21/06/2010",
                            "20/06/2029",
                            "19/06/2028",
                            "18/06/2027",
                            "17/06/2006",
                            "16/06/2005",
                            "15/06/2004",
                            "14/06/2003",
                            "13/06/2002",
                            "12/12/2001",
                            "11/11/2000",
                            "10/10/1999",
                            "9/09/1998",
                            "8/08/1997",
                            "07/07/1996",
                            "06/06/1995",
                            "05/05/1994",
                            "4/04/1993",
                            "3/03/1992",
                            "02/02/1991",
                            "01/01/1990"
                        ]

        self.tx_list    = [-21.9,
                            -20,
                            8.5,
                            7.7,
                            -21.9,
                            -20,
                            8.5,
                            7.7,
                            -7.71,
                            -5.99,
                            -6.2,
                            -13.45,
                            -5.5,
                            -1078.85,
                            -31.35,
                            93.63,
                            -5,
                            -7.5,
                            -13.45,
                            0.1,
                            0.00001,
                            0,
                            0.0,
                            999999.999999
                            -999999.999999,
                            -54.123456789,
                            26.75,
                            -20,
                            35,
                            -20,
                            -37.69,
                            -269.74
                        ]

        self.desc_list  = ["Transfer to other Bank NetBank Rent 1JohnStreet",
                            "Menulog Pty Ltd Sydney AU AUS Card xx1234 Value Date: 26/06/2020",
                            "TRANSPORTFORNSW OPAL Sydney  AUS Card xx1234 Value Date: 27/06/2020",
                            "Transfer From JOHN SMITH drinks",
                            "Transfer to other Bank NetBank Rent 1JohnStreet",
                            "Menulog Pty Ltd Sydney AU AUS Card xx1234 Value Date: 26/06/2020",
                            "TRANSPORTFORNSW OPAL Sydney  AUS Card xx1234 Value Date: 27/06/2020",
                            "Transfer From JOHN SMITH drinks",
                            "CASH DEPOSIT CBA ATM   Sydney A    NSW 214001   AUS",
                            "ALDI STORES - Sydney    AU Card xx1234",
                            "Spotify P1046A79B2 Sydney AU AUS Card xx1234 Value Date: 25/06/2020",
                            "COLES 0852  Sydney AU AUS Tap and Pay xx5678 Value Date: 25/06/2020",
                            "OPORTO  Sydney Sydney NS AUS Tap and Pay xx5678 Value Date: 25/06/2020",
                            "Direct Debit 279234 PAYPAL AUSTRALIA 1009437198445",
                            "JB HI FI  Sydney NS AUS Tap and Pay xx5678 Value Date: 23/06/2020",
                            "Direct Debit 279234 PAYPAL AUSTRALIA 1009431063682",
                            "Direct Credit 123456 AlchemyTuition AlchemyTu_HYfNVhqW",
                            "SOUL ORIGIN (Sydney NS AUS Tap and Pay xx5678 Value Date: 23/06/2020",
                            "Flower Child Cafe Sydney NS AUS Tap and Pay xx5678 Value Date: 23/06/2020",
                            "OPORTO  Sydney Sydney NS AUS Tap and Pay xx5678 Value Date: 23/06/2020",
                            "UBER   *TRIP Sydney AU AUS Card xx1234 Value Date: 22/06/2020",
                            "Direct Credit 123456 AlchemyTuition AlchemyTu_HYfNVhqW",
                            "UBER TRIP HELP.UBER.CO 14518236738 AU AUS Card xx1234 Value Date: 20/06/2020",
                            "Salary JB HI FI GROUP P ADPR00123   123456",
                            "Menulog Pty Ltd Sydney AU AUS Card xx1234 Value Date: 19/06/2020",
                            "Direct Debit 279234 PAYPAL AUSTRALIA 1009382629358",
                            "Direct Credit 123456 AlchemyTuition AlchemyTu_HYfNVhqW",
                            "Menulog Pty Ltd Sydney AU AUS Card xx1234 Value Date: 17/06/2020",
                            "Direct Credit 123456 AlchemyTuition AlchemyTu_HYfNVhqW",
                            "TRANSPORTFORNSW OPAL Sydney  AUS Card xx1234 Value Date: 18/06/2020",
                            "Direct Debit 279234 PAYPAL AUSTRALIA 1009351503255"
                        ]

        self.currbal_list = [2691.75,
                            2713.65,
                            2733.65,
                            2725.15,
                            2691.75,
                            2713.65,
                            2733.65,
                            2725.15,
                            2717.45,
                            2725.16,
                            2731.15,
                            2737.35,
                            2750.8,
                            2756.3,
                            3835.15,
                            3866.5,
                            3772.87,
                            3777.87,
                            3785.37,
                            3798.82,
                            3814.95,
                            3657.12,
                            3692.31,
                            3584.13,
                            3608.73,
                            3663.123456789,
                            -9999.9999,
                            9999.9999,
                            0,
                            0.0,
                            3641.93
                        ]

        self.test_dict  = {"Date": self.dates_list, "Tx": self.tx_list, "Description":self.desc_list, "Curr_Balance":self.currbal_list}
        self.df         = pd.DataFrame(data=self.test_dict)
        self.account    = AccountData(account_frame=self.df)

        self.assertEqual(len(self.dates_list), len(self.desc_list))
        self.assertEqual(len(self.desc_list), len(self.tx_list))
        self.assertEqual(len(self.currbal_list), len(self.desc_list))
        self.assertTrue(self.df.shape, (len(self.desc_list), len(self.test_dict.keys())))
        
        self.numBadVals_int = 4
        self.Emptydata_dict = {"Date":[], "Tx":[], "Description":[], "Curr_Balance":None}
        
        # TODO generate a local.env file here

    def tearDown(self):
        del self.account
        del self.df
        del self.dates_list
        del self.tx_list
        del self.desc_list
        del self.currbal_list
        del self.test_dict
        del self.numBadVals_int
        del self.Emptydata_dict

    def test_setSavings(self):
        _dataNotSavings                = self.Emptydata_dict
        _dataNotSavings["Date"]        = ["23/06/2020"]
        _dataNotSavings["Tx"]          = [-2000]
        _dataNotSavings["Description"] = ["abc123"]
        self.account.setSavings(_dataNotSavings)

        self.assertNotIn(_dataNotSavings["Date"][0], self.account.savings.Date.tolist())
        self.assertNotIn(_dataNotSavings["Tx"][0], self.account.savings.Value.tolist())
        self.assertNotIn(_dataNotSavings["Description"][0], self.account.savings.Description.tolist())

        _dataSavings                = self.Emptydata_dict
        _dataSavings["Date"]        = ["23/06/2020"]
        _dataSavings["Tx"]          = [-2000]
        _dataSavings["Description"] = ["xx1234"]
        self.account.setSavings(_dataSavings)

        self.assertIn(_dataSavings["Date"][0], self.account.savings.Date.tolist())
        self.assertIn(fabs(_dataSavings["Tx"][0]), self.account.savings.Value.tolist())
        self.assertIn(_dataSavings["Description"][0], self.account.savings.Description.tolist())

    def test_setExpenditures(self):
        _dataNotExpenditure             = self.Emptydata_dict
        _dataNotExpenditure["Date"]        = ["23/06/2020"]
        _dataNotExpenditure["Tx"]          = [-2000]
        _dataNotExpenditure["Description"] = ["abc123"]
        self.account.setExpenditures(_dataNotExpenditure)

        self.assertNotIn(_dataNotExpenditure["Date"][0], self.account.expenditures.Date.tolist())
        self.assertNotIn(_dataNotExpenditure["Tx"][0], self.account.expenditures.Value.tolist())
        self.assertNotIn(_dataNotExpenditure["Description"][0], self.account.expenditures.Description.tolist())

        _dataExpenditure                = self.Emptydata_dict
        _dataExpenditure["Date"]        = ["23/06/2020"]
        _dataExpenditure["Tx"]          = [-2000]
        _dataExpenditure["Description"] = ["Spotify"]
        self.account.setExpenditures(_dataExpenditure)

        self.assertIn(_dataExpenditure["Date"][0], self.account.expenditures.Date.tolist())
        self.assertIn(_dataExpenditure["Tx"][0], self.account.expenditures.Value.tolist())
        self.assertIn(_dataExpenditure["Description"][0], self.account.expenditures.Description.tolist())

    def test_setIncomes(self):
        _dataNotIncome                = self.Emptydata_dict
        _dataNotIncome["Date"]        = ["23/06/2020"]
        _dataNotIncome["Tx"]          = [-2000]
        _dataNotIncome["Description"] = ["abc123"]
        self.account.setIncomes(_dataNotIncome)

        self.assertNotIn(_dataNotIncome["Date"][0], self.account.incomes.Date.tolist())
        self.assertNotIn(_dataNotIncome["Tx"][0], self.account.incomes.Value.tolist())
        self.assertNotIn(_dataNotIncome["Description"][0], self.account.incomes.Description.tolist())

        _dataIncome                = self.Emptydata_dict
        _dataIncome["Date"]        = ["23/06/2020"]
        _dataIncome["Tx"]          = [2000]
        _dataIncome["Description"] = ["Alchemy"]
        self.account.setIncomes(_dataIncome)

        self.assertIn(_dataIncome["Date"][0], self.account.incomes.Date.tolist())
        self.assertIn(_dataIncome["Tx"][0], self.account.incomes.Value.tolist())
        self.assertIn(_dataIncome["Description"][0], self.account.incomes.Description.tolist())

    @expectedFailure
    def test_accountdata(self):
        """Check the data paths for the account exist and are valid. Check the expenditures, savings and incomes frames exist and have valid data, i.e. that they don't validate with incorrect data and that they checksum correctly.""" 
        
        base_path = os.path.abspath(self.account.BASE_DIR)
        for path_str in self.account.SUB_FOLDERS:
            dataPath_path = os.path.join(base_path, path_str)
            self.assertTrue(os.path.exists(dataPath_path))

        test_dataframe = pd.DataFrame() # dummy df for test instance
        self.assertTrue(type(self.account.expenditures), type(test_dataframe))
        self.assertTrue(type(self.account.savings), type(test_dataframe))
        self.assertTrue(type(self.account.incomes), type(test_dataframe))
        
        with self.subTest(msg="Checking Date dimension of self.account.XXXX.Date"):
            valueNotInList_date = "01/01/1800"
            self.assertNotIn(valueNotInList_date, self.account.expenditures.Date)
            self.assertNotIn(valueNotInList_date, self.account.savings.Date)
            self.assertNotIn(valueNotInList_date, self.account.incomes.Date)
            exp_list = self.account.expenditures.Date.tolist()
            sav_list = self.account.savings.Date.tolist()
            inc_list = self.account.incomes.Date.tolist()
            ctr = 0

            for elem in self.dates_list:
                if elem in exp_list:
                    ctr += 1
                elif elem in sav_list:
                    ctr += 1
                elif elem in inc_list:
                    ctr += 1

            self.assertEqual(ctr, len(self.dates_list))

            del ctr
            del exp_list
            del sav_list
            del inc_list

        with self.subTest(msg="Checking Description dimension of self.account.XXXX.Description"):
            valueNotInList_str = "not in list"
            self.assertNotIn(valueNotInList_str, self.account.expenditures.Description)
            self.assertNotIn(valueNotInList_str, self.account.savings.Description)
            self.assertNotIn(valueNotInList_str, self.account.incomes.Description)
            exp_list = self.account.expenditures.Description.tolist()
            sav_list = self.account.savings.Description.tolist()
            inc_list = self.account.incomes.Description.tolist()
            ctr = 0

            for elem in self.desc_list:
                if elem in exp_list:
                    ctr += 1
                elif elem in sav_list:
                    ctr += 1
                elif elem in inc_list:
                    ctr += 1

            self.assertEqual(ctr, len(self.desc_list))

            del ctr
            del exp_list
            del sav_list
            del inc_list

        with self.subTest(msg="Checking Value dimension of self.account.XXXX.Value"):
            valueNotInList_int = 12345678910
            self.assertNotIn(valueNotInList_int, self.account.expenditures.Value)
            self.assertNotIn(valueNotInList_int, self.account.savings.Value)
            self.assertNotIn(valueNotInList_int, self.account.incomes.Value)
            exp_list = self.account.expenditures.Value.tolist()
            sav_list = self.account.savings.Value.tolist()
            inc_list = self.account.incomes.Value.tolist()
            ctr = 0

            for elem in self.tx_list:
                if elem in exp_list:
                    ctr += 1
                elif elem in sav_list:
                    ctr += 1
                elif elem in inc_list:
                    ctr += 1
            self.assertEqual(ctr, len(self.tx_list))

            del ctr
            del exp_list
            del sav_list
            del inc_list

    @expectedFailure
    def test_accountdataCategories(self):
        # if not equal, we must be missing categories in local.env config
        totalTxs_int = self.account.expenditures.shape[0] + self.account.savings.shape[0] + self.account.incomes.shape[0]
        self.assertEqual(totalTxs_int, len(self.tx_list))

if __name__ == '__main__':
    unittest.main()
