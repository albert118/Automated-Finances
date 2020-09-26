"""
.. module:: payslips
    :platform: Unix, Windows
    :synopsis: Paycheck integrations for custom pdf and APIs.
.. moduleauthor:: Albert Ferguson <albertferguson118@gmail.com>
"""

# core
from src.core import environConfig, graphing, images, timeManips

# third party libs
import pandas as pd
from pandas import Timestamp
import numpy as np
from tabula import read_pdf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# python core
from datetime import datetime
import math
import os
import traceback
import warnings
import sys

def Payslips_getPayslips(basedir: str, subfolder: str, true_col_header_index=5, filename=str(datetime.strftime(datetime.today(), "%d-%m-%Y")+".pdf")) -> (pd.DataFrame, pd.DataFrame):
        """Retreive the payslip pdf and create aggregate and latest_week frames.
        
        Convert "structured" pdf to frames for easy use later, this is lots of *icky* scraping/conversion code.
        
        **Args:**
            payslip_name(str): The name of the file to scrape. Default is the download name default.

			true_col_header_index(int): This value is used to relatively find further dataframes from the pdf.\
				The row index where column titles are actually located. This overrides the default behaviour of\
				tabula guessing where this would be otherwise (and being wrong typically).\
				
				**Yes this is a magic number.**\
				.. note:: No it isn't tested for everything, only for my example with ADP.\
				Inspect this value yourself if the data is incorrectly parsed.\

        **Notes:**
            .. note:: This function is intended to call up the latest payslip for\
                weekly displays, the stats function for income then aggregates data for\
                longer timeframes.\

        **Returns:**
            latest_week_income(pandas.DataFrame): The the dataframe with hourly data, commissions and deductions.\
                ['Description_Hours', 'Rate', 'Hours', 'Value_Hours', 'Description_Other', 'Tax_Ind', 'Value_Other']\

            aggregate_income(pandas.DataFrame): The aggregate income values.\
                ['Gross', 'Taxable Income', 'Post Tax Allows/Deds', 'Tax', 'NET']\
        """

        ########################################################################
        # Internal Utilities
        ########################################################################
        
        def _rename_headers(dataframe, header_index, cols_default_headers):
            """ Rename the column headers from default guesses to the correct values.
            
            .. note:: Also performs some housekeeping by reindexing and dropping the header row. 
            
            .. note:: Due to the nature of separating a frame like this, it is possible to create duplicate\
                header titles if _split_merged_columns is applied next, keep this in mind.\
            """

            try:
                i = 0                          
                for col in cols_default_headers:
                    dataframe.rename(columns={col:str(dataframe.loc[header_index, col])}, inplace=True, copy=False)
                    i -=- 1

                dataframe = dataframe.drop(header_index)
                row_id = list(range(len(dataframe)))
                dataframe["row_id"] = row_id
                dataframe.set_index("row_id", inplace=True)
                
                if "Tax Ind" in dataframe.columns:
                    dataframe.rename(columns={"Tax Ind": "Tax_Ind"}, inplace=True, copy=False)
                
                if np.NaN in dataframe.columns:
                    dataframe = dataframe.drop(np.NaN, axis=1)
                return dataframe

            except TypeError:
                print("The header index was not correctly calculated, please check the header for the frame manually.\n")
                traceback.print_exc()
                traceback.print_stack()
                return

            except Exception as e:
                print("An unknown exception occured in renaming the headers of the frame.\n")
                print(type(e), '\n')
                print('Current frame:\n', dataframe, '\n')
                input()
                return 

        def _split_merged_columns(dataframe) -> pd.DataFrame:
            hdr_vals = dataframe.columns.tolist() # check first row 'splittable'
            idxs_added = []

            i = 0
            # start by splitting column names and inserting blank columns ready for data
            for val in hdr_vals:
                if ' ' in str(val):
                    new_hdrs = val.split()
                    # insert a new column at this position with NaN type values
                    try:
                        dataframe.insert(i + 1, new_hdrs[1], np.NaN)
                    except ValueError:
                        # Duplicate case, we had a duplicate to insert!
                        # add _Other to distinguish it
                        dataframe.insert(i + 1, new_hdrs[1] + '_Other', np.NaN)

                    # rename the current column and record the insertion idx
                    # edge case, possible that column renamed to existing name
                    if new_hdrs[0] in dataframe.columns.values:
                        idx = dataframe.columns.values.tolist().index(new_hdrs[0])
                        dataframe.columns.values[idx] = str(new_hdrs[0] + '_Other')
                        dataframe.columns.values[i] = new_hdrs[0]
                    else:
                        dataframe.columns.values[i] = new_hdrs[0]
                    
                    idxs_added.append(i)
                    i -=- 2 # skip the col we just made
                else:
                    i -=- 1 # not splitable, skip

            # now split the vals in cols tracked by idxs_added
            # and perform type conversion and string formatting
            for i in range(len(dataframe)):
                # idxs_added tracked which cols need data_points split to vals
                for idx in idxs_added:
                    data_point = dataframe.iloc[i, idx]
                    if type(data_point) == float:
                        continue # skip nan types		
                    vals = data_point.split()

                    try:
                        string_val = ''
                        # val is unknown combination of strings, floats (maybe nan) vals
                        length = len(vals)
                        j = 0
                        while j < length:
                            try:
                                vals[j]= float(vals[j])
                                j -=- 1
                            except ValueError: 
                                # val was string, apply string formating
                                string_val += ' ' + vals.pop(j).replace('*', '').lower().capitalize()
                                length -= 1
                                
                        if len(string_val) != 0: 
                            # apply final string formating
                            string_val = string_val.strip()

                        if len(vals) > 2:
                            # we dont know what is there then, RuntimeError and hope for the best
                            raise RuntimeError
                        elif len(vals) == 0:
                            vals =tuple([np.nan, string_val])
                        elif len(string_val) == 0:
                            vals = tuple(vals)
                        else:
                            vals = tuple(vals.append(string_val))

                    except Exception as e:
                        # TODO: add logging to log file here
                        # we dont know error, pass and hope it's caught else where
                        print(e)
                        # traceback.print_stack()
                        pass

                    # add the data to the new column
                    dataframe.iloc[i, idx + 1] = vals[1]
                    # then replace the merged values with the single column value
                    dataframe.iloc[i, idx] = vals[0]

            return dataframe.drop(['nan'], axis= 1)

        ########################################################################

        aggregate_income = pd.DataFrame()
        latest_week_income = pd.DataFrame()
        aggregate_income_header_idx = None

        # retrieve the payslip, uses the tabula pdf_reader
        f_dir = os.path.join(basedir, subfolder)
        fn = filename
        payslip_name = os.path.join(f_dir, fn)
        data = read_pdf(payslip_name)[0] # using Tabula for the pdf read out to data (dataframe)

        try:
            data = data.drop(["Unnamed: 0", "Status"], axis=1)
            cols_default_headers = data.columns.values
        except (KeyError, ValueError) as e:
            raise e

        # split the data into new columns where tabula merged them, this must be
        # dynamic as user could work further combinations of work rates, etc...
        for i in range(true_col_header_index, len(data)):
            row_split_check = data.iloc[[i]].isnull()
            if row_split_check.values.any():
                bool_header_NaN_spacing = row_split_check.sum().tolist()

                if bool_header_NaN_spacing.count(0) == 1:
                    # this is the index where we split the data frame from aggregate
                    # and stat's income values, break after saving the new df
                    latest_week_income = data[true_col_header_index:i]
                    aggregate_income = data[i + 1:len(data)]
                    aggregate_income_header_idx = i + 1
                    break

        # use correct titles in row_id = true_col_header_index for column header values
        if latest_week_income.empty or aggregate_income.empty:
            print("A frame was incorrectly initialised.")
            raise ValueError
        else:
            latest_week_income = _rename_headers(
                latest_week_income, true_col_header_index, cols_default_headers)
            aggregate_income = _rename_headers(
                aggregate_income, aggregate_income_header_idx, cols_default_headers)
            
        try:
            # now the frames have been split horizontally, 
            # split vertically where some columns have been merged
            latest_week_income = _split_merged_columns(latest_week_income)
        except Exception  as e:
            print(e, "\nCould not split merged data values of income stats data.")
            pass

        # manually correct the some column titles
        aggregate_income.rename(
            columns={
                aggregate_income.columns[2]: "Pre_Tax_Deds",
                aggregate_income.columns[2]: "Post_Ta_Deds",
            },
            inplace=True,
            copy=False)
        # now we try to remove NaNs from the frames
        # return our corrected frames
        # first elem of col sets type, assume blanks never form in first row
        t_vals = [type(e) for e in list(latest_week_income.loc[0,:])]
        cols = list(latest_week_income.columns.values)
        # latest_week_income = latest_week_income.astype(dict(zip(cols, t_vals)))
        for i, col in enumerate(cols):
            t_type = t_vals[i]
            # determine k: the first idx with NaN in a col
            # i.e. the first True index in a isnull() test
            try:
                k = list(latest_week_income[col].isnull()).index(True)
            except ValueError:
                k = -1

            if k == -1:
                continue
            elif t_type == str:
                latest_week_income.loc[:,col].values[k:] = ''
            elif t_type == float or t_type == np.float64:
                latest_week_income.loc[:,col].values[k:] = 0.0
            else:
                pass

        try:
            # add summative data from latetst_week_income to aggregate_income
            aggregate_income["Total_Hours"] = sum(latest_week_income.Hours) 
        except (AttributeError, Exception) as e:
            pass
        
        return aggregate_income, latest_week_income

