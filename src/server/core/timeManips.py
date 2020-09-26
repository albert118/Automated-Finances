"""
.. module:: timeManips
    :platform: Unix, Windows
    :synopsis: The time manipulation utility functions.
.. moduleauthor:: Albert Ferguson <albertferguson118@gmail.com>
"""


# third party libs
import numpy as np
import pandas as pd

# python core
import math
import os
import sys
import pickle
from datetime import datetime

def timeManips_groupbyTimeFreq(data: pd.DataFrame, time='D'):
    """Reindex for time by retyping. 
    Retype and apply a PeriodIndex, selecting the M (monthly) opt. as a default.\
    Use the groupby function of a dataframe and the column we want to groupby, return a sum\
    
    Info on sorting:\
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html\
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.asfreq.html\
    
    .. note:: This is a destructive method.
    
    **Args:**
        data(pd.DataFrane): A dataframe with the data to index by time.
        time(str):          Options are 'Y', 'M' and 'D'. Defined by pd.PeriodIndex.\
                See: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html\
                See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.PeriodIndex.html\
    
    **Returns:**
        data(pd.DataFrame): The dataframe grouped by time.
    """
    
    try:
        data.Date = pd.to_datetime(data.Date, dayfirst=True, infer_datetime_format=True, errors="ignore")
        data      = data.groupby(pd.PeriodIndex(data.Date, freq=time), axis = 0).sum()
        data.Date = pd.Series(data.index) 
        data.sort_values(by=["Date"], ascending=True, inplace=True, kind="mergesort")
        data.Date = data.Date.dt.to_timestamp() # .to_timestamp must be accessed via .dt class?
    except AttributeError:
        data = pd.to_datetime(data, dayfirst=True, infer_datetime_format=True, errors="ignore")
        data = pd.DataFrame(data).groupby(pd.PeriodIndex(data, freq=time), axis = 0).sum()
        data = pd.Series(data.index)
        data.sort_values(ascending=True, inplace=True,  kind="mergesort")
        data = data.dt.to_timestamp() # .to_timestamp must be accessed via .dt class?
        data = data.tolist()
    finally:
        return data

def timeManips_timestampConvert(data: list):
    """Retype for timestamp type.

    **Args:**
        data(list): The data of time-esque data.
      
    **Returns:**
        data(list): The data correctly typed.
    """

    data = pd.to_datetime(data)
    data.sort_values(ascending=True)
    data = pd.to_datetime(data) # .to_timestamp must be accessed via .dt class?
    data = data.tolist()
    return data

def timeManips_strConvert(value: str) -> str:
    """Convert a string to a datetime formatted string.
    
    **Args:**
        input(str): A string to convert to consistent format.
       
    **Returns:**
        output(str): A datetime string formatted as %d/%m/%Y.\
            Given an invalid string or unknown format, the input is returned as is.
    """

    output = value

    if '/' or '-' not in value:
        return value

    outstyle_str = "%d/%m/%Y"

    try:
        output = datetime.strptime(value, "%d/%m/%Y")
    except ValueError:
        output = datetime.strptime(value, "%d-%m-%Y")
    else:
        output = datetime.strftime(output, outstyle_str)

    return output
