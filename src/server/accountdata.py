"""
.. module:: accountdata
    :platform: Unix, Windows
    :synopsis: Data layer management of the automated finance project. 
.. moduleauthor:: Albert Ferguson <albertferguson118@gmail.com>
"""

# core
from src.core import environConfig, graphing, images, timeManips
from src.paychecks import Payslips_getPayslips as getPayslips

# third party libs
import pandas as pd
from pandas import Timestamp
import numpy as np
from tabula import read_pdf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.legend import Legend

# python core
from datetime import datetime
from io import BytesIO
import math
import os
import traceback
import warnings
import sys

CMAP =  plt.get_cmap('Paired') # Global colour map variable

class TxData():
    def __init__(self, value=0, date=Timestamp.today(), description='', catAndSub ='', fileSource = os.path.abspath(''), **kwargs):
        """The transaction data class.

        **Args:**
            value(float):           The transaction value, default 0.
            date(pandas.Timestamp): The time of the transaction. Default current date.
            description(str):       A description of the transaction record.
            catAndSub(str):         A string description of the subcategory. TODO: Make foreign key ref? e.g. `utilities:Electricity`
            fileSource(os.path):    The filepath to the original file for the transaction.

        **Kwargs:**
            data(dict):             A dictionary of data to create the object from.
            TODO: implement.

        **Example:**
            >>> TxData('new transaction', value=20.04)
        """

        if type(date) is not str:
            self.date = date.strftime("%d-%m-%Y")
        else:
            self.date = date

        self.value       = float(value)
        self.description = str(description)
        self.catAndSub   = str(catAndSub)
        self.fileSource  = os.path.abspath(fileSource)
        self.fileHash    = self._genFileHash() # generate a hash code of the original file.

    def _genFileHash(self) -> str:
        # check if file already hashed.
        pass

    def getValue(self) -> float:
        return self.value

    def getDate(self) -> Timestamp:
        return self.date

    def getDescription(self) -> str:
        return self.description

    def getSubCat(self) -> str:
        return self.catAndSub

    def getFileSource(self) -> os.path:
        return self.fileSource

    def getFileHash(self) -> str:
        return self.fileHash

    def toDict(self) -> dict:
        return {'Value': self.getValue(), 
             'Date': self.getDate(),
             'Description': self.getDescription(), 
             'CatAndSub': self.getSubCat(),
             'File Source': self.getFileSource(),
             'File Hash': self.getFileHash()
             }

class AccountData():
    """Track several user finance details and accounts.
    
    This is an object designed to track several finance accounts of the
    user. It aims to combine multiple finance related accounts into one
    location for visualisation and informative statistics. 
    
    **Attributes:**
        incomes(dict): Each key accesses the respective data of the descriptor.
            keys:   ['primary_income', 'supplemental_income', 'investment_income', 'latest_week_income', 'aggregate_income']
            
            .. note:: TODO Adjust comments to match upcoming changes on incomes data structs.
            .. note:: TODO Adjust comments of savings and expenditures to update changes to ds'.

        savings(dict):
            key(str):   Description of savings.
            val(list):  elems(timestamp, float):    For described savings.

        expenditures(dict):
            key(str):   Category as defined in local.env and fetched by __init__.
            val(list):  elems(timestamp, float, str):   Time of tx, val of tx, descrip. of tx.

    **Methods:**
        

        getBankData(self) -> pd.DataFrame
            get the income bank data, pass it back to getIncomes for handling.

        setExpenditures(self, data): Set the expenditures transaction data from given data.
    
        setSavings(self, data): Set the savings transaction data from given data.

        setIncomes(self, data):  Set the incomes transaction data from given data.

        display_income_stats(self, n_charts_top = 3, figsize=(10,10)): 

        display_savings_stats(self, figsize=(10,10)):

        display_expenditure_stats(self, figsize=(10,10)): 

    **Example:**
        Call the charts then display several charts on the named categories

        >>> import pandas as pd
        >>> import accountdata as d
        >>> df = pd.read_csv("CSVData.csv", names=["Date","Tx", "Description", "Curr_Balance"])
        >>> a = d.AccountData(account_frame=df)
        >>> from report import Report as r
        # generates the output in the current directory with filename 'Personal Finance Report'
        >>> r(a)
    """ 

    def __init__(self, **kwargs):
        """Track several user finance details and accounts.

        **Kwargs:**     
            *optional kwarg overrides to add these for testing, etc.*

            account_frame(pandas.DataFrame):    Banking data input.
            payslip_frame(pandas.DataFrame):    Payslip data input.

        **Example:**
            >>> import pandas as pd
            >>> import dataframe_worker as w
            >>> df = pd.read_csv("CSVData.csv", names=["Date","Tx", "Description", "Curr_Balance"])
            >>> a = w.account_data(df)
        """

        # ensure safe env on account object instantiation
        env = environConfig.safe_environ()
        self.BASE_DIR    = env("PARENT_DIR")
        self.SUB_FOLDERS = env.list("DATA_SRCS")

        ########################################################################
        # Categories and Search terms
        ########################################################################
        
        # TODO : dynamic unpacking of listed vars for categories
        self.INCOMES = {
            'primary_income':       env.list("primary_income"), 
            'supplemental_income':  env.list("supplemental_income"), 
            'investment_income':    env.list("investment_income"),
        }

        self.EXPENDITURES = { 
            'utilities':            env.list("UTILITIES"),
            'health':               env.list("HEALTH"),
            'eating_out':           env.list("EATING_OUT"),
            'coffee':               env.list("COFFEE"),
            'subscriptions':        env.list("SUBSCRIPTIONS"),
            'groceries':            env.list("GROCERIES"),
            'shopping':             env.list("SHOPPING"),
            'enterainment':         env.list("ENTERTAINMENT"),
        }

        self.SAVINGS_IDS = env.list("SAVINGS")

        if "account_frame" in kwargs:
            # assign debug frame for testing if in kwargs
            account_frame = kwargs["account_frame"]

        ########################################################################
        # AccountData data structure
        ########################################################################

        self.expenditures = pd.DataFrame()
        self.savings      = pd.DataFrame()
        self.incomes      = pd.DataFrame()

        # Call the sets for expenditures, savings and incomes with all the
        # initial banking data (beauty of this is, more can be plugged in later!).
        self.setExpenditures(account_frame)
        self.setSavings(account_frame)
        self.setIncomes(account_frame)

    ############################################################################
    # Data Ingest
    #   Factories for sifting raw data and returning tx objects to the class.
    ############################################################################
    
    def getBankData(self) -> pd.DataFrame:
        """Retrieve the latest bank data CSV scrape.
        
        **Returns:**
            account_dataframe(pd.DataFrame):    A dataframe representing the account data parsed.
        """
        
        f_dir  = os.path.join(self.BASE_DIR, self.SUB_FOLDERS[0])
        # TODO: This function assumes that there is a CSV from the date that we have run.
        #   this then relies on being called at the correct time and is an easy way to 
        #   create errors. FIX IT
        fn_str = datetime.now().strftime("%d-%m-%Y")+".csv" 
        file   = os.path.join(f_dir, fn_str)

        account_dataframe = pd.read_csv(file, names=["Date","Tx", "Description", "Curr_Balance"])
        # TODO: Check the below methods for possible dataframe duplication.
        # format the account df and perform date-time refactoring
        account_dataframe.Description = account_dataframe.Description.apply(str.upper)
        account_dataframe.Date        = pd.to_datetime(account_dataframe.Date, format="%d/%m/%Y")

        return account_dataframe

    ############################################################################
    # Setters
    ############################################################################

    def setExpenditures(self, data):
        """Set the expenditures transaction data.

        .. warning:: This is a culminative process! Data going in is appended to existing data.

        Search the account frame for the expenditure categories and \
        sub-categories known to exist. Retrieve the tx, val, data and\
        description to create a dictionary of the categories and sub-cat\
        tx objects.

        **Args:**
            data(pd.DataFrame): The data to parse for savings information.

            data(dict):         An alternate data structure. Note: This will\
                                consume further memory as a local copy will \
                                be made to convert the data to pd.DataFrame type.

        **Raises:**
            AttributeError:     If the datatype of of data is incorrect an attribute\
                                error will be raised.\
            ValueError:         If an invalid typecast arises.
        """

        expenditures_list = []

        if type(data) is dict:
            _data = pd.DataFrame(data)
        elif type(data) is pd.DataFrame:
            _data = data
        else:
            _data = None
            raise AttributeError

        for i in range(len(_data)):
            try:
                desc_str = str(_data.loc[i, "Description"])
            except KeyError:
                desc_str = '[GET EXPENDITURES: an error occured in retrieving this description]' 
            
            # iterate through category key values, then iterate through any subcategories foreach.
            for category_str, subcat_list in self.EXPENDITURES.items():
                for subCat_str in subcat_list:
                    idx = desc_str.upper().find(subCat_str.upper()) # INSTRUMENTAL TO 'NOT FINDING' CAT's IS INCLUDING THE STRIP FUNCTION!!
                
                    if idx == -1:
                        continue # wasn't found, skip this iteration.
                    else:
                        try:
                            tx_float = float(_data.loc[i, "Tx"])
                            if tx_float > 0:
                                continue
                        except (KeyError, ValueError):
                            tx_float = 0.0

                        try:
                            filesource_path = _data.loc[i, "Filesource"]
                        except KeyError:
                            filesource_path = ' '

                        date_Timestamp = timeManips.timeManips_strConvert(str(_data.loc[i, "Date"]))
                        txObject_dict = TxData(tx_float, date_Timestamp, desc_str, category_str+':'+subCat_str, filesource_path).toDict()
                        expenditures_list.append(txObject_dict)
                        break
                    break

        self.expenditures = pd.concat([self.expenditures, pd.DataFrame.from_dict(expenditures_list, orient='columns')], axis=0)
        return

    def setSavings(self, data):
        """Set the savings transaction data from given data.

        .. warning:: This is a culminative process! Data going in is appended to existing data.

        Search the account frame for savings id's known to exist. Retrieve the \
        tx val, date and description to create a list of tx objects (dicts).

        **Args:**
            data(pd.DataFrame): The data to parse for savings information.

            data(dict):         An alternate data structure. Note: This will\
                                consume further memory as a local copy will \
                                be made to convert the data to pd.DataFrame type.\
        **Raises:**
            AttributeError:     If the datatype of of data is incorrect an attribute\
                                error will be raised.\
            ValueError:         If an invalid typecast arises.
        """

        savings_list = []

        if type(data) is dict:
            _data = pd.DataFrame(data)
        elif type(data) is pd.DataFrame:
            _data = data
        else:
            _data = None
            raise ValueError

        for i in range(len(_data)):
            try:
                desc_str = str(_data.loc[i, "Description"])
            except KeyError:
                desc_str = '[GET SAVINGS: an error occured in retrieving this description]' 

            try:
                tx_float = float(_data.loc[i, "Tx"])
            except (KeyError, ValueError):
                tx_float = 0.0

            try:
                filesource_path = _data.loc[i, "Filesource"]
            except KeyError:
                filesource_path = ' '

            date_Timestamp = timeManips.timeManips_strConvert(str(_data.loc[i, "Date"]))
            # tx for savings should includes the category (account id typically) ref
            for cat in self.SAVINGS_IDS:
                if cat in desc_str:
                    txObject_dict = TxData(-1*tx_float, date_Timestamp, desc_str, cat+": ", filesource_path).toDict()
                    savings_list.append(txObject_dict)
                    break

        self.savings = pd.concat([self.savings, pd.DataFrame.from_dict(savings_list, orient='columns')], axis=0)
        return

    def setIncomes(self, data):
        """Set the the incomes transaction data.

        .. warning:: This is a culminative process! Data going in is appended to existing data.
        
        **Args:**
            data(pd.DataFrame): The data to parse for savings information.

            data(dict):         An alternate data structure. Note: This will\
                                consume further memory as a local copy will \
                                be made to convert the data to pd.DataFrame type.\

        **Raises:**
            AttributeError:     If the datatype of of data is incorrect an attribute\
                                error will be raised.\
            ValueError:         If an invalid typecast arises.
        """

        incomes_list = []

        if type(data) is dict:
            _data = pd.DataFrame(data)
        elif type(data) is pd.DataFrame:
            _data = data
        else:
            _data = None
            raise AttributeError

        for i in range(len(_data)):
            try:
                desc_str = str(_data.loc[i, "Description"])
            except KeyError:
                desc_str = '[GET INCOMES: an error occured in retrieving this description]'
        
            for category_str, subcat_list in self.INCOMES.items():
                for subCat_str in subcat_list:
                    idx = desc_str.upper().find(subCat_str.upper()) # INSTRUMENTAL TO 'NOT FINDING' CAT's IS INCLUDING THE STRIP FUNCTION!!
                    
                    if idx == -1:
                        continue # wasn't found, skip this iteration.
                    else:
                        try:
                            tx_float = float(_data.loc[i, "Tx"])
                            if tx_float < 0:
                                continue

                        except (KeyError, ValueError):
                            tx_float = 0.0

                        try:
                            filesource_path = _data.loc[i, "Filesource"]
                        except KeyError:
                            filesource_path = ' '
                        
                        date_Timestamp = timeManips.timeManips_strConvert(str(_data.loc[i, "Date"]))
                        txObject_dict = TxData(tx_float, date_Timestamp, desc_str, category_str+':'+subCat_str, filesource_path).toDict()
                        incomes_list.append(txObject_dict)
                        break
                    break
        
        self.incomes = pd.concat([self.incomes, pd.DataFrame.from_dict(incomes_list, orient='columns')], axis=0)
        return

    ############################################################################
    # Displayers
    ############################################################################

    def display_income_stats(self, figsize=(10,10)):
        """ Display some visualisations and print outs of the income data.
    
        **Args:**
            figsize(int tuple): The size of the figure to generate. Default is (10, 10).    
        **Returns:**
            images.image_buffer_to_svg(PIL.image): An SVG PIL image.
        """

        # setup the grids for holding our plots, attach them to the same figure
        # inner_**** are for use with plotting, outer is purely spacing
        # n_charts_top = len(self.incomes)
        fig          = plt.figure(figsize=figsize)
        outer        = gridspec.GridSpec(2, 1, figure=fig)
        inner_top    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        graphRad_int = 1.3
        charLim_int  = 15

        labelsDes_list  = self.incomes.Description.values.tolist()
        labelsDes_list  = [label[:charLim_int] for label in labelsDes_list]
        labelsCat_list  = self.incomes.CatAndSub.values.tolist()
        value_list      = self.incomes.Value.values.tolist()
        labelsCat_dict  = {'unknown expenditures': 0} # tracks total expenditures per category

        # Retrieve a title name for the graph to be made.
        catIdx_int    = labelsCat_list[0].find(":")
        if catIdx_int == -1:
            title_str = "Uknown Income Data"
        else:
            title_str = str(labelsCat_list[0][:catIdx_int])
        
        #####################
        # General Income data
        #####################

        for i in range(len(self.incomes)):
            catIdx_int = labelsCat_list[i].find(":")
            if catIdx_int == -1:
                labelsCat_dict['unknown expenditures'] += value_list[i]
            else:
                cat_str = str(labelsCat_list[i][catIdx_int+1:])
                if cat_str not in labelsCat_dict:
                    labelsCat_dict[cat_str] = value_list[i]
                if cat_str in labelsCat_dict:
                    labelsCat_dict[cat_str] += value_list[i]
                else:
                    labelsCat_dict['unknown expenditures'] += value_list[i]
        if labelsCat_dict['unknown expenditures'] == 0:
            del labelsCat_dict['unknown expenditures']

        currentIncome_ax = fig.add_subplot(inner_top[0, 0]) # this is also one of the cleaner ways to create the axis
        currentIncome_ax.set_prop_cycle(color=[CMAP(j) for j in range(1,10)])
        graphing.Graphing_PieChart(None, list(labelsCat_dict.values()), currentIncome_ax, category=title_str)
        currentIncome_ax.legend(list(labelsCat_dict.keys()), loc="lower right", ncol=1, bbox_to_anchor=(1.04,0.5), bbox_transform=fig.transFigure)
        fig.add_subplot(currentIncome_ax)

        ##############
        # JB Data
        # Single Chart
        ##############

        jbAggregate_dataframe, jbIncomeLatestWeek_dataframe  = getPayslips(self.BASE_DIR, self.SUB_FOLDERS[1])

        # labels
        # hourlyLabels_list        = jbIncomeLatestWeek_dataframe["Description"].values.tolist()
        hourWithCommsLabels_list = jbIncomeLatestWeek_dataframe["Description_Other"].values.tolist()

        # data
        # hourlyData_list        = np.array(jbIncomeLatestWeek_dataframe["Value"].values, dtype=np.float32).tolist()
        hourWithCommsData_list = np.array(jbIncomeLatestWeek_dataframe["Value_Other"].values, dtype=np.float32).tolist()
        incomeTaxData_list     = [np.array(jbAggregate_dataframe["Tax"].values, dtype=np.float32)[0], np.array(jbAggregate_dataframe["NET INCOME"].values, dtype=np.float32)[0]]
        hourWithCommsLabels_list.append("Tax (PAYG)")
        hourWithCommsLabels_list.append("NET Income (Received)")
        hourWithCommsLabels_list = [label[:charLim_int] for label in hourWithCommsLabels_list]

        jbAggregate_ax = fig.add_subplot(inner_bottom[0, 0])
        jbAggregate_ax.set_prop_cycle(color=[CMAP(j) for j in range(1,10)])
        graphing.Graphing_PieChart(None,incomeTaxData_list, jbAggregate_ax, rad=graphRad_int)
        graphing.Graphing_PieChart(None,hourWithCommsData_list, jbAggregate_ax, category="Latest Week JB Hi-Fi Overview", rad=(1-graphRad_int))
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot, BEST EXPLANATION
        jbAggregate_ax.legend(hourWithCommsLabels_list, loc="lower right", bbox_to_anchor=(1,0), bbox_transform=fig.transFigure, ncol=2)
        fig.add_subplot(jbAggregate_ax)

        return images.img_buffer_to_svg(fig)

    def display_savings_stats(self, figsize=(10,10)):
        """Generate the display for savings data, based on bank account drawn data. 
        
        .. note:: TODO Integrate options for REST Super.
        
        **Kwargs:**
            figsize(int tuple): The size of the figure to generate. Default is (10, 10).
        
        **Returns:**
            images.image_buffer_to_svg(PIL.image): An SVG PIL image.
        """
            
        fig = plt.figure(figsize=figsize)
        # Display savings across accounts, bar per acc., i.e. bar figure
        # Trendline of account, with short range projection (1 month)
        #   plot 1 month predic. line
        #   plot 1 month best-case (optimal saving)
        
        # set the display stack of two charts with grid_spec

        # setup the grids for holding our plots, attach them to the same figure
        n_charts_top = len(self.savings)
        outer_grid_spec = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
        disp_top        = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid_spec[0],
                    wspace=0.1, hspace=0.1)
        disp_bottom     = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid_spec[1],
                    wspace=0.1, hspace=0.1)

        # labelsDes_list    = self.savings.Description.values.tolist()
        # labelsCat_list    = self.savings.CatAndSub.values.tolist()
        try:
            dates_list = self.savings.Date.values.tolist()
            value_list = self.savings.Value.values.tolist()
        except AttributeError:
            return

        dates_list.reverse()
        value_list.reverse()
        
        diff_list       = []
        total = 0
        for x in value_list:
            total += x
            diff_list.append(total)

        culmChange_ax = plt.Subplot(fig, disp_top[0])
        culmChange_ax.plot(dates_list, diff_list)
        
        # monthAgg_dataframe = timeManips.timeManips_groupbyTimeFreq(self.savings, time="M")
        # graphing.Graphing_BarChart(list(monthAgg_dataframe.Date.values), list(monthAgg_dataframe.Value.values), culmChange_ax, "Culminative Savings")
        culmChange_ax.set_ylabel("Savings $")
        fig.add_subplot(culmChange_ax)
        plt.setp(culmChange_ax.xaxis.get_majorticklabels(), rotation=45)
        plt.xticks(fontsize=8)

        barChartSavings_ax  = plt.Subplot(fig, disp_bottom[0])
        graphing.Graphing_BarChart(dates_list, value_list, barChartSavings_ax)
        barChartSavings_ax.set_ylabel('Savings $')
        barChartSavings_ax.set_xlabel('Date and Description (dd/mm/yyy)')
        fig.add_subplot(barChartSavings_ax)
        
        culmChange_ax.set_title("Savings Overview")
        plt.xticks(fontsize=8)
        return images.img_buffer_to_svg(fig)

    def display_expenditure_stats(self, figsize=(10,10)):
        """ Display some visualisations and print outs of the income data.

        **Kwargs:**
            figsize(int tuple): The size of the figure to generate. Default is (10, 10).
        
        **Returns:**
            images.image_buffer_to_svg(PIL.image): An SVG PIL image.
        """
                
        # setup the grids for holding our plots, attach them to the same figure
        # inner_**** are for use with plotting, outer is purely spacing
        # colCtr_int   = math.ceil(len(self.expenditures.keys()))
        fig          = plt.figure(figsize=figsize)
        outer        = gridspec.GridSpec(3, 1, figure=fig)
        inner_top    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        inner_middle = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2], wspace=0.1, hspace=0.1)
        colours      = [CMAP(j) for j in range(1,10)]

        labelsCat_dict  = {'unknown expenditures': 0} # tracks total expenditures per category
        labelsCat_list  = self.expenditures.CatAndSub.values
        # dates_list        = self.expenditures.Date.values
        value_list      = self.expenditures.Value.values

        for i in range(len(self.expenditures)):
            catIdx_int = labelsCat_list[i].find(":")
            if catIdx_int == -1:
                labelsCat_dict['unknown expenditures'] += value_list[i]
            else:
                cat_str = str(labelsCat_list[i][:catIdx_int])
                if cat_str not in labelsCat_dict:
                    labelsCat_dict[cat_str] = value_list[i]
                if cat_str in labelsCat_dict:
                    labelsCat_dict[cat_str] += value_list[i]
                else:
                    labelsCat_dict['unknown expenditures'] += value_list[i]
        if labelsCat_dict['unknown expenditures'] == 0:
            del labelsCat_dict['unknown expenditures']

        expendituresPieChart_ax = fig.add_subplot(inner_top[0]) # this is also one of the cleaner ways to create the axis
        expendituresPieChart_ax.set_prop_cycle(color=colours)
        # barchart reverses the data to appear in timeorder. Repeat for pie chart to get the same order of colours
        reversedtoMatchValues_list = list(labelsCat_dict.values()); reversedtoMatchValues_list.reverse()
        reversedtoMatchLabels_list = list(labelsCat_dict.keys()); reversedtoMatchLabels_list.reverse()
        graphing.Graphing_PieChart(reversedtoMatchLabels_list, reversedtoMatchValues_list, expendituresPieChart_ax, category='Expenditure By Category (%)')
        expendituresPieChart_ax.legend().remove()
        fig.add_subplot(expendituresPieChart_ax)

        expendituresTimeChart_ax = fig.add_subplot(inner_middle[0])
        expendituresTimeChart_ax.set_prop_cycle(color=[CMAP(j) for j in range(1,10)])
        expendituresTimeChart_ax.set_ylabel('Expenditure Outgoing ($)')
        monthAgg_dataframe = timeManips.timeManips_groupbyTimeFreq(self.expenditures, time="M")
        graphing.Graphing_TimePlot(list(monthAgg_dataframe.Value.values), list(monthAgg_dataframe.Date.values), expendituresTimeChart_ax, "Expenditure History (yyyy-mm)")
        fig.add_subplot(expendituresTimeChart_ax)

        expendituresBarChart_ax = fig.add_subplot(inner_bottom[0])
        graphing.Graphing_BarChart(list(labelsCat_dict.keys()), list(labelsCat_dict.values()), expendituresBarChart_ax, colours=colours)
        expendituresBarChart_ax.set_ylabel('Expenditure Outgoing ($)')
        expendituresBarChart_ax.set_xlabel('Category of Expenditure')
        plt.suptitle("Expenditure Statistics")
        plt.xticks(rotation=20)
        fig.add_subplot(expendituresBarChart_ax)

        return images.img_buffer_to_svg(fig)
