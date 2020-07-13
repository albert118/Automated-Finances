"""
.. module:: accountdata
    :platform: Unix, Windows
    :synopsis: Data layer management of the automated finance project. 
.. moduleauthor:: Albert Ferguson <albertferguson118@gmail.com>

.. note:: Returns of most functions are ByteIO streams.
.. note:: TODO convert all to this where appropriate.
"""

# core
from core import environConfig, graphing, images

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
			value(float):			The transaction value, default 0.
			date(pandas.Timestamp):	The time of the transaction. Default current date.
			description(str):		A description of the transaction record.
			catAndSub(str):			A string description of the subcategory. TODO: Make foreign key ref? e.g. `utilities:Electricity`
			fileSource(os.path):	The filepath to the original file for the transaction.

		**Kwargs:**
			data(dict):				A dictionary of data to create the object from.
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
			keys:	['primary_income', 'supplemental_income', 'investment_income', 'latest_week_income', 'aggregate_income']
			
			.. note:: TODO Adjust comments to match upcoming changes on incomes data structs.
			.. note:: TODO Adjust comments of savings and expenditures to update changes to ds'.

		savings(dict):
			key(str):	Description of savings.
			val(list):	elems(timestamp, float):	For described savings.

		expenditures(dict):
			key(str): 	Category as defined in local.env and fetched by __init__.
			val(list):	elems(timestamp, float, str):	Time of tx, val of tx, descrip. of tx.

	**Methods:**
		

		getBankData(self) -> pd.DataFrame
			get the income bank data, pass it back to getIncomes for handling.

		getPayslips(self, payslip_name='payslip.pdf', true_col_header_index = 5) -> pd.DataFrame
			get the income data from payslip data, pass it back to getIncomes 
			for handling.

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

			account_frame(pandas.DataFrame): 	Banking data input.
			payslip_frame(pandas.DataFrame):	Payslip data input.

		**Example:**
			>>> import pandas as pd
			>>> import dataframe_worker as w
			>>> df = pd.read_csv("CSVData.csv", names=["Date","Tx", "Description", "Curr_Balance"])
			>>> a = w.account_data(df)
		"""

		# ensure safe env on account object instantiation
		env = environConfig.safe_environ()
		self.BASE_DIR = env("PARENT_DIR")
		self.SUB_FOLDERS = env.list("DATA_SRCS")

		########################################################################
		# Categories and Search terms
		########################################################################
		
		# TODO : dynamic unpacking of listed vars for categories
		self.INCOMES = {
			'primary_income': 		env.list("primary_income"), 
			'supplemental_income': 	env.list("supplemental_income"), 
			'investment_income': 	env.list("investment_income"),
		}

		self.EXPENDITURES = { 
			'utilities': 			env.list("UTILITIES"),
			'health': 				env.list("HEALTH"),
			'eating_out': 			env.list("EATING_OUT"),
			'coffee': 				env.list("COFFEE"),
			'subscriptions':		env.list("SUBSCRIPTIONS"),
			'groceries': 			env.list("GROCERIES"),
			'shopping':  			env.list("SHOPPING"),
			'enterainment': 		env.list("ENTERTAINMENT"),
		}

		self.SAVINGS_IDS = [env("ACC_1"), env("ACC_2")]

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

		# income_week_data = self.getPayslips()
		# incomes["latest_week_income"] = income_week_data
		
	############################################################################
	# Data Ingest
	# 	Factories for sifting raw data and returning tx objects to the class.
	############################################################################
	
	def getBankData(self) -> pd.DataFrame:
		"""Retrieve the latest bank data CSV scrape.
		
		**Returns:**
			account_dataframe(pd.DataFrame):	A dataframe representing the account data parsed.
		"""
		
		f_dir  = os.path.join(self.BASE_DIR, self.SUB_FOLDERS[0])
		# TODO: This function assumes that there is a CSV from the date that we have run.
		# 	this then relies on being called at the correct time and is an easy way to 
		#   create errors. FIX IT
		fn_str = datetime.now().strftime("%d-%m-%Y")+".csv" 
		file   = os.path.join(f_dir, fn_str)

		account_dataframe = pd.read_csv(file, names=["Date","Tx", "Description", "Curr_Balance"])
		# TODO: Check the below methods for possible dataframe duplication.
		# format the account df and perform date-time refactoring
		account_dataframe.Description = account_dataframe.Description.apply(str.upper)
		account_dataframe.Date        = pd.to_datetime(account_dataframe.Date, format="%d/%m/%Y")

		return account_dataframe
	
	def getPayslips(self, true_col_header_index = 5) -> pd.DataFrame:
		"""Retreive the payslip pdf and create aggregate and latest_week frames.
		
		Convert "structured" pdf to frames for easy use later, this is lots of
		icky scraping/conversion code.
		
		**Args:**
			payslip_name(str): The name of the file to scrape. Default is the
				download name default. This is usually for testing purposes.

		true_col_header_index(int): This value is used to relatively find further dataframes from the pdf.
			The row index where column titles are actually located. This
			over-rides the default behaviour of tabula guessing where this
			would be otherwise (and being wrong typically).
			
			**Yes this is a magic number.**
			.. note:: No it isn't tested for everything, only for my example with ADP.
			.. note:: Inspect this value yourself if the data is incorrectly parsed.

		**Notes:**
			.. note:: This function is intended to call up the latest payslip for
				weekly displays, the stats function for income then aggregates data for
				longer timeframes.

		**Returns:**
			latest_week_income(pandas.DataFrame): The the dataframe with hourly data, commissions and deductions.
				['Description_Hours', 'Rate', 'Hours', 'Value_Hours', 'Description_Other', 'Tax_Ind', 'Value_Other']

			aggregate_income( pandas.DataFrame): The aggregate income values.
				['Gross', 'Taxable Income', 'Post Tax Allows/Deds', 'Tax', 'NET']
		"""

		########################################################################
		# Internal Utilities
		########################################################################
		
		def _rename_headers(dataframe, header_index, cols_default_headers):
			""" Rename the column headers from default guesses to the correct values.
			
			.. note:: Also performs some housekeeping by reindexing and dropping the header row. 
			
			.. note:: Due to the nature of separating a frame like this, it is possible to create duplicate 
				header titles if _split_merged_columns is applied next, keep this in mind.
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
								
						if len(string_val) is not 0: 
							# apply final string formating
							string_val = string_val.strip()

						if len(vals) > 2:
							# we dont know what is there then, RuntimeError and hope for the best
							raise RuntimeError
						elif len(vals) is 0:
							vals =tuple([np.nan, string_val])
						elif len(string_val) is 0:
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
		f_dir = os.path.join(self.BASE_DIR, self.SUB_FOLDERS[1])
		fn = datetime.now().strftime("%d-%m-%Y")+".pdf"
		payslip_name = os.path.join(f_dir, fn)
		data = read_pdf(payslip_name)[0]

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

				if bool_header_NaN_spacing.count(0) is 1:
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

	############################################################################
	# Setters
	#	
	############################################################################

	def setExpenditures(self, data):
		"""Set the expenditures transaction data.

		.. warning:: This is a culminative process! Data going in is appended to existing data.

		Search the account frame for the expenditure categories and \
		sub-categories known to exist. Retrieve the tx, val, data and\
		description to create a dictionary of the categories and sub-cat\
		tx objects.

		**Args:**
			data(pd.DataFrame):	The data to parse for savings information.

			data(dict):			An alternate data structure. Note: This will\
								consume further memory as a local copy will \
								be made to convert the data to pd.DataFrame type.

		**Raises:**
			AttributeError:		If the datatype of of data is incorrect an attribute\
								error will be raised.\
			ValueError:			If an invalid typecast arises.
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
							date_Timestamp = datetime.strptime(_data.loc[i, "Date"], "%d/%m/%Y")
						except ValueError:
							try:
								date_Timestamp = datetime.strptime(_data.loc[i, "Date"], "%d-%m-%Y")
							except KeyError:
								date_Timestamp = Timestamp.today
						except KeyError:
							date_Timestamp = Timestamp.today

						try:
							filesource_path = _data.loc[i, "Filesource"]
						except KeyError:
							filesource_path = ' '
						
						txObject_dict = TxData(tx_float, date_Timestamp, desc_str, category_str+':'+subCat_str, filesource_path).toDict()
						expenditures_list.append(txObject_dict)
						break
					break

		self.expenditures = pd.concat([self.expenditures, pd.DataFrame(data=expenditures_list)], axis=0)
		return

	def setSavings(self, data):
		"""Set the savings transaction data from given data.

		.. warning:: This is a culminative process! Data going in is appended to existing data.

		Search the account frame for savings id's known to exist. Retrieve the \
		tx val, date and description to create a list of tx objects (dicts).

		**Args:**
			data(pd.DataFrame):	The data to parse for savings information.

			data(dict):			An alternate data structure. Note: This will\
								consume further memory as a local copy will \
								be made to convert the data to pd.DataFrame type.\
		**Raises:**
			AttributeError:		If the datatype of of data is incorrect an attribute\
								error will be raised.\
			ValueError:			If an invalid typecast arises.
		"""

		savings_list = []

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
				desc_str = '[GET SAVINGS: an error occured in retrieving this description]' 

			try:
				tx_float = float(_data.loc[i, "Tx"])
			except (KeyError, ValueError):
				tx_float = 0.0

			try:
				date_Timestamp = datetime.strptime(_data.loc[i, "Date"], "%d/%m/%Y")
			except ValueError:
				try:
					date_Timestamp = datetime.strptime(_data.loc[i, "Date"], "%d-%m-%Y")
				except KeyError:
						date_Timestamp = Timestamp.today
			except KeyError:
				date_Timestamp = Timestamp.today

			try:
				filesource_path = _data.loc[i, "Filesource"]
			except KeyError:
				filesource_path = ' '
		
			# tx for savings should includes the category (account id typically) ref
			for cat in self.SAVINGS_IDS:
				if cat in desc_str:
					txObject_dict = TxData(tx_float, date_Timestamp, desc_str, cat+": ", filesource_path).toDict()
					savings_list.append(txObject_dict)
					break

		self.savings = pd.concat([self.savings, pd.DataFrame(data=savings_list)], axis=0)
		return

	def setIncomes(self, data):
		"""Set the the incomes transaction data.

		.. warning:: This is a culminative process! Data going in is appended to existing data.
		
		**Args:**
			data(pd.DataFrame):	The data to parse for savings information.

			data(dict):			An alternate data structure. Note: This will\
								consume further memory as a local copy will \
								be made to convert the data to pd.DataFrame type.\

		**Raises:**
			AttributeError:		If the datatype of of data is incorrect an attribute\
								error will be raised.\
			ValueError:			If an invalid typecast arises.
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
							date_Timestamp = datetime.strptime(_data.loc[i, "Date"], "%d/%m/%Y")
						except ValueError:
							try:
								date_Timestamp = datetime.strptime(_data.loc[i, "Date"], "%d-%m-%Y")
							except KeyError:
								date_Timestamp = Timestamp.today
						except KeyError:
							date_Timestamp = Timestamp.today

						try:
							filesource_path = _data.loc[i, "Filesource"]
						except KeyError:
							filesource_path = ' '

						txObject_dict = TxData(tx_float, date_Timestamp, desc_str, category_str+':'+subCat_str, filesource_path).toDict()
						incomes_list.append(txObject_dict)
						break
					break
		self.incomes = pd.concat([self.incomes, pd.DataFrame(data=incomes_list)], axis=0)
		return

	############################################################################
	# Displayers
	############################################################################

	def display_income_stats(self, figsize=(10,10)):
		""" Display some visualisations and print outs of the income data.
	
		**Args:**
			figsize(int tuple):	The size of the figure to generate. Default is (10, 10).	
		**Returns:**
			images.image_buffer_to_svg(PIL.image): An SVG PIL image.
		"""

		# setup the grids for holding our plots, attach them to the same figure
		# inner_**** are for use with plotting, outer is purely spacing
		# n_charts_top = len(self.incomes)
		fig          = plt.figure(figsize=figsize)
		outer        = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
		inner_top    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
		inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1],wspace=0.1, hspace=0.1)

		# i = 0

		labelsDes_list 	= self.incomes.Description.values
		labelsCat_list 	= self.incomes.CatAndSub.values
		# dates_list		= self.incomes.Date.values
		value_list      = self.incomes.Value.values

		# Retrieve a title name for the graph to be made.
		catIdx_int    = labelsCat_list[0].find(":")
		if catIdx_int == -1:
			title_str = "Uknown Income Data"
		else:
			title_str = str(labelsCat_list[0][:catIdx_int])
		
		currentIncome_ax = fig.add_subplot(inner_top[0, 0]) # this is also one of the cleaner ways to create the axis
		currentIncome_ax.set_prop_cycle(color=[CMAP(j) for j in range(1,10)])
		graphing.Graphing_PieChart(labelsDes_list, value_list, currentIncome_ax, category=title_str, LABELS=False)
		fig.add_subplot(currentIncome_ax)

		# i += 1

		# if i == n_charts_top:
		# 	break
		
		########################################################################
		# LEGACY
		########################################################################
		# we want to display pie charts showing; hours, hourly + comms dist, income + tax dist
		# incomes contains 2 frames and three sub-dicts, the data we need for charts is in frames
		# income_stats = self.incomes["latest_week_income"]
		# income_agg = self.incomes["aggregate_income"]

		# # labels
		# hour_dist_labels = income_stats["Description"].values
		# hour_plus_comms_labels = income_stats["Description_Other"].values
		# income_tax_dist_labels = ["Tax","NET income"] 
		# # data
		# hour_dist_data = np.array(income_stats["Value"].values, dtype=np.float32)
		# hour_plus_comms_data = np.array(income_stats["Value_Other"].values, dtype=np.float32)
		# income_tax_dist_data = [
		# 	# access the first element, janky I know..
		# 	np.array(income_agg.Tax.values, dtype=np.float32)[0], 
		# 	np.array(income_agg["NET INCOME"].values, dtype=np.float32)[0],			] 
		
		# # now create the subplots for each pie chart
		# ax_hour_dist = plt.Subplot(fig, inner_top[0])
		# ax_hour_plus_comms = plt.Subplot(fig, inner_top[1])
		# ax_income_tax = plt.Subplot(fig, inner_top[2])

		# list_ax = [ax_hour_dist, ax_hour_plus_comms, ax_income_tax]
		# label_val_dicts = [
		# 	dict(zip(hour_dist_labels,       hour_dist_data.tolist())), 
		# 	dict(zip(hour_plus_comms_labels, hour_plus_comms_data.tolist())), 
		# 	dict(zip(income_tax_dist_labels, income_tax_dist_data)),
		# 	]
		# list_titles = ["Hourly Distribution", "Other", "Income-Taxation Distribution"]
		# # compelete by generating charts and setting CMAP
		# for i, ax in enumerate(list_ax):
		# 	ax.set_prop_cycle(color=[CMAP(j) for j in range(1,10)])
		# 	graphing.Graphing_PieChart(label_val_dicts[i].keys(), label_val_dicts[i].values(),
		# 	ax, category=list_titles[i], LABELS=False
		# 	)
		# 	fig.add_subplot(ax)

		# # read a list from our TxData object list
		# income_raw = [tx.val for tx in self.incomes['primary_income']]
		# # now use the raw data to create a bar chart of NET income data
		# ax_bar_income_raw = plt.Subplot(fig, inner_bottom[0])
		# bar_labels = ["Week {}".format(i) for i in range(len(income_raw))]
		# # reverse to give time proceeding to the right, more intuitive to user
		# graphing.Graphing_BarChart(bar_labels, income_raw, ax_bar_income_raw)
		# ax_bar_income_raw.set_ylabel('Income')
		# ax_bar_income_raw.set_xlabel('Week of Income')
		# plt.suptitle("Income Statistics")
		# fig.add_subplot(ax_bar_income_raw)

		return images.img_buffer_to_svg(fig)

	def display_savings_stats(self, figsize=(10,10)):
		"""Generate the display for savings data, based on bank account drawn data. 
		
		.. note:: TODO Integrate options for REST Super.
		
		**Kwargs:**
			figsize(int tuple):	The size of the figure to generate. Default is (10, 10).
		
		**Returns:**
			images.image_buffer_to_svg(PIL.image): An SVG PIL image.
		"""

		fig = plt.figure(figsize=figsize)
		# Display savings across accounts, bar per acc., i.e. bar figure
		# Trendline of account, with short range projection (1 month)
		#	plot 1 month predic. line
		#	plot 1 month best-case (optimal saving)
		
		# set the display stack of two charts with grid_spec

		# setup the grids for holding our plots, attach them to the same figure
		n_charts_top = len(self.savings)
		outer_grid_spec = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
		disp_top 		= gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid_spec[0],
					wspace=0.1, hspace=0.1)
		disp_bottom 	= gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid_spec[1],
					wspace=0.1, hspace=0.1)

		labelsDes_list 	= self.savings.Description.values
		labelsCat_list 	= self.savings.CatAndSub.values
		dates_list		= self.savings.Date.values
		value_list      = self.savings.Value.values

		

		# Retrieve a title name for the graph to be made.
		catIdx_int    = labelsCat_list[0].find(":")
		if catIdx_int == -1:
			title_str = "Uknown Income Data"
		else:
			title_str = str(labelsCat_list[0][:catIdx_int])
		
		barChartSavings_ax	= plt.Subplot(fig, disp_bottom[0])
		graphing.Graphing_BarChart(labelsDes_list.tolist(), value_list.tolist(), barChartSavings_ax)
		barChartSavings_ax.set_ylabel('Savings')
		barChartSavings_ax.set_xlabel('Date and Description')
		fig.add_subplot(barChartSavings_ax)

		totalSavings_int = sum(value_list)
		scatterChartSavings_ax = plt.Subplot(fig, disp_top[0])
		graphing.Graphing_ScatterPlot(dates_list, value_list, scatterChartSavings_ax, area=totalSavings_int)
		scatterChartSavings_ax.set_ylabel("Savings Data")
		scatterChartSavings_ax.set_xlabel("Savings Date")
		fig.add_subplot(scatterChartSavings_ax)
		plt.suptitle("Savings Statistics")
		plt.xticks(rotation=30)

		
		########################################################################
		# LEGACY
		########################################################################

		# multiple savings sources, grab the raw data
		# savings_data = [[] for i in range(len(self.savings))]
		# savings_dates = [[] for i in range(len(self.savings))]
		# lbls = tuple(key for key in self.savings.keys())
		
		# for i, key in enumerate(self.savings):
		# 	savings_data[i] = [abs(tx.val) for tx in self.savings[key]]
		# 	savings_dates[i] = [tx.date for tx in self.savings[key]]

		# Add dates to savings labels
		# for i in range(len(lbls)):
		# 	# grab the label prefix and reset with correct list numbers
		# 	savings_lbls = [[] for i in range(len(self.savings))]
		# 	for j in range(len(savings_dates[i])):
		# 		savings_lbls[i].append(lbls[i] + ' ' + savings_dates[i][j])

		# TODO, not neccessairly the same week, this is intended to be used in the scatter vs. savings in same week
		# to make it simpler, draw income net from the bank acc. data not the payslip, make the file's time-stamps work for us
		# income_total_curr = float(self.incomes['aggregate_income'].Gross[0])
		# total_savings = 0

		# for savings in savings_data:
		# 	total_savings += sum(savings)
		# savings_perc = total_savings/income_total_curr		

		# TODO : Add support for graphing all 3 sub cats for accounts (combined or seperate whatever...)
		# bar chart subplot on disp_bottom
		# ax_savings_bar	= plt.Subplot(fig, disp_bottom[0])
		# graphing.Graphing_BarChart(savings_lbls[1], savings_data[1], ax_savings_bar)
		# ax_savings_bar.set_ylabel('Savings')
		# ax_savings_bar.set_xlabel('Date and Description')
		# fig.add_subplot(ax_savings_bar)

		# now create the trendline and place it in disp_top
		# ax_savings_trend = plt.Subplot(fig, disp_top[0])
		# graphing.Graphing_ScatterPlot(savings_dates[1], savings_data[1], ax_savings_trend, area=savings_perc)
		# ax_savings_trend.set_ylabel("Savings Data")
		# ax_savings_trend.set_xlabel("Savings Date")
		# fig.add_subplot(ax_savings_trend)
		# plt.suptitle("Savings Statistics")
		
		return images.img_buffer_to_svg(fig)

	def display_expenditure_stats(self, figsize=(10,10)):
		""" Display some visualisations and print outs of the income data.

		**Kwargs:**
			figsize(int tuple):	The size of the figure to generate. Default is (10, 10).
		
		**Returns:**
			images.image_buffer_to_svg(PIL.image): An SVG PIL image.
		"""
			
		# setup the grids for holding our plots, attach them to the same figure
		# inner_**** are for use with plotting, outer is purely spacing
		# colCtr_int   = math.ceil(len(self.expenditures.keys()))
		fig          = plt.figure(figsize=figsize)
		outer        = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,1])
		inner_top    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
		inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1],wspace=0.1, hspace=0.1)

		labelsCat_dict = {'unknown expenditures': 0} # tracks total expenditures per category
		# labelsDes_list 	= self.expenditures.Description.values
		labelsCat_list 	= self.expenditures.CatAndSub.values
		# dates_list		= self.expenditures.Date.values
		value_list      = self.expenditures.Value.values

		for i in range(len(self.expenditures)):
			catIdx_int = labelsCat_list[i].find(":")
			if catIdx_int == -1:
				labelsCat_dict['unknown expenditures'] += value_list[i]
			else:
				cat_str = str(labelsCat_list[i][:catIdx_int])
				if cat_str not in labelsCat_dict:
					labelsCat_dict[cat_str] = value_list[i]
				else:
					labelsCat_dict['unknown expenditures'] += value_list[i]

		expendituresPieChart_ax = fig.add_subplot(inner_top[0, 0]) # this is also one of the cleaner ways to create the axis
		expendituresPieChart_ax.set_prop_cycle(color=[CMAP(j) for j in range(1,10)])
		graphing.Graphing_PieChart(labelsCat_dict.keys(), labelsCat_dict.values(), expendituresPieChart_ax, category='Expenditure Percentages')
		fig.add_subplot(expendituresPieChart_ax)

		expendituresBarChart_ax = fig.add_subplot(inner_bottom[0])
		graphing.Graphing_BarChart(list(labelsCat_dict.keys()), list(labelsCat_dict.values()), expendituresBarChart_ax)
		expendituresBarChart_ax.set_ylabel('Expenditure')
		expendituresBarChart_ax.set_xlabel('Category of Expenditure')
		plt.suptitle("Expenditure Statistics")
		fig.add_subplot(expendituresBarChart_ax)

		########################################################################
		# LEGACY
		########################################################################

		# key_counter = 0
		# for key, term_list in self.expenditures.items():
		# 	label_vals = {}
		# 	for tx in term_list:
		# 		new_value = tx.val
		# 		new_label = tx.desc
		# 		if new_label not in label_vals:
		# 			label_vals[new_label] = new_value
		# 		else:
		# 			label_vals[new_label] += new_value

		# 	# new category creates a new axis on the upper plot region
		# 	if key_counter < col_count:
		# 		axN = fig.add_subplot(inner_top[0, key_counter]) # this is also one of the cleaner ways to create the axis
		# 	else:
		# 		axN = fig.add_subplot(inner_top[1, key_counter - col_count]) # this is also one of the cleaner ways to create the axis

		# 	axN.set_prop_cycle(color=[CMAP(i) for i in range(1,10)])
		# 	graphing.Graphing_PieChart(label_vals.keys(), label_vals.values(), axN, category=key)
		# 	totals.append(sum(label_vals.values()))
		# 	key_counter -=- 1

		# plt.suptitle("Expenditure Statistics")
		# ax_rect = fig.add_subplot(inner_bottom[0])
		# graphing.Graphing_BarChart(list(self.expenditures.keys()), totals, ax_rect)
		
		# ax_rect.set_ylabel('Expenditure')
		# ax_rect.set_xlabel('Category of Expenditure')
		# fig.add_subplot(ax_rect)

		return images.img_buffer_to_svg(fig)
