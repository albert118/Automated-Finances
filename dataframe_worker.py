import pandas as pd
import numpy as np

from tabula import read_pdf as r

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import environ
from datetime import date
import math
import os
import traceback
import warnings

# Global colour map variable
CMAP =  plt.get_cmap('Paired')

def safe_environ():
	"""Check for the environment settings and config file. Attempt to gracefully
	import the local.env file and deal with a FileNotFoundError or other
	configuration error.

	Returns
	----------
	env, Env builtin object
		the environemtn class object is returned on successful detection of the
		local.env file
	default_warn, str
		this is returned if the file cannot be found. Prints a message to stderr
	"""

	default_warn = "[ENVIRON SETTINGS] Environment settings not found"
	try:
		# grab local environ settings and attempt to read settings file
		env = environ.Env()
		env_file = os.path.join(os.getcwd(), "local.env")
		env.read_env(env_file)
		return env
	except FileNotFoundError:
		return sys.exit(default_warn)

class AccountData(accountframe):
	"""AccountData(accountframe)
	This is an object designed to track several finance accounts of the user. It aims to
	combine multiple finance related accounts into one location for visualisation and
	informative statistics. 

	IDEA: 
	Monthly overview of bank accounts,
		* incoming review; avg., total p/mnth, total p each week, by category:
		* outgoing overview; total, avg., by category 
		  (coffee, shopping, eating, groceries, utilities, health)
		* savings overview: avg., total p/mnth, total p each week
		* opening balance : closing balance : delta
		* Category and subcategory visualisations to help track progress to 
		  goals and see current position!
		* planned/upcoming expenditure integrations (see calendar integrations)
		* user savings goals and calculator tools for interest - fee accumulations

	Paycheck integrations from ADP payroll solutions
		* 4-week average,
		* last received,
		* hourly and commission based stats

	Events and Calendar Integrations
		* facebook event tracking (group specific)
		* iCal/Google Cal integrations
		* other??

	Superannuaton Interation
		* Review of fees
		* Investment gains
		* other??

	Parameters
	----------
	accountframe: pandas DataFrame
		a CSV bank account data dump
	|tbd| payslipframe, pandas DataFrame
		a preprocessed payslip-to-df conversion from ADP services 

	See Also:
	|tbd| refactoring will separate the get_payslips functionality out of this class
	
	NOTE: categories are defined with all caps, this distinguishes 
	from data values for the categories of the same name.

	e.g. INCOME represents the category containing the subcategories
	primary, supplemental and investment where as incomes 
	represents the data associated per sub-category.

	Examples
	----------
	>>> import pandas as pd
	>>> import dataframe_worker as w
	>>> df = pd.read_csv("CSVData.csv", names=["Date","Tx", "Description", "Curr_Balance"])
	>>> a = w.account_data(df)
	>>> 
	>>> a.display_income_stats()
	>>> a.display_expenditure_stats()
	>>> a.display_savings_stats()

	this will display several charts on the named areas.
	""" 

	def __init__(self, account_frame):
		"""Initialize self.  See help(type(self)) for accurate signature."""

		# ensure safe env on account object instantiation
		env = safe_environ()

		########################################################################
		# Categories and Search terms
		########################################################################
		
		self.INCOME = {
			'primary_income': 		env.str("primary_income"), 
			'supplemental_income': 	env.str("supplemental_income"), 
			'investment_income': 	env.str("investment_income"),
			'latest_week_income': 	env.str("latest_week_income"),
			'aggregate_income':		env.str("aggregate_income"),
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

		self.SAVINGS_IDS = [env("CACHE_ID"), env("SAVINGS_ID")]

		# format the account df and perform date-time refactoring
		account_frame.Description = account_frame.Description.apply(str.upper)
		account_frame.Date=pd.to_datetime(account_frame.Date, format="%d/%m/%Y")

		# initialize the account tracking data
		self.incomes = self.get_income(account_frame)
		self.savings = self.get_savings(account_frame)
		self.expenditures = self.get_expenditures(account_frame)

		# we need to maintain a list of stats for every sub category and its
		# relevant stats dicts/lists, dynamically configure this based on cat's 
		# defined for the class above...
		self.curr_income_stats, self.curr_savings_stats, self.curr_expenditure_stats = ([] for i in range(3))

	############################################################################
	# Getters
	############################################################################
	
	def get_income(self, acc_frame):
		"""Get the user's bank details on income and combine with payroll data.
		Inputs:
			* acc_frame, acc_frame, the bank details frame

			* kwargs:
				* payslip_name, defaults to simple title. Use this to set the payslip
				file name

				* true_col_header_index, defaults to a magic number that works for my
				ADP payroll pdf data. Use this if inspection of your pdf begins
				the table earlier/later.
		Returns:
			* incomes, income_week_data, income_aggregate_data
				a list of income data. incomes is intended for simple graphing
				where as income week and aggregate datas are intended for
				week to week performance metrics and tax checks, etc...
		"""
		incomes = self.get_bank_incomes(acc_frame)
		income_aggregate_data, income_week_data = self.get_payslips()
		incomes["aggregate_income"] = income_aggregate_data
		incomes["latest_week_income"] = income_week_data
		return incomes

	def get_bank_incomes(self, acc_frame):
		"""Get any aggregate income details present in banking data. This is
		primarily a categorical search.
		----------------------------------------------------------------------------
		Inputs:
			* acc_frame, the bank details frame
		Returns:
			* incomes, frame with income categories and discovered sub categories.
		"""

		date_income, date_supplemental, date_investment = ([] for i in range(3))
		# programatically update incomes dict and associated lists to maintain
		incomes = dict(zip(self.INCOME.keys(), ([] for i in range(len(self.INCOME)))))

		for cat_key, sub_cat_list in self.INCOME.items():
			for i in range(0, len(acc_frame)):
				if str(sub_cat_list).strip() in acc_frame.Description[i]:
					incomes[cat_key].append([acc_frame.Date[i], acc_frame.Tx[i], sub_cat_list])

		return incomes

	def get_payslips(self, payslip_name='payslip.pdf', true_col_header_index = 5):
		"""Retreive the pdf, convert to dataframes for income stats and aggregate
		income data. true_col_header_index is the constant value with the first
		data frames actual headers that require extraction. This value is used
		to relatively find further dataframes from the pdf.
		
		Note on usability: this function is intended to call up the latest payslip
		for weekly displays, the stats function for income then aggregates for
		monthly displays, etc...
		----------------------------------------------------------------------------
		Inputs:
			* kwargs:
				payslip_name, defaults to simple title. Use this to set the payslip
				file name

				true_col_header_index, defaults to a magic number that works for my
				ADP payroll pdf data. Use this if inspection of your pdf begins
				the table earlier/later.

		Returns:
			* income_stats_data is the dataframe with hourly data, commissions and 
				deductions
				[Description, Rate, Hours, Value] [Description, Tax Index, Value]
			* income_data is the aggregate income values 
				[Gross, Taxable, Ded's pre and post, Tax, Net]

		"""
		
		########################################################################
		# Declerations and Instantiations
		########################################################################

		income_data = pd.DataFrame()
		income_stats_data = pd.DataFrame()
		income_data_header_idx = None

		# retrieve the payslip as a dataframe
		# this retrieval uses the tabula pdf_reader
		data = r(payslip_name)[0] # accessing the first of the frames returned

		########################################################################

		########################################################################
		# Internal Utilities
		########################################################################
		
		def _rename_headers(dataframe, header_index, cols_default_headers):
			""" Rename the column headers from default guesses to the correct values.
			Also performs some housekeeping by reindexing and dropping the header row. 
			
			Due to the nature of separating a frame like this, it is possible to create duplicate 
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

			except TypeError as Header_Index_Error:
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

		def _split_merged_columns(dataframe):
			# check first row 'splittable'
			hdr_vals = dataframe.columns.tolist()
			idxs_added = []

			# start by splitting column names and inserting blank columns ready for data
			i = 0
			for val in hdr_vals:
				if ' ' in str(val):
					new_hdrs = val.split()
					# insert a new column at this position with NaN type values
					try:
						dataframe.insert(i + 1, str(new_hdrs[-1]), np.NaN)
					except ValueError as already_exists:
						dataframe.insert(i + 1, str(new_hdrs[-1])+str(i), np.NaN)

					# rename the current column
					dataframe.rename(
						columns={dataframe.columns[i]: new_hdrs[-2]}, 
						inplace=True,
						copy=False)
					# record the insertion index
					idxs_added.append(i)
					# jump past the column we just inserted
					i -=- 2
				else:
					# we couldn't split, jump to next column
					i -=- 1

			# start from 1, skip the headers
			for i in range(0,len(dataframe)):
				row_vals = dataframe.iloc[i].tolist()
				# we know from our previous insertion which col idx's require splitting
				for idx in idxs_added:
					vals = dataframe.iloc[i, idx].split()
					if len(vals) > 2:
						bool_arr = [type(elem) is not str for elem in vals]
						num_val = vals.pop(bool_arr.index(False))
						str_val = '_'.join(vals)
						vals = [num_val, str_val]

					# format our description value
					if vals[1] is type(str):
						vals[1].replace('*', '').lower().capitalize()
					
					# add the data to the new column
					dataframe.iloc[i, idx + 1] = vals[1]
					# then replace the merged values with the single column value
					dataframe.iloc[i, idx] = vals[0]
			test_cols_list = ['Description', 'Rate', 'Hours', 'Value', 'Description3', 'nan', 'Tax_Ind', 'Value']
			if dataframe.columns.duplicated().any()	:
				dataframe.columns = ['Description_Hours', 'Rate', 'Hours', 'Value_Hours', 'Description_Other', 'nan', 'Tax_Ind', 'Value_Other']
				dataframe = dataframe.drop("nan", axis=1)

			return dataframe

		########################################################################

		# drop the NaN column generated on import
		# and get the default column titles
		data = data.drop(["Unnamed: 0", "Status"], axis=1)
		cols_default_headers = data.columns.values

		# split the data into new columns where tabula merged them, this must be
		# dynamic as user could work further combinations of work rates or add
		# further deductions, etc..., thus extending elem's of income stats table
		for i in range(5, len(data)):
			row_split_check = data.iloc[[i]].isnull()
			if row_split_check.values.any():
				bool_header_NaN_spacing = row_split_check.sum().tolist()

				# Note: 'is' used rather than ==, 1 is interned within Python by default
				if bool_header_NaN_spacing.count(0) is 1:
					# this is the index where we split the data frame from aggregate
					# and stat's income values, break after saving the new df
					income_stats_data = data[true_col_header_index:i]
					income_data = data[i + 1:len(data)]
					income_data_header_idx = i + 1
					break

		try:
			# use the actual titles in row_id 5 to rename the column headers for
			# income_stats_data, repeat for income_data
			if income_stats_data.empty or income_data.empty:
				print("A frame was incorrectly initialised.")
				raise ValueError
			
			else:
				income_stats_data = _rename_headers(income_stats_data, true_col_header_index, cols_default_headers)
				income_data = _rename_headers(income_data, income_data_header_idx, cols_default_headers)


		except Exception as e:
			print(type(e))
			if income_data_header_idx is None:
				print("The income and stats frames could not be dynamically calculated. ")
			else:
				print("Some error occured")
				print('Income Stats Frame:\n', income_stats_data, '\n')
				print('Income Data Frame:\n', income_data, '\n')
				print('Data Frame:\n', data, '\n')
				traceback.print_stack()
				return

		try:
			# now the frames have been split horizontally, 
			# split vertically where some columns have been merged
			income_stats_data = _split_merged_columns(income_stats_data)

		except Exception  as e:
			print(type(e))
			print("Could not split merged data values of income stats data.")
			traceback.print_stack()
			raise
		# manually correct the header titles of income_data
		hdr_vals_income = income_data.columns.values
		income_data.rename(
			columns={
				income_data.columns[2]: "Pre Tax Allows/Deds",
				income_data.columns[2]: "Post Tax Allows/Deds",
			},
			inplace=True,
			copy=False)
		
		# return our corrected frames
		return income_data, income_stats_data

	def get_savings(self, acc_frame):
		date_savings = {}
		for i in range(len(acc_frame)):
			# tx for savings always includes the acc_id ref
			for _id in self.SAVINGS_IDS:
				desc_val = acc_frame.loc[i, "Description"]
				tx_val   = acc_frame.loc[i, "Tx"]
				# test for outgoing as well as unique ref id
				if _id in desc_val and tx_val > 0:
					date_savings[desc_val] = [acc_frame.loc[i, "Date"], tx_val]
		return date_savings

	def get_expenditures(self, acc_frame):
		len_acc_frame = len(acc_frame)
		# programatically update expenditures dict and associated lists to maintain
		expenditures = dict(zip(self.EXPENDITURES.keys(), ([] for i in range(len(self.EXPENDITURES)))))

		# iterate through cateogries
		for cat_key, sub_cat_list in self.EXPENDITURES.items():
			# iterate through dataframe elements
			for i in range(0, len_acc_frame):
				# iterate through each sub-cat term in sub_cat_list
				for sub_cat in sub_cat_list:
					# INSTRUMENTAL TO BS NOT FINDING CAT'S IS INCLUDING THE STRIP FUNCTION!!!
					search_term = str(sub_cat.upper()).strip()
					search_field = str(acc_frame.Description[i].upper())
					idx = search_field.find(search_term)
					
					if idx != -1:
						expenditures[cat_key].append([acc_frame.Date[i], acc_frame.Tx[i], search_term])
						
		return expenditures

	############################################################################
	# Displayers and updaters
	############################################################################
	
	def update_income_stats(self):
		"""
		This method *assumes* that if new categories are added that they are 
		appended, hence: previously known ordered additions of stats are in 
		the same index positon and keyword order

		Further, updates the payslip data by recalling get_payslips
		"""

		# start by refeshing the categories from local.env file
		i = 0
		for income in self.incomes:
			# grab the income lists (raw data)
			dated_txs = self.incomes[income]
			if len(dated_txs) == 0:
				continue

			# check the stat's info
			# update initial vals of our specific income stats if they dont exist
			# there will be as many as these as categories in self.incomes
			if len(self.curr_income_stats)  == 0:
				self.curr_income_stats = self.stats(dated_txs)
			else:
				# push updates to running_stats
				running_stats = self.curr_income_stats['running_stats']
				self.curr_income_stats = self.stats(dated_txs, *running_stats)

			print("update income stats: {}".format(i))
			i-=-1

		return True
	
	def update_savings_stats(self):
		"""
		This method *assumes* that if new categories are added that they are 
		appended, hence: previously known ordered additions of stats are in 
		the same index positon and keyword order
		"""
		
		# controls the max iterations of the stats details below
		num_savings_accounts = 1

		for i in range(0, num_savings_accounts):
			# grab the savings lists (raw data)
			dated_txs = self.savings
			if len(dated_txs) == 0:
				continue # we skip if the length is zero, avoids divide byt zero issues

			if len(self.curr_savings_stats)  == 0:
				# update initial vals of our specific savings stats if they dont exist
				# there will be as many as these as categories in self.savings
				self.curr_savings_stats = self.stats(dated_txs)
			else:
				# recalc the stats, but call the previous ones associated with 
				# the current subcategory for reference in incrementally 
				# calculating the new stats, 
				curr_stats = self.curr_savings_stats
				#i.e. grab the running_stats dict, *curr_stats[0]
				self.curr_savings_stats = self.stats(dated_txs, *curr_stats['running_stats'])

			print("update savings stats: {}".format(i))

		return True
	
	def update_expenditure_stats(self):
		"""
		This method *assumes* that if new categories are added that they are 
		appended, hence: previously known ordered additions of stats are in 
		the same index positon and keyword order
		"""

		i = 0
		for expenditure in self.expenditures:
			# grab the expenditure lists (raw data)
			dated_txs = self.expenditures[expenditure]
			if len(dated_txs) == 0:
				continue

			if len(self.curr_expenditure_stats)  == 0:
				# update initial vals of our specific expenditure stats if they dont exist
				# there will be as many as these as categories in self.incomes
				self.curr_expenditure_stats = self.stats(dated_txs)
			else:
				# recalc the stats, but call the previous ones associated with 
				# the current subcategory for reference in incrementally 
				# calculating the new stats, 
				curr_stats = self.curr_expenditure_stats
				#i.e. grab the running_stats dict, *curr_stats[0]
				self.curr_expenditure_stats = self.stats(dated_txs, *curr_stats['running_stats'])

			print("update expenditure stats: {}".format(i))
			i-=-1

		return True

	def display_income_stats(self, n_charts_top = 3, figsize=(10,10)):
		""" Display some visualisations and print outs of the income data. """

		# setup the grids for holding our plots, attach them to the same figure
		fig = plt.figure(figsize=(10,10))
		outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
		# inner_**** are for use with plotting, outer is purely spacing
		inner_top = gridspec.GridSpecFromSubplotSpec(1, n_charts_top, subplot_spec=outer[0],
					wspace=0.1, hspace=0.1)
		inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1],
					wspace=0.1, hspace=0.1)

		# we want to display pie charts showing; hours, hourly + comms dist, income + tax dist
		# incomes contains 2 frames and three sub-dicts, the data we need for charts is in frames
		income_stats = self.incomes["latest_week_income"]
		income_agg = self.incomes["aggregate_income"]

		# labels
		hour_dist_labels = income_stats["Description_Hours"].values
		hour_plus_comms_labels = income_stats["Description_Other"].values
		income_tax_dist_labels = ["Tax","NET income"] 
		# data
		hour_dist_data = np.array(income_stats["Value_Hours"].values, dtype=np.float32)
		hour_plus_comms_data = np.array(income_stats["Value_Other"].values, dtype=np.float32)
		income_tax_dist_data = [
			# access the first element, janky I know..
			np.array(income_agg.Tax.values, dtype=np.float32)[0], 
			np.array(income_agg["NET INCOME"].values, dtype=np.float32)[0],			] 
		
		# now create the subplots for each pie chart
		ax_hour_dist = plt.Subplot(fig, inner_top[0])
		ax_hour_plus_comms = plt.Subplot(fig, inner_top[1])
		ax_income_tax = plt.Subplot(fig, inner_top[2])

		list_ax = [ax_hour_dist, ax_hour_plus_comms, ax_income_tax]
		label_val_dicts = [
			dict(zip(hour_dist_labels,       hour_dist_data.tolist())), 
			dict(zip(hour_plus_comms_labels, hour_plus_comms_data.tolist())), 
			dict(zip(income_tax_dist_labels, income_tax_dist_data)),
			]
		list_titles = ["Hourly Distribution", "Other", "Income-Taxation Distribution"]
		# compelete by generating charts and setting CMAP
		i = 0
		for ax in list_ax:
			ax.set_prop_cycle(color=[CMAP(j) for j in range(1,10)])
			pie_chart(label_val_dicts[i], ax, category=list_titles[i], LABELS=False)
			fig.add_subplot(ax)
			i -=-1

		# list conversion needed as Tkinter crashes on np.array() interaction
		income_raw = list(np.array(self.incomes['primary_income'])[:,1])
		# now use the raw data to create a bar chart of NET income data
		ax_bar_income_raw = plt.Subplot(fig, inner_bottom[0])
		bar_labels = ["Week {}".format(i) for i in range(len(income_raw))]
		# reverse to give time proceeding to the right, more intuitive to user
		bar_chart(bar_labels, income_raw, ax_bar_income_raw)
		ax_bar_income_raw.set_ylabel('Income')
		ax_bar_income_raw.set_xlabel('Week of Income')

		fig.add_subplot(ax_bar_income_raw)
		return fig

	def display_savings_stats(self, figsize=(10,10)):
		"""Generate the display for savings data, based on bank account drawn data. 
		TODO: Integrate options for REST Super"""
		
		fig = plt.figure(figsize=figsize)
		# Display savings across accounts, bar per acc., i.e. bar figure
		# Trendline of account, with short range projection (1 month)
		#	plot 1 month predic. line
		#	plot 1 month best-case (optimal saving)
		
		# set the display stack of two charts with grid_spec
		outer_grid_spec = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
		disp_top 		= gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid_spec[0],
					wspace=0.1, hspace=0.1)
		disp_bottom 	= gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid_spec[1],
					wspace=0.1, hspace=0.1)

		# create the bar chart savings quick view (think like CommBank's app viz)
		savings_dates, savings_data = ([] for i in range(2))
		raw_data = list(self.savings.values())
		savings_lbls = list(self.savings.keys()) # keys are description values
		# remember, savings data is a dict where val is a list, so calling values returns a list of lists!
		for i in range(len(self.savings.values())):
			sub_list = raw_data[i]
			savings_data.append(sub_list[1])
			savings_dates.append(sub_list[0].to_numpy())
		raw_data.clear() # not needed in memory anymore!! Save the space

		# TODO, not neccessairly the same week, this is intended to be used in the scatter vs. savings in same week
		# to make it simpler, draw income net from the bank acc. data not the payslip, make the file's time-stamps work for us
		raw_data = []
		raw_data = self.incomes["primary_income"] # already a list, comprehension is far easier this time!
		income_net = [elem[1] for elem in raw_data]
		savings_perc = [savings_data[i]/income_net[i] for i in range(len(savings_data))]
		
		# adjust the labels to include the dates
		for i in range(len(savings_lbls)):
			savings_lbls[i] = str(savings_dates[i]) + ' ' + savings_lbls[i]

		# bar chart subplot on disp_bottom
		ax_savings_bar	= plt.Subplot(fig, disp_bottom[0])
		bar_chart(savings_lbls, savings_data, ax_savings_bar)
		ax_savings_bar.set_ylabel('Savings')
		ax_savings_bar.set_xlabel('Date and Description')
		fig.add_subplot(ax_savings_bar)

		# now create the trendline and place it in disp_top
		ax_savings_trend = plt.Subplot(fig, disp_top[0])
		scatter_plotter(savings_dates, savings_data, ax_savings_trend, area=savings_perc)
		ax_savings_trend.set_ylabel("Savings Data")
		ax_savings_trend.set_xlabel("Savings Date")
		fig.add_subplot(ax_savings_trend)

		return fig

	def display_expenditure_stats(self, figsize=(10,10)):
		""" Display some visualisations and print outs of the income data. """
		# Generate a pie chart of expenditures
		# Generate a bar chart of each category vs. total budget

		totals = []		
		# create the outer subplot that will hold the boxplot and subplots
		fig = plt.figure(figsize=(10,10))
		outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,1])
		col_count = math.ceil(len(self.expenditures.keys())/2)
		inner_top = gridspec.GridSpecFromSubplotSpec(2, col_count, subplot_spec=outer[0],
					wspace=0.2, hspace=0.2)
		inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1],
					wspace=0.05, hspace=0.1)

		key_counter = 0
		for key, term_list in self.expenditures.items():
			
			labels = {} # labels and associated cost to use
			for tx in term_list:
				new_value = tx[1]
				new_label = tx[2]
				if new_label not in labels:
					labels[new_label] = new_value
				else:
					labels[new_label] += new_value

			# new category creates a new axis on the upper plot region
			if key_counter < col_count:
				axN = fig.add_subplot(inner_top[0, key_counter]) # this is also one of the cleaner ways to create the axis
			else:
				axN = fig.add_subplot(inner_top[1, key_counter-col_count]) # this is also one of the cleaner ways to create the axis

			axN.set_prop_cycle(color=[CMAP(i) for i in range(1,10)])
			pie_chart(labels, axN, category=key)

			totals.append(sum(labels.values()))
			key_counter -=- 1

		plt.suptitle("Expenditure Statistics")
		ax_rect = fig.add_subplot(inner_bottom[0])
		bar_chart(list(self.expenditures.keys()), totals, ax_rect)
		
		ax_rect.set_ylabel('Expenditure')
		ax_rect.set_xlabel('Category of Expenditure')
		fig.add_subplot(ax_rect)

		# later, add paycheck stuff here too - ADP does it well, do the same pie-chart
		# and check MoneyTree, great visualisations on that too
		return fig

	############################################################################
	# Stats
	############################################################################

	def stats(self, date_tx, curr_mean=None, curr_min=None, curr_max=None, curr_std=None, curr_tot=None):
		""" 
		IDEA: 
			Calculate various statistics on the account data passed to the function.
			* allow for continuous updates and integration of data.
		Inputs: 
			date_tx is a 2D array-list like object.
			the rest are stats as labelled, these are running "total"-eqsue stats
		Returns:
			A nested dictionary - two stat dict's and one list of dicts: 
			running stats dict, list of weekly stats dicts and a 4-weekly stats dict

		------------------------------------------------------------------------
		key-val args must be set if function is previously called, this
		is required to update the running statistics on the accounts being 
		watched as new transactions are added!
		------------------------------------------------------------------------
		"""

		# get the numpy arrays for comparison and iteration later
		array = np.array(date_tx)
		tx_vals = array[:,1]
		dates = pd.Series(array[:, 0])
		weekly_stats = []
		running_stats = {
			'_mean': curr_mean, 
			'_min': curr_min, 
			'_max': curr_max, 
			'_std': curr_std, 
			'_tot': curr_tot,
		}

		# check key-val args for pre-stats
		if None in running_stats.values():
			# then we need to set the stats init vals
			running_stats['_mean'] = tx_vals.mean()
			running_stats['_min'] = tx_vals.min()
			running_stats['_max'] = tx_vals.max()
			running_stats['_std'] = tx_vals.std()
			running_stats['_tot'] = tx_vals.sum()
		# else, incrementally update the values
		else:
			# then we need to update the stats
			running_stats['_std'] = incremental_standard_dev(running_stats['_std'], tx_vals, running_stats['_mean'], tx_vals.mean())
			running_stats['_mean'] = incremental_mean(running_stats['_mean'], tx_vals)
			running_stats['_min'] = min(tx_vals.min(), running_stats['_min'])
			running_stats['_max'] = max(tx_vals.max(), running_stats['_max'])
			running_stats['_tot'] = running_stats['_tot'] + _tot

		curr_date = date.today()
		curr_week = 1 # we iter from the first week onwards
		# comp vals for later, use these to keep memory of the single overall min and max vals
		four_min = 999999
		four_max = 0
		# as well as the total...
		total = 0
		# and incremental vals for std and mean
		four_std = 0
		four_mean = 0

		# weekly and 4-week stats, grab the indexes for each transaction per week and culm sum them for 4-week
		for i in range(0, 4):
			# TODO: Edge case of the final days of the month included on last lap for stats, otherwise we ignore 3 days
			# between for series and index for lists
			min_date = date(curr_date.year, curr_date.month, curr_week)
			max_date = date(curr_date.year, curr_date.month, curr_week+7)
			
			# this bool indexing can be applied with pandas as a "key" lookup
			bool_test = dates.between(min_date, max_date)
			# test in case of zero income in the week, avoids possible div 0 error
			if not any(bool_test):
				continue

			vals = tx_vals[bool_test]
			curr_week += 7

			# calc our stats and stuff them into the dict
			_stats_week = {
				'_mean': vals.mean(), 
				'_min': vals.min(), 
				'_max': vals.max(), 
				'_std': vals.std(), 
				'_tot': vals.sum(),
			}

			weekly_stats.append(_stats_week)
			
			if i == 0:
				four_std = _stats_week['_std']
				four_mean = _stats_week['_mean']
			else:
				# incremental calc for four_week stats
				_old_mean = four_mean
				four_mean = incremental_mean(four_mean, vals)
				four_std = incremental_standard_dev(four_std, vals, _old_mean, four_mean)

			total += _stats_week['_tot']
			four_max=max(four_max, _stats_week['_max'])
			four_min=min(four_min, _stats_week['_min'])
		
		four_week_stats = {
			'_mean': four_mean, 
			'_min': four_min, 
			'_max': four_max, 
			'_std': four_std, 
			'_tot': total,
		}

		return {'running_stats': running_stats,'weekly_stats': weekly_stats,'four_week_stats': four_week_stats}

################################################################################
# Utility
################################################################################

def auto_label(rects, ax, font_size):
	""" Attach a text label above each bar in *rects*, displaying its height. """
	for rect in rects:
		height = rect.get_height()
		ax.annotate('{:.2f}'.format(height), 
			xy=(rect.get_x() + rect.get_width() / 2, height), 
			xytext=(0, 5*np.sign(height)), # position on "top" of bar (x, y)
			textcoords="offset points", fontsize=font_size,
			ha='center', va='center_baseline')

def incremental_standard_dev(prev_std, new_vals, prev_mean, curr_mean):
	""" Calculate the standard deviation based on the previous values and update the current standard deviation.
	See here: http://datagenetics.com/blog/november22017/index.html """
	
	# use the variance to calculate incrementally, return the rooted value
	variance = math.sqrt(prev_std)
	for x in new_vals:
		variance = variance + (x-prev_mean)*(x-curr_mean)

	# return the std
	return(math.sqrt(variance/len(new_vals)))

def incremental_mean(prev_mean, new_vals):
	""" Calculate the mean based upon the previous mean and update incrementally.
	See here: http://datagenetics.com/blog/november22017/index.html  """

	# use the previous mean to incrementally update the new mean
	mean = prev_mean
	n = len(new_vals)

	for x in new_vals:
		mean = mean + (x - mean)/n

	return mean

def pie_chart(label_val_dict, ax, category=None, LABELS=None, size=0.5, font_size=9, rad=1):
	"""Pie chart constructor for given labels and sizes. This generates 'donut' pie charts with
	percentage value labelling and styling features.

	Parameters
	----------
	label_val_dict: dictionary
		the label-value paired dictionary to plot
	ax: pyplot axis
		the axis object to bind to
	category: string
		the category being plotted, if None, no title is set
	LABELS: boolean
		LABELS True sets default labels (top right), False or None sets lower center
	size: float
		controls the size of the wedges generated for the 'donut' pies
	font_size: int
		font size of labelling
	rad: float
		the radius of the pie chart. The inner radius (wedge rad) is scaled from this
	
	Returns
	----------
	fig, pyplot figure
		the generated figure object
	"""

	# initially set labels as none, update with custom legend after
	wedges, texts, autotexts = ax.pie(
		[math.fabs(x) for x in label_val_dict.values()], 
		labels=None, autopct="%1.1lf%%", 
		shadow=False, radius=rad, pctdistance=(rad+rad*0.1),
		wedgeprops=dict(width=size, edgecolor='w'))
	
	# creating the legend labels, use the label keys initially passed to us
	if LABELS is True:
		# use a bbox to set legend below pie chart for improved visibility if legend enabled
		ax.legend(wedges, loc="lower center", labels=label_val_dict.keys(), bbox_to_anchor=(1,1))
	else:
		ax.legend(wedges, loc="lower center", labels=label_val_dict.keys(), bbox_to_anchor=(rad*0.2, -0.4))

	plt.setp(autotexts, size=font_size, weight="bold")
	
	if category is not None:
		# default title
		ax.set_title(category.capitalize().replace('_', ' '), weight="bold")
	return

def bar_chart(labels, values, ax, label=None):
	"""Bar chart constructor for given labels and sizes.
	Returns the generated axis object.

	Parameters
	----------
	labels: list
		the labels to be applied to the chart
	values: list
		the values to be charted
	ax: pyplot axis
		the axis object to bind to
	label: string
		optional header title for the bar chart
	
	Returns
	----------
	fig, pyplot figure
		the generated figure object
	"""

	width = 1
	font_size = 12
	n_labels = len(labels)
	labels.reverse()
	values.reverse()

	# calculate length of x-axis then scale to match pie charts above
	x = np.arange(len(labels))
	scaled_x = [1.6*i for i in x]
	rects = ax.bar(scaled_x, values, width, color=[CMAP(i) for i in range(0,n_labels)], label=label)
	ax.set_xticks(scaled_x)
	ax.set_xticklabels([label.capitalize().replace('_', ' ') for label in labels])
	auto_label(rects, ax,font_size)
	return

def scatter_plotter(X, Y, ax, area=10, ALPHA=0.9, _cmap=CMAP):
	"""The scatterplot constructor. Generates a scatter plot with
	auto-scaling values, based on area. Also applies styling and 
	axis limiting.

	Parameters
	----------
	X: list
	Y: list
		values to be plotted to appropriate axes
	area: int
		optional, scaling value to plot a third dimension onto the graph
	ALPHA: float
		optional, sets scatter points alpha setting
	_cmap: colour map object
		optional, override the global colour map and apply a custom option
	
	Returns
	----------
	fig, pyplot figure
		the generated figure object
	"""

	if all(area) <= 1 and all(area) >= 0:
		sizing = [pow(a, -0.9) for a in area]
	else:
		sizing = [pow(a, 0.9) for a in area]

	ax.scatter(X, Y, s=sizing, c='black', cmap=_cmap, alpha=ALPHA)
	ax.set_ylim([np.asarray(Y).min(), np.asarray(Y).max()])
	ax.set_xlim([np.asarray(X).min(), np.asarray(X).max()])
	return

def normaliser(x):
	if len(x) < 2:
		raise ValueError
	else:
		pass
	def f(_x):
		return (_x-_x.min())/(_x.max()-_x.min())
	X = np.asarray(x)
	return list((map(f, X)))
