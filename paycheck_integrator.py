"""
Paycheck integrations with ADP payroll.

TBH this looks like a bitch with all of the verification and security involved....


TLDR; download the PDFs of payslip reports and scrape them for data with PyPDF2.
Downwside, this still requires a Selenium script....


1. Get pdfs with selenium scraper
2. add details to dataframe worker for extracting their text contents
	pdf = open('payslip.pdf', 'rb')
	rdr = PyPDF2.PdfFileReader(pdf)
	data = rdr.getPage(0).extractText()

	>>> data

	'050050\n\n
	MR ALBERT FERGUSON\n
	3/29 MURIEL STREET\n
	HORNSBYNSW2077\n
	Australia\n\n\n\n\n\n\n\n 
	Weekto\n 26/01/202028/01/2020049610FERGUSON, ALBERTCasual\n\n 
	050 NSW - CHATSWOOD1 OF1R1 EN ENTERTAINMENT SALEJB0JB-CHATSWOOD\n\n\n 
	Late Night23.13003.000069.39COMMISSIONB95.09\n 
	CAS21.410012.5000267.63*Rest SuperE61.70\n
	Casual Sat24.84008.7500217.35MEDECINS SANS FRONTIERB-1.00\n\n\n\n\n\n\n\n\n\n\n\n\n\n
	649.46648.4694.09.00181.00467.46\n
	* Employer Superannuation Contribution relates to period commencing 30/12/2019 up to 26/01/2020\n\n\n 
	EFT111278781 062-140 CBA467.46 Annual lve.00 (.0).00 (.0).00 (.0)\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
	24092.4324061.4331.007045.0017016.43\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n 
	JB HI FI GROUP PTY LTD(ABN) 37093114286\n'

Note: newline char is marker for data, use it as a counter to grab/split data

	data = data.split('\n')
	while '' in data:
		data.remove('')
	>>> data
	['050050', 'MR ALBERT FERGUSON', '3/29 MURIEL STREET', 
	'HORNSBYNSW2077', 'Australia', ' Weekto', ' 26/01/202028/01/2020049610FERGUSON, ALBERTCasual', 
	' 050 NSW - CHATSWOOD1 OF1R1 EN ENTERTAINMENT SALEJB0JB-CHATSWOOD', ' Late Night23.13003.000069.39COMMISSIONB95.09', 
	' CAS21.410012.5000267.63*Rest SuperE61.70', ' Casual Sat24.84008.7500217.35MEDECINS SANS FRONTIERB-1.00', 
	'649.46 648.46 94.09 .00 181.00 467.46', '* Employer Superannuation Contribution relates to period commencing 30/12/2019 up to 26/01/2020', 
	' EFT111278781 062-140 CBA467.46 Annual lve.00 (.0).00 (.0).00 (.0)', '24092.4324061.4331.007045.0017016.43', ' JB HI FI GROUP PTY LTD(ABN) 37093114286']
	
Note: array now contains relevant data at const indexes, but each index with useful data includes 7 columns of data...
Data in format:
	string Desc. double Rate double Hours double Value string Desc. chart Tax Ind. double Value
	>>> data[8:12]
Note: data[12] is the Summary of Earnings table of format: double Gross, double Taxable, double Pre Tax Allows/Deds, double Post Tax Allows/Deds, double Tax, double Net Income




	from tabula import read_pdf as r
	df = r('payslip.pdf')

Returns a list of two df's. First is primary data, second is Summary details of year to date

	inc_data = df[0]
	inc_data = inc_data.drop("Unnamed: 0", axis=1)
>>> inc_dat[4:12]
     Pay Period Pay Date Emp No.                                      Name               Unnamed: 1 Unnamed: 2 Status  Unnamed: 3
4           NaN         ELEMENTS                                       NaN  ALLOWANCES / DEDUCTIONS        NaN    NaN         NaN
5   Description       Rate Hours                         Value Description                      NaN    Tax Ind    NaN       Value
6    Late Night   23.1300 3.0000                          69.39 COMMISSION                      NaN          B    NaN       95.09
7           CAS  21.4100 12.5000                        267.63 *Rest Super                      NaN          E    NaN       61.70
8    Casual Sat   24.8400 8.7500             217.35 MEDECINS SANS FRONTIER                      NaN          B    NaN       -1.00
9           NaN              NaN                       SUMMARY OF EARNINGS                      NaN        NaN    NaN         NaN
10        Gross   Taxable Income  Pre Tax Allows/Deds Post Tax Allows/Deds                      NaN        Tax    NaN  NET INCOME
11       649.46           648.46                                     94.09                      .00     181.00    NaN      467.46

"""

import pandas as pd
import numpy as np
from tabula import read_pdf as r
import traceback
import warnings

def getter(payslip_name='payslip.pdf', true_col_header_index = 5):
			"""Retreive the pdf, convert to dataframes for income stats and aggregate
			income data. true_col_header_index is the constant value with the first
			data frames actual headers that require extraction. This value is used
			to relatively find further dataframes from the pdf.

			----------------------------------------------------------------------------
			Dataframe titles of note: 
				* income_stats_data is the dataframe with hourly data, commissions and 
					deductions
					[Description, Rate, Hours, Value] [Description, Tax Index, Value]
				* income_data is the aggregate income values 
					[Gross, Taxable, Ded's pre and post, Tax, Net]

			"""
			
			# retrieve the payslip as a dataframe
			df = r(payslip_name)

			def rename_headers(dataframe, header_index, cols_default_headers):
					""" Rename the column headers from default guesses to the correct values.
					Also performs some housekeeping by reindexing and dropping the header row. """
					#with warnings.catch_warnings(record=True) as w:
						# set warnings to error to catch potential issue with rename
						# copying a slice of the frame
						# dataframe.is_copy = False is a fix, but not past Pandas 1.0.0
						#warnings.filterwarnings("error")
						
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

					# except Warning as w:
					# 	traceback.print_stack()
					# 	print(w, '\n')
					# 	input()
					# 	return None

					except Exception as e:
						print("An unknown exception occured in renaming the headers of the frame.\n")
						print(type(e), '\n')
						print('Current frame:\n', dataframe, '\n')
						input()
						return

			def split_merged_columns(dataframe):
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
							print(bool_arr)
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
					
				return dataframe

			# drop the NaN column generated on import and get the default column titles
			data = df[0]
			data = data.drop(["Unnamed: 0", "Status"], axis=1)
			cols_default_headers = data.columns.values
			
			income_data = pd.DataFrame()
			income_stats_data = pd.DataFrame()
			income_data_header_idx = None

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
					income_stats_data = rename_headers(income_stats_data, true_col_header_index, cols_default_headers)
					income_data = rename_headers(income_data, income_data_header_idx, cols_default_headers)


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
				print(income_stats_data)
				print(income_data)
				income_stats_data = split_merged_columns(income_stats_data)
	
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
	
df1, df2 = getter()
print(df1, df2)