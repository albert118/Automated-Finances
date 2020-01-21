""" Preprocessing on the incoming dataframes: Netbanking, Payslips [+add more later]. 

Example usage, 
>>> import pandas as pd
>>> import dataframe_worker as w
>>> df = pd.read_csv("CSVData.csv", names=["Date","Tx", "Description", "Curr_Balance"])
>>> account = w.account_data(df)


import pandas as pd
import dataframe_worker as w
df = pd.read_csv("CSVData.csv", names=["Date","Tx", "Description", "Curr_Balance"])
a = w.account_data(df)

a.display_expenditure_stats()
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import environ
from datetime import date
import math
import os

CMAP =  plt.get_cmap('Paired')

class account_data():
	"""
	IDEA: Monthly overview of the account,
		* incoming review; avg., total p/mnth, total p each week, by category:
		* outgoing overview; total, avg., by category (coffee, shopping, eating, groceries, utilities, health)
		* savings overview: avg., total p/mnth, total p each week
		* opening balance : closing balance : delta
	 """
	
	# grab local environ settings and attempt to read settings file
	
	env = environ.Env()
	env_file = os.path.join(os.getcwd(), "local.env")

	if os.path.exists(env_file):
		env.read_env(env_file) 
	else: 
		raise Exception("Local environ settings file not found, please check file location dir") 

	###############################################################
	# Categories and Search terms, TODO: Move to a settings file
	###############################################################
	
	INCOME = {
		'primary_income': env.str("primary_income"), 
		'supplemental_income': env.str("supplemental_income"), 
		'investment_income': env.str("investment_income")
		}

	SAVINGS_IDS = [env("CACHE_ID"), env("SAVINGS_ID")]

	SUBSCRIPTIONS = env.list("SUBSCRIPTIONS")
	UTILITIES = env.list("UTILITIES")
	GROCERIES = env.list("GROCERIES")
	HEALTH = env.list("HEALTH")
	EATING_OUT = env.list("EATING_OUT")
	COFFEE = env.list("COFFEE")
	SHOPPING = env.list("SHOPPING")
	ENTERTAINMENT = env.list("ENTERTAINMENT")

	EXPENDITURES = { 
		'utilities': UTILITIES,
		'health': HEALTH,
		'eating_out': EATING_OUT,
		'coffee': COFFEE,
		'subscriptions': SUBSCRIPTIONS,
		'groceries': GROCERIES,
	}

	def __init__(self, account_frame):
	     account_frame.Description = account_frame.Description.apply(str.upper)
	     account_frame.Date=pd.to_datetime(account_frame.Date, format="%d/%m/%Y")

	     self.incomes = self.get_income(account_frame)
	     self.savings = self.get_savings(account_frame)
	     self.expenditures = self.get_expenditures(account_frame)

	     # we need to maintain a list of stats for every sub category and its relevant stats dicts/lists
	     # dynamically configure this based on cat's defined for the class above...
	     self.curr_income_stats = []
	     self.curr_savings_stats = []
	     self.curr_expenditure_stats = []

	###############################################################
	# Finders and Getters
	###############################################################
	
	def get_income(self, acc_frame):
		date_income, date_supplemental, date_investment = ([] for i in range(3))

		incomes = {
			'primary_income': date_income,
			'supplemental_income': date_supplemental,
			'investment_income': date_investment
			}

		for key, term in self.INCOME.items():
			for i in range(0, len(acc_frame)):
				if str(term).strip() in acc_frame.Description[i]:
					incomes[key].append([acc_frame.Date[i], acc_frame.Tx[i]])

		return incomes

	def get_savings(self, acc_frame):
		date_savings = []
		for i in range(0, len(acc_frame)):
			# tx for savings always includes the acc_id ref
			for _id in self.SAVINGS_IDS:

				# test for outgoing as well as unique ref id
				if _id in acc_frame.Description[i] and acc_frame.Tx[i] > 0:
					date_savings.append([acc_frame.Date[i], acc_frame.Tx[i]])

		return date_savings

	def get_expenditures(self, acc_frame):
		date_utilities, date_health, date_eat, date_coffee, date_subs, date_groceries = ([] for i in range(6))
		n = len(acc_frame)

		expenditures = {
			'utilities': date_utilities,
			'health': date_health,
			'eating_out': date_eat,
			'coffee': date_coffee,
			'subscriptions': date_subs,
			'groceries': date_groceries,
		}

		# iterate through cateogries
		for key, term_list in self.EXPENDITURES.items():
			# iterate through dataframe
			for i in range(0, n):
				# iterate through each sub-cat term in term_list
				for sub_cat in term_list:
					# print("Cat: " , sub_cat.upper(), "\nVal: ", acc_frame.Description[i])
					
					# INSTRUMENTAL TO BS NOT FINDING CAT'S IS INCLUDING THE STRIP FUNCTION!!!
					search_term = str(sub_cat.upper()).strip()
					search_field = str(acc_frame.Description[i].upper())
					idx = search_field.find(search_term)
					
					if idx != -1:
						expenditures[key].append([acc_frame.Date[i], acc_frame.Tx[i], search_term])
						#print([acc_frame.Date[i], acc_frame.Tx[i]])
						#input()
		return expenditures

	###############################################################
	# Displayers and updaters
	###############################################################
	
	def update_income_stats(self):
		"""
		this method *assumes* that if new categories are added that they are 
		appended, hence: previously known ordered additions of stats are in 
		the same index positon and keyword order
		"""

		i = 0
		for income in self.incomes:
			# grab the income lists (raw data)
			dated_txs = self.incomes[income]
			if len(dated_txs) == 0:
				continue

			if len(self.curr_income_stats)  == 0:
				# update initial vals of our specific income stats if they dont exist
				# there will be as many as these as categories in self.incomes
				self.curr_income_stats = self.stats(dated_txs)
			else:
				# recalc the stats, but call the previous ones associated with 
				# the current subcategory for reference in incrementally 
				# calculating the new stats, 
				curr_stats = self.curr_income_stats
				#i.e. grab the running_stats dict, *curr_stats[0]
				self.curr_income_stats = self.stats(dated_txs, *curr_stats['running_stats'])

			print("update income stats: {}".format(i))
			i-=-1

		return True
	
	def update_savings_stats(self):
		"""
		this method *assumes* that if new categories are added that they are 
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
		this method *assumes* that if new categories are added that they are 
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

	def display_income_stats(self):
		""" Display some visualisations and print outs of the income data. """

		# bar graph of income, list conversion needed as Tkinter fucks itself if it sees a numpy array...
		income_raw = list(np.array(self.incomes['primary_income'])[:,1])
		
		labels = []
		for i in range(len(income_raw)):
			labels.append("Week {}".format(i) )

		width=0.35
		x = np.arange(len(labels))
		fig, ax = plt.subplots()
		rects_income = ax.bar(x, income_raw, width, label="Primary Income")

		ax.set_ylabel('Income')
		ax.set_xlabel('Category of Income')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend()

		auto_label(rects_income, ax)
		fig.tight_layout()
		plt.show()

		# later, add paycheck stuff here too - ADP does it well, do the same pie-chart
		# and check MoneyTree, great visualisations on that too
		return True

	def display_savings_stats(self):
		pass

	def display_expenditure_stats(self):
		""" Display some visualisations and print outs of the income data. """
		# Generate a pie chart of expenditures
		# Generate a bar chart of each category vs. total budget

		totals = []		
		# create the outer subplot that will hold the boxplot and subplots
		fig = plt.figure()
		outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)

		inner_top = gridspec.GridSpecFromSubplotSpec(1, len(self.expenditures.keys()), subplot_spec=outer[0],
					wspace=0.1, hspace=0.1)
		inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1],
					wspace=0.1, hspace=0.1)

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
			axN = plt.Subplot(fig, inner_top[key_counter])
			
			axN.set_prop_cycle(color=[CMAP(i) for i in range(1,10)])
			
			pie_chart(labels, axN, category=key)
			fig.add_subplot(axN)

			totals.append(sum(labels.values()))
			key_counter -=- 1

		ax_rect = plt.Subplot(fig, inner_bottom[0])
		bar_chart(list(self.expenditures.keys()), totals, ax_rect)
		
		ax_rect.set_ylabel('Expenditure')
		ax_rect.set_xlabel('Category of Expenditure')
		fig.add_subplot(ax_rect)

		plt.show()

		# later, add paycheck stuff here too - ADP does it well, do the same pie-chart
		# and check MoneyTree, great visualisations on that too
		return True


	###############################################################
	# Stats
	###############################################################

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

###############################################################
# Utility
###############################################################

def auto_label(rects, ax, font_size):
	""" Attach a text label above each bar in *rects*, displaying its height. """
	for rect in rects:
		height = rect.get_height()
		ax.annotate('{}'.format(height), 
			xy=(rect.get_x() + rect.get_width() / 2, height), 
			xytext=(80, 0), # position on "top" of bar (x, y)
			textcoords="offset points", fontsize=font_size,
			ha='center', va='bottom')

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

def pie_chart(label_val_dict, ax, category=None):
	""" Pie chart constructor for given labels and sizes.
	Returns the generated figure and axis objects. """

	#cmap = plt.get_cmap("tab20c")
	#colours = cmap(np.array(len(labels))*4)
	size = 0.5 # setting for controlling width of wedges, creates donut shape
	font_size = 9
	rad = 1
	print(label_val_dict)

	# initially set labels as none, update with custom legend after
	wedges, texts, autotexts = ax.pie(
		[math.fabs(x) for x in label_val_dict.values()], 
		labels=None, autopct="%1.1lf%%", 
		shadow=False, radius=rad, pctdistance=(rad+rad*0.1),
		wedgeprops=dict(width=size, edgecolor='w'))
	
	# creating the legend labels, use the label keys that would have been
	# initially passed to pie()
	# use a bbox to set legend below pie chart for improved visibility
	ax.legend(wedges, loc="lower center", labels=label_val_dict.keys(), bbox_to_anchor=(rad*0.2, -0.4))
	plt.setp(autotexts, size=font_size, weight="bold")
	
	if category is not None:
		ax.set_title(category.capitalize().replace('_', ' '), weight="bold")
	else:
		pass

	# DEPRECIATED! This is done within the pie() method automatically now
	# Equal aspect ratio ensures that pie is drawn as a circle.
	# ax.axis("equal") 

	return

def bar_chart(labels, values, ax):
	""" Bar chart constructor for given labels and sizes.
	Returns the generated axis object. """
	
	width = 1
	font_size = 12
	
	# calculate length of x-axis then scale to match pie charts above
	x = np.arange(len(labels))
	scaled_x = [1.6*i for i in x]
	rects = ax.bar(scaled_x, values, width, color=[CMAP(i) for i in range(2,len(labels))])

	ax.set_xticks(scaled_x)
	ax.set_xticklabels([label.capitalize().replace('_', ' ') for label in labels])
	auto_label(rects, ax,font_size)

	return

