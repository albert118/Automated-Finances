# core

# third party libs
import numpy as np
import pandas as pd

# python core
import math
import os
import sys
from datetime import datetime


def stats(date_tx, curr_mean=None, curr_min=None, curr_max=None, curr_std=None, curr_tot=None) -> dict:
    """ Calculate various statistics on the account data passed to the function.
    Allow for continuous updates and integration of data.
    
    **Args:**
        date_tx(list):  A 2D array-list like object.
        curr_mean(float):   The current mean value.
        curr_min(float):    The current min value.
        curr_max(float):    The current max value.
        curr_std(float):    The current standard deviation value.
        curr_tot(float):    The current total value.

    .. note:: These are running total stats.
    
    **Returns:**
        return_dict(dict):  A nested dictionary - two stat dict's and one list of dicts: running stats dict, list of weekly stats dicts and a 4-weekly stats dict

    .. note:: The key-val args must be set if function is previously called, this\
    is required to update the running statistics on the accounts being \
    watched as new transactions are added!\

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
        running_stats['_tot'] = running_stats['_tot'] + curr_tot

    curr_date = datetime.today()
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
        min_date = datetime(curr_date.year, curr_date.month, curr_week)
        max_date = datetime(curr_date.year, curr_date.month, curr_week+7)
        
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

    return_dict = {'running_stats': running_stats,'weekly_stats': weekly_stats,'four_week_stats': four_week_stats}
    return return_dict

def incremental_standard_dev(prev_std, new_vals, prev_mean, curr_mean):
	""" Calculate the standard deviation based on the previous values and update the current standard deviation.
	See here: http://datagenetics.com/blog/november22017/index.html"""
	
	# use the variance to calculate incrementally, return the rooted value
	variance = math.sqrt(prev_std)
	for x in new_vals:
		variance = variance + (x-prev_mean)*(x-curr_mean)

	# return the std
	return(math.sqrt(variance/len(new_vals)))

def incremental_mean(prev_mean, new_vals):
	"""Calculate the mean based upon the previous mean and update incrementally.
	See here: http://datagenetics.com/blog/november22017/index.html  """

	# use the previous mean to incrementally update the new mean
	mean = prev_mean
	n = len(new_vals)

	for x in new_vals:
		mean = mean + (x - mean)/n

	return mean

def normaliser(x):
	"""Apply a simple min-max normalisation to the 1D data X."""

	if len(x) < 2:
		raise ValueError
	else:
		pass
	def f(_x):
		return (_x-_x.min())/(_x.max()-_x.min())
	X = np.asarray(x)
	return list((map(f, X)))

# def update_income_stats(self):
# 	"""
# 	This method *assumes* that if new categories are added that they are 
# 	appended, hence: previously known ordered additions of stats are in 
# 	the same index positon and keyword order

# 	Further, updates the payslip data by recalling get_payslips
# 	"""

# 	# start by refeshing the categories from local.env file
# 	i = 0
# 	for income in self.incomes:
# 		# grab the income lists (raw data)
# 		dated_txs = self.incomes[income]
# 		if len(dated_txs) == 0:
# 			continue

# 		# check the stat's info
# 		# update initial vals of our specific income stats if they dont exist
# 		# there will be as many as these as categories in self.incomes
# 		if len(self.curr_income_stats)  == 0:
# 			self.curr_income_stats = self.stats(dated_txs)
# 		else:
# 			# push updates to running_stats
# 			running_stats = self.curr_income_stats['running_stats']
# 			self.curr_income_stats = self.stats(dated_txs, *running_stats)

# 		print("update income stats: {}".format(i))
# 		i-=-1

# 	return True

# def update_savings_stats(self):
# 	"""
# 	This method *assumes* that if new categories are added that they are 
# 	appended, hence: previously known ordered additions of stats are in 
# 	the same index positon and keyword order
# 	"""
    
# 	# controls the max iterations of the stats details below
# 	num_savings_accounts = 1

# 	for i in range(0, num_savings_accounts):
# 		# grab the savings lists (raw data)
# 		dated_txs = self.savings
# 		if len(dated_txs) == 0:
# 			continue # we skip if the length is zero, avoids divide byt zero issues

# 		if len(self.curr_savings_stats)  == 0:
# 			# update initial vals of our specific savings stats if they dont exist
# 			# there will be as many as these as categories in self.savings
# 			self.curr_savings_stats = self.stats(dated_txs)
# 		else:
# 			# recalc the stats, but call the previous ones associated with 
# 			# the current subcategory for reference in incrementally 
# 			# calculating the new stats, 
# 			curr_stats = self.curr_savings_stats
# 			#i.e. grab the running_stats dict, *curr_stats[0]
# 			self.curr_savings_stats = self.stats(dated_txs, *curr_stats['running_stats'])

# 		print("update savings stats: {}".format(i))

# 	return True

# def update_expenditure_stats(self):
# 	"""
# 	This method *assumes* that if new categories are added that they are 
# 	appended, hence: previously known ordered additions of stats are in 
# 	the same index positon and keyword order
# 	"""

# 	i = 0
# 	for expenditure in self.expenditures:
# 		# grab the expenditure lists (raw data)
# 		dated_txs = self.expenditures[expenditure]
# 		if len(dated_txs) == 0:
# 			continue

# 		if len(self.curr_expenditure_stats)  == 0:
# 			# update initial vals of our specific expenditure stats if they dont exist
# 			# there will be as many as these as categories in self.incomes
# 			self.curr_expenditure_stats = self.stats(dated_txs)
# 		else:
# 			# recalc the stats, but call the previous ones associated with 
# 			# the current subcategory for reference in incrementally 
# 			# calculating the new stats, 
# 			curr_stats = self.curr_expenditure_stats
# 			#i.e. grab the running_stats dict, *curr_stats[0]
# 			self.curr_expenditure_stats = self.stats(dated_txs, *curr_stats['running_stats'])

# 		print("update expenditure stats: {}".format(i))
# 		i-=-1

# 	return True
