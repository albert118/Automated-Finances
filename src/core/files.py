# core
from core import environConfig

# third party libs
from matplotlib.backends.backend_pdf import PdfPages

# python core
import os
import sys
from datetime import datetime
from io import BytesIO

# app modules
from accountdata import AccountData

def pdf_maker(account: AccountData):
	"""Generate the output pdf for review.
	
	**Args:**
		account(AccountData):	AccountData object

	"""

	env = environConfig.safe_environ()

	# Get the figures to save
	savings_lists = [len(lst) for lst in account.savings.values()]
	savings_sum = sum(savings_lists)
	
	if savings_sum is 0:
		figs = [ 
			account.display_income_stats(), account.display_expenditure_stats()]
	else:
		figs = [
		account.display_income_stats(),
		account.display_expenditure_stats(),
		account.display_savings_stats(),
		]
 
	par_dir = os.path.abspath(env("PARENT_DIR"))
	out_fn = os.path.normpath("data/output.pdf")
	save_dir = os.path.join(par_dir, out_fn)

	with PdfPages(save_dir) as pdf:
		for fig in figs:
			pdf.savefig(fig, bbox_inches='tight', papertype='a4')
		
		d = pdf.infodict()
		d['Title'] = 'Personal Finance Report'
		d['Author'] = 'ALBERT-DEV'
		d['Subject'] = 'personal finance review'
		d['Keywords'] = 'Finance'
		d['CreationDate'] = datetime.today()
		d['ModDate'] = datetime.today()

	os.system(save_dir)
