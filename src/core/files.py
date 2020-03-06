import os
import sys
import environ
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def pdf_maker(account):
	"""Generate the output pdf for review.
	
	Parameters
	----------
	account : dataframe_worker.AccountData
		AccountData object
	"""

	# Get the figures to save
	s = 0
	for lst in account.savings.values():
		s += len(lst)
	
	if s is 0:
		figs = [
		account.display_income_stats(),
		account.display_expenditure_stats(),
		]
		
	else:
		figs = [
		account.display_income_stats(),
		account.display_expenditure_stats(),
		account.display_savings_stats(),
		]

	with PdfPages("output.pdf") as pdf:
		for fig in figs:
			pdf.savefig(fig, bbox_inches='tight', papertype='a4')
		
		d = pdf.infodict()
		d['Title'] = 'Personal Finance Report'
		d['Author'] = 'ALBERT-DEV'
		d['Subject'] = 'personal finance review'
		d['Keywords'] = 'Finance'
		d['CreationDate'] = datetime.today()
		d['ModDate'] = datetime.today()

	os.system("output.pdf")
