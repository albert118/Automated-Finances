import os
import sys
import environ
from matplotlib.backends.backend_pdf import PdfPages

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


def pdf_maker(account):
	"""Generate the output pdf for review.
	
	Parameters
	----------
	account : dataframe_worker.AccountData
		AccountData object
	"""

	# Get the figures to save
	if len(account.savings) == 0:
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
