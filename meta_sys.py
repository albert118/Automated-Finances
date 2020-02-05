"""Script runner, this will display several charts on the income, savings and spending areas."""

import pandas as pd
import dataframe_worker as w
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

df = pd.read_csv("CSVData.csv", names=["Date","Tx", "Description", "Curr_Balance"])
account = w.account_data(df)

# figsize = (50, 50)
# fig_size = plt.gcf().get_size_inches() # getting current size
# factor = 2
# plt.gcf().set_size_inches(factor*fig_size)
# plt.subplots_adjust(bottom=-10)

figs = [
	account.display_income_stats(),
	account.display_expenditure_stats(),
	account.display_savings_stats(),
]

# plt.show()

# generate the output pdf
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

