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
	050 NSW -  Sydney1 OF1R1 EN ENTERTAINMENT SALEJB0JB- Sydney\n\n\n 
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
	' 050 NSW -  Sydney1 OF1R1 EN ENTERTAINMENT SALEJB0JB- Sydney', ' Late Night23.13003.000069.39COMMISSIONB95.09', 
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
