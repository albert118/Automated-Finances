# core application
from accountdata import AccountData
from core import environConfig, files
import file_handler as mover
import cba_scraper
from report import Report

# third party libs
import matplotlib.pyplot as plt
import pandas as pd
from watchdog.observers import Observer

# python core
from datetime import datetime
import os
import time

# Globals
DAY = 86400
SLEEP_CTR = DAY
DEBUG = False

if DEBUG == True:
	SLEEP_CTR = 60
else:
	SLEEP_CTR = DAY

env = environConfig.safe_environ()

# setup event handlers
observer = Observer()
downloads_handler = mover.DownloadEventHandler()
# local_env_handler = mover.EnvironmentFileHandler()

# schedule event handlers
observer.schedule(downloads_handler, env("PARENT_DOWNLOAD_DIR"))
# observer.schedule(local_env_handler, BASE_DIR, recursive=False)

# now the observer watches for updates to the .env file and Downloads folder
observer.start()

try:
	while True:
		if DEBUG == True:	
			data = os.path.join("D:\Documents\GitHub\Automated-Finances\Data\Banking", "06-03-2020.csv")
			df = pd.read_csv(data, names=["Date","Tx", "Description", "Curr_Balance"])
			account = AccountData(**{"account_frame": df})
		else:
			# scrape data
			# TODO: add HTTPS security and set to headless
			# TODO: stabilise the browser interface (Chrome updates break WebDriver)
			cba_scraper.get_account_data()
			# generate account data info
			account = AccountData()

		# create pdf the old way and show it to the user
		# files.pdf_maker(account)
		# create a PDF with the ReportLab tool and show it to the user
		# sleep until next check
		data_report = Report(account)
		time.sleep(SLEEP_CTR)
except KeyboardInterrupt:
	observer.stop()

observer.join()
