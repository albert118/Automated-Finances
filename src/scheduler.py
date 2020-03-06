

import dataframe_worker
import file_handler as mover
import cba_scraper
from core import environConfig, files

import matplotlib.pyplot as plt
import pandas as pd
from watchdog.observers import Observer

from datetime import datetime
import os
import time

DAY = 86400
SLEEP_CTR = DAY
DEBUG = True

if DEBUG == True:
	SLEEP_CTR = 10
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
			data = os.path.join("D:\Downloads\Finance\Banking", "18-02-2020.csv")
			df = pd.read_csv(data, names=["Date","Tx", "Description", "Curr_Balance"])
			account = dataframe_worker.AccountData(**{"account_frame": df})
		else:
			# scrape data
			# TODO: add HTTPS to scraper and hide window
			cba_scraper.get_account_data()
			# generate account data info
			account = dataframe_worker.AccountData()
		# create pdf and show it to user
		files.pdf_maker(account)
		# sleep until next check
		time.sleep(SLEEP_CTR)
except KeyboardInterrupt:
	observer.stop()

observer.join()
