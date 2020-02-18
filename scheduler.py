import dataframe_worker
import file_handler as mover
import cba_scraper
import utilities

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from watchdog.observers import Observer

from datetime import datetime
import os
import time

INPUT_FOLDER =  "D:\\Downloads"
BASE_DIR = os.getcwd()
SLEEP_CTR = 86400

# setup event handlers
downloads_handler = mover.DownloadEventHandler()
local_env_handler = mover.EnvironmentFileHandler()

observer = Observer()

# schedule event handlers
observer.schedule(downloads_handler, INPUT_FOLDER, recursive=True)
observer.schedule(local_env_handler, BASE_DIR, recursive=False)

# now the observer watches for updates to the .env file and Downloads folder
observer.start()
try:
	while True:
		time.sleep(SLEEP_CTR)
		cba_scraper.get_account_data()
		df = pd.read_csv("CSVData.csv", names=["Date","Tx", "Description", "Curr_Balance"])
		account = dataframe_worker.AccountData(df)
		utilities.pdf_maker(account)
except KeyboardInterrupt:
	observer.stop()

observer.join()
