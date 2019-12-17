""" The scraper for grabbing data from Netbank and Super portals """

from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common import exceptions

import pandas as pd
import os
import environ


def get_acc_export():

	try:
		driver.implicitly_wait(5)
		txnExport_link = driver.find_element_by_xpath('//*[@id="ctl00_CustomFooter"]/div/div/div[3]')
		txnExport_link = driver.find_element_by_xpath('//*[@id="ctl00_CustomFooterContentPlaceHolder_updatePanelExport1"]/div/a').click()
		print("Click: Export link")
		driver.implicitly_wait(2)

	except exceptions.StaleElementReferenceException as e:
		print(e)
		print("Click: Export link try two")
		driver.implicitly_wait(5)
		txnExport_link = driver.find_element_by_xpath('//*[@id="ctl00_CustomFooter"]/div/div/div[3]')
		txnExport_link = driver.find_element_by_xpath('//*[@id="ctl00_CustomFooterContentPlaceHolder_updatePanelExport1"]/div/a').click()
		driver.implicitly_wait(2)

	try:
		type_ddl = driver.find_element_by_xpath('//*[@id="ctl00_CustomFooterContentPlaceHolder_updatePanelExport1"]/div/div/fieldset/div[2]/div[1]').click()
		print("Click: Type ddl")

	except exceptions.StaleElementReferenceException as e:
		driver.implicitly_wait(2)
		print(e)
		type_ddl = driver.find_element_by_xpath('//*[@id="ctl00_CustomFooterContentPlaceHolder_updatePanelExport1"]/div/div/fieldset/div[2]/div[1]').click()
		print("Click: Type ddl try two")

	try:
		type_ddl = driver.find_element_by_xpath('//*[@id="ctl00_CustomFooterContentPlaceHolder_updatePanelExport1"]/div/div/fieldset/div[2]/div[1]/div/div[2]/span/span/span[2]/ul/li[2]/a/span').click()
		print("Click: Export Option")
		
	except exceptions.StaleElementReferenceException as e:
		driver.implicitly_wait(2)
		print(e)
		type_ddl = driver.find_element_by_xpath('//*[@id="ctl00_CustomFooterContentPlaceHolder_updatePanelExport1"]/div/div/fieldset/div[2]/div[1]/div/div[2]/span/span/span[2]/ul/li[2]/a/span').click()
		print("Click: Export Option try two")

	try:
		export_transactions = driver.find_element_by_id("ctl00_CustomFooterContentPlaceHolder_lbExport1").click()
		print("Click: Exporting Transactions")

	except exceptions.StaleElementReferenceException as e:
		print(e)
		driver.implicitly_wait(10)
		export_transactions = driver.find_element_by_xpath('/html/body/form/div[6]/div/div/div[3]/div/div/div/fieldset/div[2]/div[2]/a').click()

	return


def get_account_data():
	""" Use Selenium and the Chrome WebDriver to access each account and download the latest CSV data of their transactions. """

	env = environ.Env()
	env_file = os.path.join(os.getcwd(), "local.env")

	if os.path.exists(env_file):
		env.read_env(env_file)
	
	# path to the ChromeDriver (WebDriver)
	webdriver = os.path.abspath(env("WEB_DRIVER"))
	URL_NETBANK_LOGIN = env("URL_NETBANK_LOGIN")
	NETBANK_ID = env("NETBANK_ID")
	NETBANK_PASS = env("NETBANK_PASS")

	if NETBANK_ID is None or NETBANK_PASS is None:
		raise Exception('Please check the local environ file is accessible and login details are correct.')

	with Chrome(webdriver) as driver:
		
		driver.maximize_window()
		if URL_NETBANK_LOGIN is None:
			raise Exception("The CBA login URL was not found, please check the local environ file.")

		print("Accessing: " + URL_NETBANK_LOGIN + "\n")
		driver.get(URL_NETBANK_LOGIN) # get req
		
		# element names are set in environ file, if frontend is changed by CBA, edit the change there!
		_FIELD_USR = env("_FIELD_USR")
		_FIELD_PSWD = env("_FIELD_PSWD")
		_FIELD_LGON = env("_FIELD_LGON")

		usr = driver.find_element_by_name(_FIELD_USR)
		usr.send_keys(NETBANK_ID)
		pswd = driver.find_element_by_name(_FIELD_PSWD)
		pswd.send_keys(NETBANK_PASS)
		
		login = driver.find_element_by_name(_FIELD_LGON)
		login.click()
		driver.implicitly_wait(2) # wait so CSS exists, otherwise we may load too quick and miss view accounts
		
		# interface fuckery to get through the drop down list (ddl) and access each account
		view_accounts = driver.find_element_by_link_text("View accounts").click()
		print("Cick: View accounts")
		
		label_ddl = driver.find_element_by_id("ctl00_ContentHeaderPlaceHolder_ddlAccount_field")
		print("Click: focussing on ddl")
		accounts_ddl = driver.find_element_by_class_name("ddl_selected").click()
		
		smart_access_account = driver.find_element_by_class_name("option1")
		print("Click: Smart Access Account")
		smart_access_account.click()

		# access the CSV and download it once the account is selected...this is a fucky UI process. DO NOT TOUCH IT
		get_acc_export()
		
		return


get_account_data()
