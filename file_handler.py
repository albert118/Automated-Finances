from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import utilities

import os
from datetime import datetime

CRITS = []
MIMES = []
PARENT_DOWNLOAD_DIR = ''
DOWN_SUB_FOLDERS = []

###############################################################
# Utility functions
###############################################################

def update_file_mover_globals():
	"""If changes to local.env file detected, update the globals"""
	env = utilities.safe_environ()
	global CRITS 
	global MIMES
	global PARENT_DOWNLOAD_DIR
	global DOWN_SUB_FOLDERS

	try:
		CRITS = env.list("CRITS")
		MIMES = env.list("MIMES")
		PARENT_DOWNLOAD_DIR = str(env("PARENT_DOWNLOAD_DIR"))
		DOWN_SUB_FOLDERS = env.list("DOWN_SUB_FOLDERS")
	except KeyError:
		return False
	
	return True

def check_match_crit(filename):
	"""Check for matching terms in the filename. 
	
	Parameters:
	----------
	filename : str
		The filename including .type to check
	
	Returns
	-------
	crit_comp : str
		from CRITS Global

	False : Boolean
		If criteria were not found in CRITS Global
	"""

	try:
		crit_comp = CRITS.index(filename)
	except ValueError:
		return False
	else:
		return CRITS[crit_comp]

def file_exists(file_dest, filename, duplicate_file_ctr):
	"""Check if the exisiting filename, incl. dir, exists.
	
	Iterate until a unique name is found.

	Parameters:
	----------
	filename : str
		The filename including .type to check
	
	duplicate_file_ctr : int
		The duplicate file counter variable
	
	file_dest : str
		string resolving to os.path of target destination of new file

	Returns
	-------
	new_name : os.path
		new unique filename incl. file dest dir
	"""
	
	try:
		date = datetime.now().strftime("%d-%m-%Y")
		curr_type = filename[filename.find('.'):]
		new_name = date + curr_type
		file_exists = os.path.isfile(os.path.join(file_dest, new_name))

		while file_exists:
			# handle duplicate named files by incrementing a counter
			duplicate_file_ctr -=- 1	
			new_name = date + "_" + str(duplicate_file_ctr) + curr_type
			file_exists = os.path.isfile(os.path.join(file_dest, new_name))
		return os.path.join(file_dest, new_name)
	except OverflowError:
		if duplicate_file_ctr > (10**5):
			duplicate_file_ctr = 1 
		else:
			# file naming exceeding maximum length for directory in OS
			fn = curr_type.strip('.') + duplicate_file_ctr + curr_type
		# recursive solve
		return file_exists(file_dest, fn, duplicate_file_ctr)
	except RecursionError:
		return os.path.join(file_dest, "ERROR"+curr_type)
		

###############################################################
# Event handling logic
###############################################################
class EnvironmentFileHandler(FileSystemEventHandler):
	"""The local environment handler object"""

	def on_modified(self, event):
		"""Modification of the target directory folder_to_track runs this."""

		update_file_mover_globals()

class DownloadEventHandler(FileSystemEventHandler):
	"""The download event handler object"""

	def on_modified(self, event):
		""" Collect subfolders in the target directory and sub_folders listed. 

		Dynamically observe new file's type and any matching criteria. Move them
		to the new target folder for later retrieval.

		Default filname is __unkown__.uknw
		
		If a recursion error occurs due to filename becoming too long,
		the filename is set to ERROR.[original_file_type]
		
		Duplicate files are renamed with a counter: 1,2,3 ... n
		"""
		update_file_mover_globals()
		folder_to_track = PARENT_DOWNLOAD_DIR
		down_subfolders = DOWN_SUB_FOLDERS
		target_folders = []

		# collect the target folders
		for folder in down_subfolders:
			target_folders.append(os.path.join(folder_to_track, folder))
		
		# default_fn = "__unknown__.uknw"

		for filename in os.listdir(folder_to_track):
			# first check the filetypes for mimetypes
			f_type = filename[filename.find('.'):]
			if f_type not in MIMES:
				continue

			# now check if we have some matching criteria to sort by
			new_name = filename
			# key_word = check_match_crit(filename)
			
			# if not key_word:
			# 	new_name = default_fn

			if f_type == MIMES[0]:
				target_save_dir = target_folders[0]
			elif f_type == MIMES[1]:
				target_save_dir = target_folders[1]
			else: 
				target_save_dir = os.path.join(folder_to_track, "Finance")

			duplicate_file_ctr = 1
			# create the new file destination, checking for duplicates and including a datetime value
			new_destination = file_exists(target_save_dir, new_name, duplicate_file_ctr)
			src = os.path.join(folder_to_track, filename)
			try:
				os.rename(src, new_destination)
			except NotImplementedError:
				return False
			except TypeError:
				return False
