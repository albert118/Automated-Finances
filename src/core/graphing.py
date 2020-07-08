# core

# third party libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# python core
import math
import os
import sys
from datetime import datetime

CMAP =  plt.get_cmap('Paired') # Global colour map variable

def auto_label(rects, ax, font_size):
	""" Attach a text label above each bar in *rects*, displaying its height. """
	for rect in rects:
		height = rect.get_height()
		ax.annotate('{:.2f}'.format(height), 
			xy=(rect.get_x() + rect.get_width() / 2, height), 
			xytext=(0, 5*np.sign(height)), # position on "top" of bar (x, y)
			textcoords="offset points", fontsize=font_size,
			ha='center', va='center_baseline')

def pie_chart(labels, values, ax, category=None, LABELS=None, size=0.5, font_size=9, rad=1):
	"""Pie chart constructor with custom design.
	
	Pie chart constructor for given labels and sizes. This generates 'donut' pie charts with
	percentage value labelling and styling features.

	Parameters
	----------
	labels : list : str
		list of string labels for the wedges
	
	values : list : float
		list of float values to create chart with

	ax : matplotlib.axes
		the axis object to bind to

	category : str
		the category being plotted, if None, no title is set

	LABELS : Boolean
		LABELS True sets default labels (top right), False or None sets lower center

	size : float
		controls the size of the wedges generated for the 'donut' pies

	font_size : int
		font size of labelling

	rad : float
		the radius of the pie chart. The inner radius (wedge rad) is scaled from this
	"""

	# initially set labels as none, update with custom legend after
	wedges, texts, autotexts = ax.pie(
		[math.fabs(x) for x in values], 
		labels=None, autopct="%1.1lf%%", 
		shadow=False, radius=rad, pctdistance=(rad+rad*0.1),
		wedgeprops=dict(width=size, edgecolor='w'))
	
	# creating the legend labels, use the label keys initially passed to us
	if LABELS is True:
		# use a bbox to set legend below pie chart for improved visibility if legend enabled
		ax.legend(wedges, loc="lower center", labels=labels, bbox_to_anchor=(1,1))
	else:
		ax.legend(wedges, loc="lower center", labels=labels, bbox_to_anchor=(rad*0.2, -0.4))

	plt.setp(autotexts, size=font_size, weight="bold")
	
	if category is not None:
		# default title
		ax.set_title(category.capitalize().replace('_', ' '), weight="bold")
	return

def bar_chart(labels, values, ax, label=None):
	"""Bar chart constructor for given labels and sizes.

	Parameters
	----------
	labels: list
		the labels to be applied to the chart

	values: list
		the values to be charted

	ax: matplotlib.axes
		the axis object to bind to

	label: str
		optional header title for the bar chart
	"""

	width = 1
	font_size = 12
	n_labels = len(labels)
	labels.reverse()
	values.reverse()

	# calculate length of x-axis then scale to match pie charts above
	x = np.arange(len(labels))
	scaled_x = [1.6*i for i in x]
	rects = ax.bar(scaled_x, values, width, color=[CMAP(i) for i in range(0,n_labels)], label=label)
	ax.set_xticks(scaled_x)
	ax.set_xticklabels([label.capitalize().replace('_', ' ') for label in labels])
	auto_label(rects, ax,font_size)

def scatter_plotter(X, Y, ax, area=10, ALPHA=0.9, _cmap=CMAP):
	"""Scatter plot constructor for given data and custom design.
	
	Generates a scatter plot with auto-scaling values, based on area. 
	Also applies styling and axis limiting.

	Parameters
	----------
	X: list
	Y: list
		values to be plotted to appropriate axes

	area: int
		optional, scaling value to plot a third dimension onto the graph
	ALPHA: float
		optional, sets scatter points alpha setting

	_cmap: colour map object
		optional, override the global colour map and apply a custom option
	"""
	try:
		ax.scatter(X, Y, c='black', cmap=_cmap, alpha=ALPHA)
		if len(Y) is 1:
			ax.set_ylim([Y[0]*0.9, Y[0]*1.1])
		else:
			ax.set_ylim([np.asarray(Y).min(), np.asarray(Y).max()])
		if len(X) is 1:
			pass
		else:
			ax.set_xlim([np.asarray(X).min(), np.asarray(X).max()])
	except (TypeError, Exception):
		ax.scatter(X, Y, c='black', cmap=_cmap, alpha=ALPHA)
