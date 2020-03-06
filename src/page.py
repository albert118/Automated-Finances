# user defined
from core import environConfig

# third party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# core
from datetime import datetime
import math
import os
import sys


class Page():

    # magic methods
    def __init__(self):
		# create a GridSpec to hold our page data
		fig = plt.figure(figsize=(10,10))
		outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,1])
		col_count = math.ceil(len(self.expenditures.keys())/2)
		inner_top = gridspec.GridSpecFromSubplotSpec(2, col_count, subplot_spec=outer[0],
					wspace=0.2, hspace=0.2)
		inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1],
					wspace=0.05, hspace=0.1)

        # axN = fig.add_subplot(inner_top[0, key_counter]) # this is also one of the cleaner ways to create the axis
        # axN.set_prop_cycle(color=[CMAP(i) for i in range(1,10)])
        # pie_chart(label_vals.keys(), label_vals.values(), axN, category=key)
		# plt.suptitle("Expenditure Statistics")
		# ax_rect = fig.add_subplot(inner_bottom[0])
		# bar_chart(list(self.expenditures.keys()), totals, ax_rect)
		# ax_rect.set_ylabel('Expenditure')
		# ax_rect.set_xlabel('Category of Expenditure')
		# fig.add_subplot(ax_rect)

        pass
    def __str__(self):
        pass
    def __del__(self):
        pass

    # getters/setters/updaters

    # class methods
    def add_graph(self):
        pass
    def add_text(self):
        pass
    def add_overlay(self):
        pass
