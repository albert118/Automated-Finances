"""
.. module:: graphing
    :platform: Unix, Windows
    :synopsis: Graphing and data visualisation abstraction layer.
.. moduleauthor:: Albert Ferguson <albertferguson118@gmail.com>
"""

# core

# third party libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import axes, container, gridspec
import seaborn as sns

# python core
import math
import os
import sys
import pickle
from datetime import datetime

CMAP =  plt.get_cmap('Paired') # Global colour map variable

def autoLabel(rects: container.BarContainer, ax: axes.Axes, fontSize: int):
    """ Attach a text label above each bar in rects, displaying its height.
    
    **Args:**
        rects(container.BarContainer):   The rectangles to bind to.
        ax(axes.Axes):                        The axis to bind to.
        fontSize(int):               The size of the font used.

    """

    for rect in rects:
        height = rect.get_height()
        # XYtext is the position on "top" of bar (x, y)
        ax.annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5*np.sign(height)), textcoords="offset points", fontsize=fontSize, ha='center', va='center_baseline')
    
    return

def Graphing_PieChart(labels: list, values:list, ax: axes.Axes, category=None, LABELS=None, size=0.5, fontSize=9, rad=1):
    """Pie chart constructor with custom design.
    
    Pie chart constructor for given labels and sizes. This generates 'donut' pie charts with
    percentage value labelling and styling features.

    **Args:**
        labels(list):   elems(str):    A list of string labels for the wedges
        values(list):   elems(float):  A list of float values to create chart with
        ax(axes.Axes):  The axis object to bind to
        category(str):  The category being plotted, if None, no title is set
        LABELS(bool):   LABELS True sets default labels (top right), False or None sets lower center
        size(float):    A vlaue that controls the size of the wedges generated for the 'donut' pies
        fontSize(int):  The font size of labelling.
        rad(float):     The radius of the pie chart. The inner radius (wedge rad) is scaled from this

    """

    # Initially set labels as none, update with custom legend after
    wedges, texts, autotexts = ax.pie(
            [math.fabs(x) for x in values], 
            labels=None, autopct="%1.1lf%%", 
            shadow=False, radius=rad, pctdistance=(rad+rad*0.1),
            wedgeprops=dict(width=size, edgecolor='w')
        )
    
    # Creating the legend labels, use the label keys initially passed to us
    # Use a bbox to set legend below pie chart for improved visibility if legend enabled
    if LABELS:
        ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(1,1))
    else:
        ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(rad*0.2, -0.4))
    
    if category is not None:
        ax.set_title(category.capitalize().replace('_', ' '), weight="bold") # default title
    
    plt.setp(autotexts, size=fontSize, weight="bold")

    return

def Graphing_BarChart(labels: list, values: list, ax: axes.Axes, label="Default Bar Chart Label"):
    """Bar chart constructor for given labels and sizes.

    **Args:**
        labels(list):  The labels to be applied to the chart
        values(list):  The values to be charted
        ax(axes.Axes): The axis object to bind to
        label(str):    The optional header title for the bar chart

    """

    width             = 1
    fontSize          = 12
    scaleFactor_float = 1.6
    n_labels          = len(labels)
    x                 = np.arange(n_labels)
    scaled_x          = [scaleFactor_float*i for i in x] # calculate length of x-axis then scale to match pie charts above

    labels.reverse()
    values.reverse()

    rects = ax.bar(scaled_x, values, width, color=[CMAP(i) for i in range(0,n_labels)], label=label)
    ax.set_xticks(scaled_x)
    ax.set_xticklabels([label.capitalize().replace('_', ' ') for label in labels])
    autoLabel(rects, ax,fontSize)

    return

def Graphing_ScatterPlot(X: list, Y: list, ax: axes.Axes, area=10, ALPHA=0.9):
    """Scatter plot constructor for given data and custom design.
    Generates a scatter plot with auto-scaling values, based on area. 
    Also applies styling and axis limiting.

    **Args:**
        X(list):        Values to be plotted.
        Y(list):        Values to be plotted.
        area(int):      The optional scaling value to plot a third dimension onto the graph
        ALPHA(float):   The optional setting for scatter points alpha visibility value.
        ax(axes.Axes):  The axis to bind to.

    """

    try:
        ax.scatter(X, Y, c='black', cmap=CMAP, alpha=ALPHA)
        if len(Y) == 1:
            ax.set_ylim([Y[0]*0.9, Y[0]*1.1])
        else:
            ax.set_ylim([np.asarray(Y).min(), np.asarray(Y).max()])
        
        if len(X) != 1:
            ax.set_xlim([np.asarray(X).min(), np.asarray(X).max()])
            
    except (TypeError, Exception):
        ax.scatter(X, Y, c='black', cmap=CMAP, alpha=ALPHA)
    
    return

def Graphing_plotCorrKendall(data: pd.DataFrame, ax: axes.Axes, labels=True):
    """Plot the Kendall Correlation as a heatmap matrix. By default, this applies
    annotations to each cell in the corr-matrix.

    **Args:**
        data(pd.DataFrame):	A dataframe to calculate correlation coeff's on.
        ax(axes.Axes):      An existing axis to attach the graph to.
        label(bool):	 	Option to toggle cell labelling. Default is 'on' == True.

    """

    sns.heatmap(data.corr(method ='kendall') , annot=True, ax=ax, xticklabels=labels)

    return

def Graphing_KernelDenSityEstPlot(feature: pd.Series, ax: axes.Axes, colour: str, label:str, y_label="Fitted Probability Density"):
    """Fit and plot a univariate or bivariate kernel density estimate.
    Takes the df series to plot with type: pd.Series and the axis to attach to. See: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    
    **Args:**
        feature(pd.Series): The data series to plot.
        ax(axes.Axes):      The axis to attach the graph to.
        colour(str):        The colour to shade the kde plot with. e.g. 'r', 'b', 'g'.
        label(str):         The x-label for the graph.
        y_label(str):       The y-label for the graph. Default is set to `Fitted Probability Density`.
    
    .. note:: This will cut the data to range of data by default.

    """

    kde_dict = {"cut":0,
                "shade":True,
                "bw":0.2,
                "color":colour
            }

    sns.distplot(feature, kde_kws=kde_dict, ax=ax)
    ax.set_ylabel(y_label)
    ax.set_xlabel(label)

    return

def Graphing_HistPlot(x: np.ndarray, bins: int, label:str, ax: axes.Axes) -> list:
    """Generate a step'ed histogram plot for n bins.
    
    **Args:**
        x(np.ndarray):  The x-data to plot.
        bins(int):      The integer number of bins used for binning.
        label(str):     The label for the graph.
        ax(axes.Axes):  An existing axis to attach the graph to.
        
    **Returns:**
        returnVals_list(list): [n, bins, patches]
    
    .. note:: The return data list is not required for the graph to attach to the axis. It only returns as optional extra data.

    """

    if type(x) is not np.ndarray:
        raise TypeError

    if sum(x.shape) == 0:
        raise ValueError
    elif x.sum() == 0:
        raise ValueError
    
    if ax is None:
        raise ValueError

    try:
        returnVals_list = list(ax.hist(x, bins=bins, density=True, histtype="step", cumulative=True, label=label, align="mid"))
    except ValueError:
        pass
    return returnVals_list

def Graphing_CulmDataPlot(data: pd.DataFrame, bins: int, ax: axes.Axes, label: str):
    """Plot the values, culminatively, against to time.	
    
    **Args:**
        data(pd.DataFrame): The dataframe with the data to index by time.
        bins(int):          A value for the number of bins to apply.
        ax(axes.Axes):      A premade axis to attach the graph to.
        label(str):         A label for the graph title.

    """

    if type(data) is not pd.DataFrame:
        raise TypeError
    
    if sum(data.shape) == 0:
        raise ValueError
    
    if ax is None:
        raise ValueError
    
    Graphing_HistPlot(data.Value, bins, label, ax)

    # xticks logic sets a tick for every month, as data is in time order this places the hist in time order
    ax.legend(loc="right")
    ax.set_title(label)
    ax.set_xticks(
                ticks=np.arange(0, ax.get_xlim()[1], ax.get_xlim()[1]/bins),
                labels=data.dateRep,
                rotation=30
            )

    return

def Graphing_TimeDeltaPlot(data: pd.DataFrame, ax: axes.Axes, label: str, rotation=20):
    """Plot the change in cases and deaths per country as a timeline.
        
    **Args:**
        data(pd.dataframe): The data to index by time. This should include Value and Date as attributes.
        ax(axes.Axes):      A premade axis to attach the graph to.
        label(str):         The graph title.
        rotation(int):      X-axis label rotation. Default is 20.

    """
    
    dates_axis = data.Date
    yData_axis = data.Value.diff()

    markerline, stemline, baseline = ax.stem(dates_axis, yData_axis, linefmt="C3-", basefmt="k-", use_line_collection=True)

    plt.setp(markerline, mec="k", mfc="w", zorder=3) # recolour markers and remove line through them
    markerline.set_ydata(np.zeros(len(data.dateRep)))

    # Annotate lines
    vert = np.array(['top', 'bottom'])[(yData_axis > 0).astype(int)]
    for d, l, r, va in zip(dates_axis, list(yData_axis), yData_axis, vert):
        if not (l > 0 or l < 0):
            continue
        text = ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),
                textcoords="offset points", va=va, ha="right")
        text.set_rotation(-45)
    
    # Formatting
    ax.set_title(label)
    ax.set_xticks(roation=rotation)
    ax.get_yaxis().set_visible(False)
    
    # Make spines invisible.
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    return
