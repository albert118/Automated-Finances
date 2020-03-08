__version__='1.13.0'
__doc__ = """The Page object, designed to replace the dataframe worker's Account
class in page generation features. Provides access to all the plot management,
text boxes, overlays and 'canvasing' tools for generating reports."""

# core
from core import environConfig, stats

# third party libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm, inch
from svglib.svglib import svg2rlg

# python core
from datetime import datetime
import math
import os
import sys
from io import BytesIO

class Report():

    PAGESIZE = (140 * mm, 216 * mm) # width, height
    BASEMARGIN = 0.5 * mm
    AUTHOR = "Albert Ferguson"
    ENCRYPT = None
    TITLE = "Personal Finance Report"
 
    def __init__(self):
        ######
        # PLATYPUS reportlab stuff
        ######
        self.flowables = []
        # self.flowables.append(flowables)

        self.sample_style_sheet = getSampleStyleSheet()
        self.author = self.AUTHOR
        self.encrypt = self.ENCRYPT
        self.title = self.TITLE

        self.blurb = """\tA detailed personal finance report on investments, incomes,\n\t
            expenditures and incomes.\n\tCompletely modular and upgradeable.\n\t
            See the GitHub https://github.com/albert118/Automated-Finances\n"""

        self.report_val = self.build_report()
        self.write_pdf()

        #######
        # plt figure stuff, TODO: move to helper foo()
        #######
		# create a GridSpec to hold our page data
		# fig = plt.figure(figsize=(10,10))
		# outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,1])
		# col_count = math.ceil(len(self.expenditures.keys())/2)
		# inner_top = gridspec.GridSpecFromSubplotSpec(2, col_count, subplot_spec=outer[0],
		# 			wspace=0.2, hspace=0.2)
		# inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1],
		# 			wspace=0.05, hspace=0.1)

        # axN = fig.add_subplot(inner_top[0, key_counter]) # this is also one of the cleaner ways to create the axis
        # axN.set_prop_cycle(color=[CMAP(i) for i in range(1,10)])
        # pie_chart(label_vals.keys(), label_vals.values(), axN, category=key)
		# plt.suptitle("Expenditure Statistics")
		# ax_rect = fig.add_subplot(inner_bottom[0])
		# bar_chart(list(self.expenditures.keys()), totals, ax_rect)
		# ax_rect.set_ylabel('Expenditure')
		# ax_rect.set_xlabel('Category of Expenditure')
		# fig.add_subplot(ax_rect)

    def __repr__(self):
        return("pdf report: {title} by {author}".format(title=self.title, author=self.author))
    
    def __del__(self):
        del self.report_val
        del self.author
        del self.encrypt
        del self.title
        del self.blurb

    # getters/setters/updaters

    # class methods
    def add_graph(self):
        pass
    def add_text(self):
        pass
    def add_overlay(self):
        pass
    
    def add_page_num(self, canvas, doc):
        """Page number util function for report builder."""
        canvas.saveState()
        canvas.setFont('Times-Roman', 10)
        page_num_txt = "{}".format(doc.page)
        canvas.drawCentredString(
            0.75 * inch,
            0.75 * inch,
            page_num_txt,
        )
        canvas.restoreState()

    def get_body_style(self):
        style = self.sample_style_sheet
        body_style = ParagraphStyle(
            'BodyStyle',
            fontName="Times-Roman",
            fontSize=10,
            parent=style['Heading2'],
            alignment=0,
            spaceAfter=0,
        )
        return body_style

    def get_title_style(self):
        style = self.sample_style_sheet
        title_style = ParagraphStyle(
            'TitleStyle',
            fontName="Times-Roman",
            fontSize=18,
            parent=style['Heading1'],
            alignment=1,
            spaceAfter=0,
        )
        return title_style

    def build_report(self):
        # create a byte buffer for our pdf, allows returning for multiple cases
        with BytesIO() as report_buffer :
            report_pdf = SimpleDocTemplate(
                report_buffer, 
                pagesize=self.PAGESIZE,
                topMargin=self.BASEMARGIN,
                leftMargin=self.BASEMARGIN,
                rightMargin=self.BASEMARGIN,
                bottomMargin=self.BASEMARGIN,
                title=self.title,
                author=self.author,
                encrypt=self.encrypt,
                )

            title_style = self.get_title_style()
            body_style = self.get_body_style()
            flowables = [
                Paragraph(self.title, title_style),
                Paragraph(self.blurb, body_style),
            ]
            # append and update with any other flowable data from __init__
            # flowables.append(self.flowables)

            report_pdf.build(
                flowables,
                onFirstPage=self.add_page_num,
                onLaterPages=self.add_page_num,
            )

            report_val = report_buffer.getvalue()
        return report_val

    def write_pdf(self):
        fn = self.title + ".pdf"
        open(fn, 'wb').write(self.report_val)
