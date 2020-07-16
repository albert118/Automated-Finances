__version__='1.13.0'
__doc__ = """The Page object, designed to replace the dataframe worker's Account
class in page generation features. Provides access to all the plot management,
text boxes, overlays and 'canvasing' tools for generating reports."""

# core
from src.core import environConfig, stats, images, graphing
from src.accountdata import AccountData

# third party libs
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, KeepTogether, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
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
    # TODO: move these settings to local.env
    PAGESIZE = A4 # opt: PAGESIZE = (140 * mm, 216 * mm) # width, height
    BASEMARGIN = 0.5 * mm
    AUTHOR = "Albert Ferguson"
    ENCRYPT = None
    TITLE = "Personal Finance Report"
 
    def __init__(self, Account: AccountData, savedir_path: str):
        ######
        # Account Data
        ######
        self.account = Account
        self.savedir_path = savedir_path 

        ######
        # PLATYPUS reportlab content control
        ######

        self.sample_style_sheet = getSampleStyleSheet()
        self.author             = self.AUTHOR
        self.encrypt            = self.ENCRYPT
        self.title              = self.TITLE

        # note: string lit's are not rendered into the final pdf
        self.blurb = """A detailed personal finance report on investments, incomes,
            expenditures and incomes. Completely modular and upgradeable.
            See the GitHub https://github.com/albert118/Automated-Finances"""

        title_style = self.get_title_style()
        body_style  = self.get_body_style()
        
        self.flowables = [
            Paragraph(self.title, title_style),
            Paragraph(self.blurb, body_style),
        ]

        # add in the graphs
        figsize = (A4[0]/92,A4[1]/92) # janky magic number scaling.
        income_graphs = self.account.display_income_stats(figsize=figsize)
        savings_graphs = self.account.display_savings_stats(figsize=figsize)
        expenditure_graphs = self.account.display_expenditure_stats(figsize=figsize)

        self.flowables.append(income_graphs)
        try:
            self.make_table(self.account.incomes.drop(labels=["File Source"], axis=1), body_style, "Income Transactions")
        except KeyError:
            pass

        self.flowables.append(savings_graphs)

        try:
            self.make_table(self.account.savings.drop(labels=["File Source"], axis=1), body_style, "Savings Transactions")
        except KeyError:
            pass

        self.flowables.append(expenditure_graphs)
        
        try:
            self.make_table(self.account.expenditures.drop(labels=["File Source"], axis=1), body_style, "Expenditures Transactions")
        except KeyError:
            pass

        self.report_val = self.build_report()
        self.write_pdf()


    def __repr__(self):
        return("pdf report: {title} by {author}".format(title=self.title, author=self.author))

    def make_table(self, data: pd.DataFrame, style:str, title:str):
        def _strip(val: str, charLimitLen=42):
            return (val[:charLimitLen] + "...")

        data["Description"] = data["Description"].apply(_strip)
        t = Table(data.values.tolist(), spaceBefore=10)
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                       ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))
        header = Paragraph("<b><font size=18>{}:</font></b>".format(title), style)

        for each in range(len(data)):
            if each % 2 == 0:
                bg_color = colors.whitesmoke
            else:
                bg_color = colors.lightgrey

            t.setStyle(TableStyle([('BACKGROUND', (0, each), (-1, each), bg_color)]))        
        aW = 500
        aH = 720

        w, h = header.wrap(aW, aH)
        self.flowables.append(header)
        aH = aH - h
        w, h = t.wrap(aW, aH)
        
        self.flowables.append(t)
        return


    # class methods
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

            report_pdf.build(
                self.flowables,
                onFirstPage=self.add_page_num,
                onLaterPages=self.add_page_num,
            )

            report_val = report_buffer.getvalue()
        return report_val

    def write_pdf(self):
        filename_str  = self.title + ".pdf"
        filename_path = os.path.join(self.savedir_path, filename_str)
        
        with open(filename_path, 'wb') as file:
            file.write(self.report_val)
