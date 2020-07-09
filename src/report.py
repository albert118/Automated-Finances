__version__='1.13.0'
__doc__ = """The Page object, designed to replace the dataframe worker's Account
class in page generation features. Provides access to all the plot management,
text boxes, overlays and 'canvasing' tools for generating reports."""

# core
from core import environConfig, stats, images
from accountdata import AccountData

# third party libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, KeepTogether
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

    # PAGESIZE = (140 * mm, 216 * mm) # width, height
    PAGESIZE = A4
    BASEMARGIN = 0.5 * mm
    AUTHOR = "Albert Ferguson"
    ENCRYPT = None
    TITLE = "Personal Finance Report"
 
    def __init__(self, Account: AccountData):
        ######
        # Account data application control
        ######
        self.account = Account

        ######
        # PLATYPUS reportlab content control
        ######

        self.sample_style_sheet = getSampleStyleSheet()
        self.author = self.AUTHOR
        self.encrypt = self.ENCRYPT
        self.title = self.TITLE

        # note: string lit's are not rendered into the final pdf
        self.blurb = """A detailed personal finance report on investments, incomes,
            expenditures and incomes. Completely modular and upgradeable.
            See the GitHub https://github.com/albert118/Automated-Finances"""

        title_style = self.get_title_style()
        body_style = self.get_body_style()
        
        self.flowables = [
            Paragraph(self.title, title_style),
            Paragraph(self.blurb, body_style),
        ]
        self.add_graph()

        self.report_val = self.build_report()
        self.write_pdf()


    def __repr__(self):
        return("pdf report: {title} by {author}".format(title=self.title, author=self.author))

    # getters/setters/updaters

    # class methods
    def add_graph(self):
        figsize = (A4[0]/92,A4[1]/92) # janky magic number scaling.
        # TODO: figure out what conversion plt -> svg -> drawing -> canvas even does????
        income_graphs = self.account.display_income_stats(figsize=figsize)
        savings_graphs = self.account.display_savings_stats(figsize=figsize)
        expenditure_graphs = self.account.display_expenditure_stats(figsize=figsize)

        self.flowables.append(income_graphs)
        self.flowables.append(savings_graphs)
        self.flowables.append(expenditure_graphs)
        
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
        fn = self.title + ".pdf"
        open(fn, 'wb').write(self.report_val)