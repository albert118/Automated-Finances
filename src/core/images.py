# core
from core import environConfig

# third party libs
from svglib.svglib import svg2rlg

# python core
import os
import sys
from datetime import datetime
from io import BytesIO

def img_buffer(figure):
	with BytesIO as figure_buffer:
		figure.savefig(figure_buffer, format='svg')
		figure_buffer.seek(0)
	return figure_buffer