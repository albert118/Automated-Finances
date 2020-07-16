# core
from src.core import environConfig

# third party libs
from svglib.svglib import svg2rlg
from reportlab.platypus import Image as platImage
from reportlab.lib import utils
from reportlab.lib.units import cm
from PIL import Image
# python core
import os
import sys
from datetime import datetime
from io import BytesIO

def img_buffer_to_svg(figure) -> BytesIO:
	"""Create an image bytes buffer using BytesIO from a given plt figure.
	Create an RLG image based on a figure.
	
	**Args:**
		figure(plt.fig):	The figure to create a buffer from.
	
	**Returns:**
		image:	The output buffer.

	"""

	with BytesIO() as figure_buffer:
		figure.savefig(figure_buffer, format='svg')
		figure_buffer.seek(0)
		image = svg2rlg(figure_buffer)
	return image

def resize_img_aspect(img_buffer: BytesIO, width=1*cm, img_mode='RGB', size= (900, 900)) -> platImage:
	"""Resize the given image buffer to the given width (height is based on width).

	**Args:**
		img_buffer(BytesIO):	The in memory image buffer.
		width(float):			The desired width of the returned image in cm. Default is 1cm.
		img_mode(str):			The image colour mode. Default is RGB.
		size(tuple):			elems(int):	A tuple of the height and width of the image size. Default 'guess' is 900x900.

	**Example Height Calculation:**
		height = width * (image_height / image_width)

	**Returns:**
		return_img(reportlab.platypus.Image):	The output image in the new aspect ratio (based on given width size).

	"""

	img = Image.frombytes(img_mode, size, img_buffer.getvalue())
	iw, ih = img.size
	aspect = ih / float(iw)
	return_img =  platImage(img_buffer, width=width, height=(width * aspect))

	return return_img
