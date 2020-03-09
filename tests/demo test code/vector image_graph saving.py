# third party libs
import matplotlib.pyplot as plt
from svglib.svglib import svg2rlg
from reportlab.platypus import Image as platImage
from reportlab.pdfgen import canvas
from reportlab.lib import utils
from reportlab.lib.units import cm
from reportlab.graphics import renderPDF
from PIL import Image
# python core
import os
import sys
from datetime import datetime
from io import BytesIO

def img_buffer_to_svg(figure):
	with BytesIO() as figure_buffer:
		figure.savefig(figure_buffer, format='svg')
		figure_buffer.seek(0)	
		image = svg2rlg(figure_buffer)
	return image

def plotter():
    fig = plt.figure(figsize=(3,4))
    plt.plot([1,2,3,4,5,6,7,8,9,10])
    plt.ylabel("Some nums")
    plt.xlabel("Some more numbers")
    return fig

def make_canvas(drawing, x=0, y=0):
    c = canvas.Canvas("matplolib graph testing.pdf")
    renderPDF.draw(drawing, c, x, y)
    c.drawString(x+10, y+100, "Actually works??")
    c.showPage()
    c.save()
    return


fig = plotter()

print("test svg Drawing class method")
img_drawing = img_buffer_to_svg(fig)
make_canvas(img_drawing)
input()

print("test basic buffer method")
img_buffer = BytesIO()
fig.savefig(img_buffer, format='svg')
img_buffer.seek(0)
drawing = svg2rlg(img_buffer)
make_canvas(drawing)
