from reportlab.pdfgen import canvas

def hello(c):
from reportlab.lib.units import cm
# move the origin up and to the left
c.translate(cm, cm)
# define a larger font
c.setFont("Helvetica", 14)
# choose some colours
c.setStrokeColorRGB(0.2,0.5,0.3)
c.setFillColorRGB(1,0,1)
# draw some lines
c.line(0,0,0,1.7*cm)
c.line(0,0, 1*cm, 0)
# draw a rectangle
c.rect(0.2*cm, 0.2*cm, 1*cm, 1.5*cm, fill=1)
# make text go straight up
c.rotate(90
)# change colour
c.setFillColorRGB(0, 0, 0.77)
# say hello (not after rotate the y coord needs to be neg)
c.drawString(0.3*cm, -cm, "Hello, World!")

c = canvas.Canvas("hello.pdf")
hello(c)
c.showPage()
c.save()



# line operations
canvas.line(x1,y1, x2, y2)
canvas.lines(linelist)
# shape methods
canvas.grid(xlist, ylist)
canvas.bezier(x1, y1, x2, y2, x3, y3, x4, y4)
canvas.arc(x1, y1, x2, y2)
canvas.rect(x, y, width, height, stroke=1, fill=0)
canvas.elipse(x1, y1, x2, y2, stroke=1, fill=0)
canvas.wedge(x1, y1, x2, y2, startAng, extent, stroke=1, fill=0)
canvas.circle(x_cen, y_cen, r, stroke=1, fill=0)
canvas.roundRect(x, y, width, height, radius, stroke=1, fill=0)

# string methods
canvas.drawString(x, y, text)
canvas.drawRightString(x, y, text)
canvas.drawCentredSring(x, y, text)
# complex text with text objects, object methods allow formating
textObject = canvas.beginText(x, y)
canvas.drawText(textObject)

# path methods, complex shapes. Methods of path allow population with shapes
path = canvas.beginPath()
canvas.drawPath(path, stroke=1, fill=0, fillMode=None)
canvas.clipPath(path, stroke=1, fill=0, fillMode=None) # creates clipping region to remove content

# image methods
# depreciated, but excellent speed results for single use embedded images
canvas.drawInlineImage(self, image, x,y, width=None,height=None)
# PIL image objects or filenames are supported as arguments of image param incl. GIF

# implements a chaching system, once an image is added it is never re-added within
# the pdf, very efficient for large/repeated images.
# Smart checks PIL content for changes pre-embedding. Filename vals aren't smart checked
canvas.drawImage(self, image, x,y, width=None,height=None, mask=None)
# mask param allows alpha val to be set (transparent param)
# mask=[minR,maxR, minG,maxG, minB,maxB] - determine the background colour in 0-255 RGB range
# to mask it with this param

# state controls
canvas.saveState() # saves the current graphic state (fonts, colours, etc...)
canvas.restoreState() # restore to previous saved saveState

canvas.showPage() # finishes the current page object
# all states of current page are reset on the subsequent new page that will be
# defaulted to on calling this!


# Changing colours
canvas.setFillColorCMYK(c, m, y, k)
canvas.setStrikeColorCMYK(c, m, y, k)
canvas.setFillColorRGB(r, g, b)
canvas.setStrokeColorRGB(r, g, b)
canvas.setFillColor(acolor)
canvas.setStrokeColor(acolor)
canvas.setFillGray(gray)
canvas.setStrokeGray(gray) 
# changing fonts
canvas.setFont(psfontname, size, leading = None)
# changing graphical line styles
canvas.setLineWidth(width)
canvas.setLineCap(mode)
canvas.setLineJoin(mode)
canvas.setMiterLimit(limit)
canvas.setDash(self, array=[], phase=0) 
# changing geometry
canvas.setPageSize(pair)
canvas.transform(a,b,c,d,e,f):
canvas.translate(dx, dy)
canvas.scale(x, y)
canvas.rotate(theta) 
canvas.skew(alpha, beta) 


# meta options
canvas.setAuthor()
canvas.addOutlineEntry(title, key, level=0, closed=None)
canvas.setTitle(title)
canvas.setSubject(subj)
canvas.pageHasData()
canvas.showOutline()
canvas.bookmarkPage(name)
canvas.bookmarkHorizontalAbsolute(name, yhorizontal)
canvas.doForm()
canvas.beginForm(name, lowerx=0, lowery=0, upperx=None, uppery=None)
canvas.endForm()
canvas.linkAbsolute(contents, destinationname, Rect=None, addtopage=1, name=None, **kw)
canvas.linkRect(contents, destinationname, Rect=None, addtopage=1, relative=1, name=None, **kw)
canvas.getPageNumber()
canvas.addLiteral()
canvas.getAvailableFonts()
canvas.stringWidth(self, text, fontName, fontSize, encoding=None)
canvas.setPageCompression(onoff=1)
canvas.setPageTransition(self, effectname=None, duration=1,
direction=0,dimension='H',motion='I')

# PLATYPUS - Page Layout and Typography Using Scripts
# DocTemplates - top layer
# PageTemplates
# Frames - regions containing flowables
# Flowables - variable unstructured data content (images, text)
# pdfgen.Canvas, this low level is receiver of above PLATYPUS layers