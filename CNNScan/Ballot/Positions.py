"""
This module contains data classes for representing picture coordinates in multiple coordinate systems.

Pixel points are absolute, but will become incorrect if an image is rescaled. These are necessary
when applying manipulations to images. For example, applying a mark to ballot requires having 
the pixel cooridnates of the box to fill.

Pixel coordinates are not suitable for describing an image in a PDF format, because a PDF may be rendered at any scale.
For these scale-invariant formats, there are relative positions. Relative positions use percentages to describe locations.
The width and height of a document are each 1.0, and a relative point specifies a location based on the % size of the document.
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence

@dataclass
class PixelPoint:
	x:int=0
	y:int=0
	def __init__(self, x:int=0, y:int=0):
		self.x = x 
		self.y = y
		# Future code expects that pixel positions  be integers, or image manipulations will fail.
		assert isinstance(x, int) and isinstance(y, int)

@dataclass
class RelativePoint:
	x:float = 0
	y:float = 0

@dataclass
class PixelPosition:
	upper_left:PixelPoint = PixelPoint()
	lower_right:PixelPoint = PixelPoint()
	def __init__(self, ul:PixelPoint = PixelPoint(), lr:PixelPoint = PixelPoint()):
		self.upper_left = ul
		self.lower_right = lr
		# Require that bounding rectangle be un-inverted. The upper left corner
		# must be strictly less than the bottom right corner.
		assert self.lower_right.y > self.upper_left.y
		assert self.lower_right.x > self.upper_left.x

	def size(self) -> (int, int):
		return (self.lower_right.x - self.upper_left.x, self.lower_right.y - self.upper_left.y)

@dataclass
class RelativePosition:
	upper_left:RelativePoint = RelativePoint()
	lower_right:RelativePoint = RelativePoint()
	page:int = 1
	
	def size(self) -> (int, int):
		return (self.lower_right.x - self.upper_left.x, self.lower_right.y - self.upper_left.y)
	
# Construct a Pixel-based bounding rectangle from four points.
def to_pixel_pos(x0, y0, x1, y1):
	return PixelPosition(PixelPoint(x0, y0), PixelPoint(x1, y1))

def to_percent_pos(x0, y0, x1, y1, page):
	return RelativePosition(RelativePoint(x0, y0), RelativePoint(x1, y1), page)

# TODO: Given the dimensions of a page, convert from a relative point (% based) to a pixel based point.
def convert_relative_absolute(size:Tuple[int, int], point:RelativePoint) -> PixelPoint:
	pass

