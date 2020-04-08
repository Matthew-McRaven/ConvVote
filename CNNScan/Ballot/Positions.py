from dataclasses import dataclass
from typing import Dict, Tuple, Sequence

@dataclass
class PixelPoint:
	x:int = 0
	y:int = 0

@dataclass
class RelativePoint:
	x:float = 0
	y:float = 0

@dataclass
class PixelPosition:
	upper_left:PixelPoint = PixelPoint()
	lower_right:PixelPoint = PixelPoint()

@dataclass
class RelativePosition:
	upper_left:RelativePoint = RelativePoint()
	lower_right:RelativePoint = RelativePoint()
	page:int = 1
	
# Construct a Pixel-based bounding rectangle from four points.
def to_pixel_pos(x0, y0, x1, y1):
	return PixelPosition(PixelPoint(x0, y0), PixelPoint(x1, y1))

# TODO: Given the dimensions of a page, convert from a relative point (% based) to a pixel based point.
def convert_relative_absolute(size:Tuple[int, int], point:RelativePoint) -> PixelPoint:
	pass

