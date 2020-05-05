
from .Marks import *
from .__main__ import *
from . import gan
from . import encoder

from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt 
import cv2

def as_hist_RGB(img, plt):
	img = numpy.array(img.convert("RGBA")) 
	# Convert RGB to BGR 
	color = ('r', 'g', 'b', 'orange')
	bins = range(256)

	for i,col in enumerate(color):
		hist = cv2.calcHist([img],[i],None,[256],[0,256])
		plt.hist(hist, bins=bins, color = col, alpha=.25, density=True)

	plt.set_xlim([-10,255])
	return plt

def as_hist_L(img, plt):
	img = numpy.array(img.convert("L")) 
	histr = cv2.calcHist([img],[0],None,[256],[0,256])
	plt.hist(histr, bins=range(256), color = 'orange', alpha=1)
	plt.set_xlim([-10,255])
	return plt

def raster_images(images, path, base_name="file", show_images=False, dpi=400):
	# Convert a 2 channel tensor to an image.
	asLA= torchvision.transforms.Compose([
		torchvision.transforms.Normalize((-1/127.5,),(1/127.5,)),
		torchvision.transforms.ToPILImage()
	])
	# Convert a 1 channel tensor to an image.
	asL= torchvision.transforms.Compose([
		torchvision.transforms.Normalize((-1/127.5,),(1/127.5,)),
		torchvision.transforms.ToPILImage()
	])

	if not os.path.isdir(path):
		os.makedirs(path)

	my_filter = ImageFilter.FIND_EDGES#ImageFilter.Kernel((3,3), [0,1,0,1,-4,1,0,1,0], scale=4)
	# Perform channel decomposition and analysis on each of the generated images
	for i,image in enumerate(images):
		fig, ((og, l, a, la), (og_grad, l_grad, a_grad, la_grad), (og_hs, l_hs, a_hs, la_hs)) = plt.subplots(3, 4)
		fig.suptitle(f'Image {i} Channel Decomposition')

		im1 = asLA(image)
		RGBA = (1/3)*image[0] + (1/3)*image[1] + (1/3)*image[2]
		im2 = asL( torch.stack((RGBA,)) )
		im3 = asL( torch.stack((image[3],))  )
		im4 = asL( torch.stack((RGBA-image[3],)) )

		# Add axis titles and images.
		og.set_title("Original Image")
		og.imshow(im1)
		og_grad.set_title("Original Gradient")
		og_grad.imshow(im1.filter(my_filter))
		og_hs.set_title("Original Hist.")
		as_hist_RGB(im1, og_hs)

		l.set_title("L Channel")
		l.imshow(im2,cmap='Greys_r')
		l_grad.set_title("L Gradient")
		l_grad.imshow(im2.filter(my_filter),cmap='Greys')
		l_hs.set_title("L Hist.")
		as_hist_L(im2, l_hs)

		a.set_title("A Channel")
		a.imshow(im3,cmap='Greys')
		a_grad.set_title("A Gradient")
		a_grad.imshow(im3.filter(my_filter),cmap='Greys')
		a_hs.set_title("A Hist.")
		as_hist_L(im3, a_hs)

		la.set_title("L-A Channel")
		la.imshow(im4,cmap='Greys')
		la_grad.set_title("L-A Gradient")
		la_grad.imshow(im4.filter(my_filter),cmap='Greys')
		la_hs.set_title("L-A Hist")
		as_hist_L(im4, la_hs)

		fig.tight_layout()
		for ax in fig.get_axes():
			ax.label_outer()

		fig.savefig(path+f"/{base_name}{i}.png", dpi=dpi)

		if show_images:
			Image.open(path+f"/{base_name}{i}.png").show()