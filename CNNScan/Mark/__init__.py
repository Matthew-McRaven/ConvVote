
from .Marks import *
from .__main__ import *
from . import gan
from . import encoder

from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt 

def raster_images(images, path, show_images=False):
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
	my_filter = ImageFilter.FIND_EDGES#ImageFilter.Kernel((3,3), [0,1,0,1,-4,1,0,1,0], scale=4)
	# Perform channel decomposition and analysis on each of the generated images
	for i,image in enumerate(images):
		fig, ((og, l, a, la), (og_grad, l_grad, a_grad, la_grad)) = plt.subplots(2, 4)
		fig.suptitle(f'Image {i} Channel Decomposition')

		im1 = asLA(image)
		RGBA = image[0]#(1/3)*image[0] + (1/3)*image[1] + (1/3)*image[2]
		im2 = asL( torch.stack((RGBA,)) )
		im3 = asL( torch.stack((image[1],))  )
		im4 = asL( torch.stack((RGBA-image[1],)) )

		# Add axis titles and images.
		og.set_title("Original Image")
		og.imshow(im1)
		og_grad.set_title("Original Gradient")
		og_grad.imshow(im1.filter(my_filter))

		l.set_title("L Channel")
		l.imshow(im2,cmap='Greys_r')
		l_grad.set_title("L Gradient")
		l_grad.imshow(im2.filter(my_filter),cmap='Greys')

		a.set_title("A Channel")
		a.imshow(im3,cmap='Greys_r')
		a_grad.set_title("A Gradient")
		a_grad.imshow(im3.filter(my_filter),cmap='Greys')

		la.set_title("L-A Channel")
		la.imshow(im4,cmap='Greys')
		la_grad.set_title("L-A Gradient")
		la_grad.imshow(im4.filter(my_filter),cmap='Greys')

		for ax in fig.get_axes():
			ax.label_outer()

		fig.savefig(path+f"/file{i}.png", dpi=400)

		if show_images:
			Image.open(path+f"/file{i}.png").show()