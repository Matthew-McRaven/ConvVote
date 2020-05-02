import tempfile
import os
import pickle
import importlib

import torch
import torchvision
import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt 
import matplotlib

from CNNScan.Mark import gan
import CNNScan.Mark.Settings

def raster_images(images, my_dir):
	# Must use this line to prevent crashing.
	matplotlib.use('agg')
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

		fig.savefig(my_dir+f"/file{i}.png", dpi=400)


def do_once(config):
	transforms = torchvision.transforms.Compose([
												#torchvision.transforms.Grayscale(),
												torchvision.transforms.ToTensor(),
												torchvision.transforms.Normalize((1,),(127.5,))
												#torchvision.transforms.Lambda(lambda x: (x[0] + x[1] + x[2])/3)
												])
	data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark, transforms)
	loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

	disc_model = CNNScan.Mark.gan.MarkDiscriminator(config)
	gen_model = CNNScan.Mark.gan.MarkGenerator(config, config['gen_seed_len'])
	
	CNNScan.Mark.gan.train_once(config, gen_model, disc_model, loader, loader)

	count=4
	images = CNNScan.Mark.gan.generate_images(gen_model, count, config)

	my_dir = "."
	with open(my_dir+"/settings.p", "wb") as my_file:
		pickle.dump(config, my_file)
	raster_images(images, my_dir)

	del disc_model
	del gen_model
	del data
	del loader
	del images
	torch.cuda.empty_cache()

def main():
	config = CNNScan.Mark.Settings.generate_default_settings()
	if importlib.util.find_spec("ray") is None:
		raise NotImplementedError("Must install raytune.")
	else:
		from ray import tune
		import ray
		ray.init()
		config['epochs'] = tune.grid_search([100, 200, 400])
		config['learning_rate'] = tune.grid_search([0.0001, 0.00001, 0.000001])
		config['gen_seed_len'] = tune.grid_search([10, 50, 100, 200])
		layers = []
		layers.append([
		# Make sure the kernel size is SMALLER than the feature being recognized.
		CNNScan.Settings.conv_def(2, 4, 1, 0, 1, False),
		CNNScan.Settings.pool_def(4)
		])
		layers.append([
		# Make sure the kernel size is SMALLER than the feature being recognized.
		CNNScan.Settings.conv_def(2, 16, 1, 0, 1, False),
		CNNScan.Settings.pool_def(2)
		])
		layers.append([
		# Make sure the kernel size is SMALLER than the feature being recognized.
		CNNScan.Settings.conv_def(4, 16, 1, 0, 1, False),
		CNNScan.Settings.conv_def(4, 16, 1, 0, 1, True),
		CNNScan.Settings.pool_def(4)
		])
		layers.append([
		# Make sure the kernel size is SMALLER than the feature being recognized.
		CNNScan.Settings.conv_def(4, 32, 1, 0, 1, False),
		CNNScan.Settings.conv_def(4, 32, 1, 0, 1, True),
		CNNScan.Settings.pool_def(4),
		CNNScan.Settings.conv_def(7, 16, 1, 0, 1, False),
		CNNScan.Settings.conv_def(7, 16, 1, 0, 1, True),
		])
		config['disc_conv_layers'] = tune.grid_search(layers)
		config['disc_full_layers'] = tune.grid_search([[200], [400,200], [800,400,200], [800,800,800]])
		#config['cuda'] = False
		# TODO: Log accuracy results within neural network.
		analysis = tune.run(do_once, config=config, resources_per_trial={ "cpu": 4, "gpu": 0.0})
# Launch a training run, with optional hyperparameter sweeping.
if __name__ == "__main__":
	main()