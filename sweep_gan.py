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
	# Must use this line to prevent crashing.
	matplotlib.use('agg')
	CNNScan.Mark.raster_images(images, my_dir)

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
		config['epochs'] = tune.grid_search([0])
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