import time, os, sys, random, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import typing
import importlib
import pathlib

from tabulate import tabulate
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import PIL
from PIL import Image

import CNNScan.Settings
import CNNScan.utils as utils

# Create a torch dataset from the marks included in CNNScan.Mark/marks
# Must pass in CNNScan.Mark (the module) as the first positional.
# This is because it is very difficult to "get" the module object for the module this code
# resides in.
def get_marks_dataset(package, transforms, subpath="only_marks"):
	# Must supply a custom loader function to pytorch dataset, otherwise it opens images in incorrect mode,
	# which makes all the pixels turn black.
	def loader(path):
		image = Image.open(path).convert('LA')
		#image.show()
		return image

	# Get the real file-system path to CNNScan/Mark/marks dataset
	real_path = importlib.resources.path(package, "marks")
	# The path must be opened with a context manage.
	with real_path as path:
		return torchvision.datasets.ImageFolder(path/subpath, transforms, loader=loader)

"""
This class takes in an image and make two predicitions per image.

The first prediction is whether or not the NN believe that the image contains a valid mark or not.
The output ∈ [0,1], where output > .5 indicates the image contains a mark, and output < .5 indicates it does not.

The second predicition is whether the image is "real" or the image came from the adversarial generator.
The output ∈ [0,1], where output > .5 indicates the image is generated, and output < .5 indicates it is real.
"""
class MarkDiscriminator(nn.Module):
	# Input size is a tuple of (channel count, height, width).
	def __init__(self, config):
		super(MarkDiscriminator, self).__init__()
		self.config = config
		self.input_size = config['im_size']
		print(self.input_size)
		# Input size is height * width * channel count
		last_size = self.input_size[1] * self.input_size[2] * self.input_size[0]
		layer_list =[]
		# Create FC layers based on configuration.
		for index, layer_size in enumerate(config['disc_full_layers']):
			layer_list.append((f"fc{index}", nn.Linear(last_size, layer_size)))
			layer_list.append((f"{config['nlo']}{index}", CNNScan.Settings.get_nonlinear(config['nlo'])))
			layer_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
			last_size = layer_size

		self.linear = nn.Sequential(collections.OrderedDict(layer_list))
		# Output is a list of 2 labels. The first indicates if the image contains a mark,
		# the second indicates if NN believes the image to be real or if it came from a generator.
		self.output = nn.Linear(last_size, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, batch_size, images):
		# Flatten the input images into a 2d array
		images = images.view(batch_size, -1)
		output = self.linear(images)
		output = self.output(output)
		# Clamp labels to [0,1].
		output = self.sigmoid(output)
		return output

class MarkGenerator(nn.Module):
	# Input_size is the number of random floats ∈ [0,1] given to the network per image.
	# TODO: Change output_size to (channel, H, W) tuple
	def __init__(self, config, input_size):
		super(MarkGenerator, self).__init__()
		self.config = config
		self.input_size = input_size
		self.output_size = config['im_size']

		last_size = input_size
		layer_list =[]
		# Create FC layers based on configuration.
		for index, layer_size in enumerate(config['gen_full_layers']):
			layer_list.append((f"fc{index}", nn.Linear(last_size, layer_size)))
			layer_list.append((f"{config['nlo']}{index}", CNNScan.Settings.get_nonlinear(config['nlo'])))
			layer_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
			last_size = layer_size

		self.linear = nn.Sequential(collections.OrderedDict(layer_list))

		conv_list = []
		# Should now be 32x32
		conv_list.append(('A1',nn.ConvTranspose2d(15, 128,4,stride=2,padding=1)))
		conv_list.append((f"B1", nn.LeakyReLU()))
		conv_list.append((f"dropout1", nn.Dropout(config['dropout'])))
		# Should now be 64x64
		conv_list.append(('A2',nn.ConvTranspose2d(128, 128,4,stride=2,padding=1)))
		conv_list.append((f"B2", nn.LeakyReLU()))
		conv_list.append((f"dropout2", nn.Dropout(config['dropout'])))
		# Should now be 128x128
		conv_list.append(('A3',nn.ConvTranspose2d(128, 128,4,stride=2,padding=1)))
		conv_list.append((f"B3", nn.LeakyReLU()))
		conv_list.append((f"dropout3", nn.Dropout(config['dropout'])))
		# Should now be 128x128
		conv_list.append(('A4',nn.Conv2d(128, 2,7,stride=1,padding=6,padding_mode='circular')))
		#conv_list.append((f"B4", nn.LeakyReLU()))
		#conv_list.append((f"dropout4", nn.Dropout(config['dropout'])))
		
		# Group all the convolutional layers into a single callable object.
		self.conv_layers = nn.Sequential(collections.OrderedDict(conv_list))

		# Use Tanh, see:
		#	https://github.com/soumith/ganhacks#1-normalize-the-inputs
		self.out_normalize = nn.Tanh()

	def forward(self, seed):
		outputs = self.linear(seed)
		print(outputs.shape)
		outputs = outputs.view(len(seed),-1,16,16)
		print(outputs.shape)
		outputs = self.conv_layers(outputs)
		print(outputs.shape)
		outputs = outputs.view(len(seed), self.output_size[0], self.output_size[1], self.output_size[2])
		outputs = self.out_normalize(outputs)
		return outputs

# Print out true vs false x positive vs negative table.
def square_print(square, title):
	print(title)
	# Actual value is in first index, recorded value in second.
	print(tabulate([["predict -",square[0][0],square[1][0]],["predict +",square[0][1],square[1][1]]],headers=['is-','is+']))

def train_once(config, generator, discriminator, train_loader, test_loader):
	# Create loss function(s) for task.
	criterion = CNNScan.Settings.get_criterion(config)

	# Choose an optimizer.
	optimizer_disc = CNNScan.Settings.get_optimizer(config, discriminator)
	optimizer_gen = CNNScan.Settings.get_optimizer(config, generator)
	steps_trained = 0
	for epoch in range(config['epochs']):
		discriminator.train()
		generator.train()
		optimizer_disc.zero_grad()
		optimizer_gen.zero_grad()
		(batch_count, disc_loss, gen_loss, steps_trained, real_square) = iterate_loader_once(
			config, generator, discriminator, train_loader, criterion, steps_trained=steps_trained,
			optimizer_disc=optimizer_disc, optimizer_gen=optimizer_gen)
		print(f"This is epoch {epoch}. Saw {batch_count} images.")
		square_print(real_square,"Real v. Generated")
		#square_print(marked_square,"Marked v. Unmarked")
		print(f"Loss is {disc_loss}, {gen_loss}")
		print("\n")


def generate_images(generator, count, config):
	return generator(torch.tensor(np.random.normal(size=(count, config['gen_seed_len'])), dtype=torch.float))

def iterate_loader_once(config, generator, discriminator, loader, criterion, do_train=True, 
                        k=1, optimizer_disc=None, optimizer_gen=None, steps_trained = 0):
	# TODO: Compute (true, false) x (positive, negative) rates for detecting marks, generated images.
	real_square = [[0,0],[0,0]]
	batch_count, disc_loss, gen_loss = 0,0,0

	# We must either train entirely on real images or entirely on generated ones
	# within a minibatch. Construct a list telling us at the i'th position which
	# dataset to use.
	batch_order = len(loader) * [0, 1]
	random.shuffle(batch_order)
	# We will need to iterate over dataloader manually
	loader_it = iter(loader)

	count = config['batch_size']

	for which in batch_order:
		# Train the generator rather than the discriminator every k steps.
		engage_gan = (((steps_trained + 1) % (k+1)) == 0)

		# This minibatch is real data, so iterate the data loader
		if which == 0:
			#try:
			images, _ = next(loader_it)
			# If we ran out of data, don't crash, and instead yield another minibatch of generated data
			#except StopIteration:
				#which = 1
		# Otherwise, this minibatch is is made of synthetic data.
		if which == 1:
			# Create an array of 0's and 1's which determine which pictures will contain marks
			# and which pictures will not.
			images = generate_images(generator, count, config)
		
		# Our real/fake label (which) needs to be as long as our image array.
		real_labels = torch.full((len(images),),which, dtype=torch.float)
		#toImage= torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None)])

		# Create random doubles between [0,.1]
		noise = 0.3*torch.tensor(np.random.random(size=(len(real_labels),)) ,dtype=torch.float) - 0.3
		# Add random noise to labels. Abs will "flip" the negative numbers about the origin.
		# See: https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels
		real_labels = abs(real_labels - noise)

		# Feed all data through the discriminator.
		out_labels = discriminator(len(images), images).view(-1)

		# If on the k'th step, train the generator rather than the discriminator.
		if engage_gan:
			# Must invert loss because reason? TODO
			# Maybe flip GAN labels when training GAN?
			# See: https://github.com/soumith/ganhacks#2-a-modified-loss-function
			loss = criterion(out_labels, 1-real_labels)
			gen_loss += loss.data.item()
			optimizer = optimizer_gen
		else:
			loss = criterion(out_labels, real_labels)
			disc_loss += loss.data.item()
			optimizer = optimizer_disc
			

		if do_train:
			loss.backward()
			optimizer.step()

			#discriminator.train(True)
			#generator.train(True)

		# Compute true +'ve -'ve, false +'ve -'ve rates for identification of
		# whether it is marked or unmarked as well as if it is real or generated.
		for i, output in enumerate(out_labels):
			predict_real = output.item()
			is_real = real_labels[i]

			# Actual value is in first index, recorded value in second.
			real_square[int(is_real.item())][round(predict_real)]+=1

		batch_count+=len(images)
		steps_trained += 1
	return (batch_count, disc_loss, gen_loss, steps_trained, real_square)