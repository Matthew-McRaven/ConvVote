import time, os, sys, random, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import typing
import importlib

from tabulate import tabulate
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import PIL
from PIL import Image

import CNNScan.Settings

# Create a torch dataset from the marks included in CNNScan.Mark/marks
# Must pass in CNNScan.Mark (the module) as the first positional.
# This is because it is very difficult to "get" the module object for the module this code
# resides in.
def get_marks_dataset(package, transforms):
	# Must supply a custom loader function to pytorch dataset, otherwise it opens images in incorrect mode,
	# which makes all the pixels turn black.
	def loader(path):
		image = Image.open(path)
		#image.show()
		#print(path)
		return image

	# Get the real file-system path to CNNScan/Mark/marks dataset
	real_path = importlib.resources.path(package, "marks")
	# The path must be opened with a context manage.
	with real_path as path:
		return torchvision.datasets.ImageFolder(path, transforms, loader=loader)

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
		self.output = nn.Linear(last_size, 2)
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
		self.embed_size = config['gen_embed_size']
		# TODO: Make number of classes vary with the number of classes in the dataset
		self.embed = nn.Embedding(2, self.embed_size)
		# Input to NN is a concatenation of seeds and embedding table values
		last_size = input_size + self.embed_size
		layer_list =[]
		# Create FC layers based on configuration.
		for index, layer_size in enumerate(config['gen_full_layers']):
			layer_list.append((f"fc{index}", nn.Linear(last_size, layer_size)))
			layer_list.append((f"{config['nlo']}{index}", CNNScan.Settings.get_nonlinear(config['nlo'])))
			layer_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
			last_size = layer_size

		self.linear = nn.Sequential(collections.OrderedDict(layer_list))
		# Output size is height * width * channel count
		self.output = nn.Linear(last_size, self.output_size[1] * self.output_size[2] * self.output_size[0])
		# Use Tanh, see:
		#	https://github.com/soumith/ganhacks#1-normalize-the-inputs
		self.out_normalize = nn.Tanh()

	def forward(self, seed, real_fake):
		embeds = self.embed(real_fake)
		inputs = torch.cat([seed, embeds], dim=-1)
		outputs = self.linear(inputs)
		outputs = self.output(outputs)
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
		(batch_count, batch_loss, steps_trained, marked_square, real_square) = iterate_loader_once(
			config, generator, discriminator, train_loader, criterion, steps_trained=steps_trained,
			generated_count=config['generated_count'], optimizer_disc=optimizer_disc, optimizer_gen=optimizer_gen)
		print(f"This is epoch {epoch}. Saw {batch_count} images.")
		square_print(real_square,"Real v. Generated")
		square_print(marked_square,"Marked v. Unmarked")
		print(f"Loss is {batch_loss}")
		print("\n")


def generate_images(generator, count, config, gen_marked):
	return generator(torch.tensor(np.random.normal(size=(count, config['gen_seed_len'])), dtype=torch.float), gen_marked)
def iterate_loader_once(config, generator, discriminator, loader, criterion, generated_count=10, do_train=True, 
                        k=1, optimizer_disc=None, optimizer_gen=None, steps_trained = 0):
	# TODO: Compute (true, false) x (positive, negative) rates for detecting marks, generated images.
	marked_square = [[0,0],[0,0]]
	real_square = [[0,0],[0,0]]
	batch_count, batch_loss = 0,0
	for (images, marked_labels) in loader:
		# Train the generator rather than the discriminator every k steps.
		engage_gan = (((steps_trained + 1) % k) == 0)

		print(sum(sum(sum(images[0]))))
		toImage= torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None)])
		#toImage(images[0]).show()
		#print(images[0])

		gen_marked = torch.tensor(np.random.randint(0, 2, generated_count), dtype=torch.long)
		gen_images = generate_images(generator, generated_count, config, gen_marked)
		# Combine real, generated data.
		all_images = torch.cat((images, gen_images))
		all_marked_labels = torch.cat((marked_labels, gen_marked))
		# Label the images as either real (0) or generated (1), and cat the labels
		real_label_tensor = torch.full((len(images),),0, dtype=torch.float)
		fake_label_tensor = torch.full((generated_count,),1, dtype=torch.float)
		all_real_labels = torch.cat((real_label_tensor, fake_label_tensor))

		# Create random doubles between [0,.1]
		noise = 0.4*torch.tensor(np.random.random(size=(len(all_real_labels),)) ,dtype=torch.float) - 0.2
		# Add random noise to labels. Abs will "flip" the negative numbers about the origin.
		# See: https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels
		all_real_labels = abs(all_real_labels - noise)

		# The true number of items being fed into the network may be different from the batch_size hyperparameter if
		# the last batch doesn't have enough items.
		local_batch_size = len(images) + generated_count

		# Shuffle real, generated data.
		reorder = torch.randperm(local_batch_size)
		all_real_labels = all_real_labels[reorder]
		all_images = all_images[reorder]
		all_marked_labels = all_marked_labels[reorder]

		# Feed all data through the discriminator.
		out_combined_labels = discriminator(local_batch_size, all_images).view(-1)
		# Combine is_real, and is_marked lables into a single tensor
		actual_combined_labels = torch.cat((all_marked_labels.type(torch.FloatTensor), all_real_labels))

		# If on the k'th step, train the generator rather than the discriminator.
		if engage_gan:
			# Must invert loss because reason? TODO
			# Maybe flip GAN labels when training GAN?
			# See: https://github.com/soumith/ganhacks#2-a-modified-loss-function
			loss = criterion(out_combined_labels, 1-actual_combined_labels)
			optimizer = optimizer_gen
		else:
			loss = criterion(out_combined_labels, actual_combined_labels)
			optimizer = optimizer_disc

		if do_train:
			loss.backward()
			optimizer.step()
		# Compute true +'ve -'ve, false +'ve -'ve rates for identification of
		# whether it is marked or unmarked as well as if it is real or generated.
		for i, output in enumerate(out_combined_labels.view(-1, 2)):
			predict_mark, predict_real = output[0], output[1]
			is_mark, is_real = all_marked_labels[i], all_real_labels[i]
			#print(predict_real, is_real)
			# Actual value is in first index, recorded value in second.
			#print(is_mark, predict_mark)
			marked_square[int(is_mark.item())][round(predict_mark.item())]+=1
			real_square[int(is_real.item())][round(predict_real.item())]+=1
		batch_loss+=loss.data.item()
		batch_count+=len(all_marked_labels)
		steps_trained += 1
	return (batch_count, batch_loss, steps_trained, marked_square, real_square)