import time, os, sys, random, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import typing
import importlib

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import PIL
from PIL import Image

# Create a torch dataset from the marks included in CNNScan.Mark/marks
# Must pass in CNNScan.Mark (the module) as the first positional.
# This is because it is very difficult to "get" the module object for the module this code
# resides in.
def get_marks_dataset(package, transforms):
	# Must supply a custom loader function to pytorch dataset, otherwise it opens images in incorrect mode,
	# which makes all the pixels turn black.
	def loader(path):
		#print(path)
		return Image.open(path)

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
	# Input size is the number of items
	# TODO: Change input_size to (channel, H, W) tuple
	def __init__(self, config, input_size):
		super(MarkDiscriminator, self).__init__()
		self.input_size = input_size

		# Create a list of linear layers from the config dict.
		last_size = input_size
		layer_list =[]
		# TODO: Add parameter to config to read NN size. 
		for index, layer_size in enumerate([200, 150, 100]):
			layer_list.append((f"fc{index}", nn.Linear(last_size, layer_size)))
			# TODO: Select the NLO from config.
			layer_list.append((f"{config['recog_full_nlo']}{index}", nn.LeakyReLU()))
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
	def __init__(self, config, input_size, output_size):
		super(MarkGenerator, self).__init__()
		self.input_size = input_size
		# TODO: Read the embedding size from config.
		self.embed_size = 5
		# TODO: Make number of classes vary with the number of classes in the dataset
		self.embed = nn.Embedding(2, self.embed_size)
		# Input to NN is a concatenation of seeds and embedding table values
		last_size = input_size + self.embed_size
		layer_list =[]
		# TODO: Add parameter to config to read NN size. 
		for index, layer_size in enumerate([200, 150, 100]):
			layer_list.append((f"fc{index}", nn.Linear(last_size, layer_size)))
			# TODO: Select the NLO from config.
			layer_list.append((f"{config['recog_full_nlo']}{index}", nn.LeakyReLU()))
			layer_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
			last_size = layer_size

		self.linear = nn.Sequential(collections.OrderedDict(layer_list))
		self.output = nn.Linear(last_size, output_size)

	def forward(self, seed, real_fake):
		embeds = self.embed(real_fake)
		inputs = torch.cat([seed, embeds], dim=-1)
		outputs = self.linear(inputs)
		outputs = self.output(outputs)
		# TODO: Change output_size to (channel, H, W) tuple
		outputs = outputs.view(len(seed), 4, 128, 128)
		return outputs

def train_once(config, generator, discriminator, train_loader, test_loader):
	# TODO: Read criterion, optimizers from settings file.
	criterion = torch.nn.MSELoss(reduction='sum')
	optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=.001, weight_decay=.01)
	optimizer_gen = torch.optim.Adam(generator.parameters(), lr=.001, weight_decay=.01)
	iterate_loader_once(config, generator, discriminator, train_loader, criterion, 
	                    generated_count=5, optimizer_disc=optimizer_disc, optimizer_gen=optimizer_gen)

def iterate_loader_once(config, generator, discriminator, loader, criterion, generated_count=10, do_train=True, 
                        k=1, optimizer_disc=None, optimizer_gen=None):
	# TODO: Compute (true, false) x (positive, negative) rates for detecting marks, generated images.
	batch_count, batch_loss, marked_correct, real_correct = 0,0,0,0
	steps_trained = 0
	for epoch in range(config['epochs']):
		for (images, marked_labels) in loader:
			# Train the generator rather than the discriminator every k steps.
			engage_gan = (((steps_trained + 1) % k) == 0)


			gen_marked = torch.tensor(np.random.randint(0, 2, generated_count), dtype=torch.long)
			gen_images = generator(torch.tensor(np.random.random((generated_count,10)), dtype=torch.float), gen_marked)

			# Combine real, generated data.
			all_images = torch.cat((images, gen_images))
			all_marked_labels = torch.cat((marked_labels, gen_marked))
			# Label the images as either real (0) or generated (1), and cat the labels
			real_label_tensor = torch.full((len(images),),0, dtype=torch.float)
			fake_label_tensor = torch.full((generated_count,),1, dtype=torch.float)
			all_real_labels = torch.cat((real_label_tensor, fake_label_tensor))
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
				loss = -criterion(out_combined_labels, actual_combined_labels)
				optimizer = optimizer_gen
			else:
				loss = criterion(out_combined_labels, actual_combined_labels)
				optimizer = optimizer_disc

			if do_train:
				loss.backward()
				optimizer.step()
				
			steps_trained += 1