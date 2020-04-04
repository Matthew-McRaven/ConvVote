# This is where our conv neural net model will go.

import time, os, sys, random, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import settings as settings
import utils as utils
import Election as Election
import math

class ImageRecognitionCore(nn.Module):
	def __init__(self, config, input_dimensions):
		super(ImageRecognitionCore, self).__init__()
		self.input_dimensions = input_dimensions
		self.output_layer_size = config['recog_out_dim']
		self.linear = nn.Linear(input_dimensions, self.output_layer_size)
		# Init self from parameters
		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

	def forward(self, images, batches):
		images = images.view(batches, -1)
		return self.linear(images)

	# Return the number of dimensions in the output size
	def output_size(self):
		return self.output_layer_size


class ImageRescaler(nn.Module):
	def __init__(self, config, imagex, imagey):
		super(ImageRescaler, self).__init__()
		# Init self from parameters
		self.x_out_res, self.y_out_res = config['target_resolution']
		# Don't allow an output size that is not a power of 2.
		assert(utils.is_power2(self.x_out_res))
		assert(utils.is_power2(self.y_out_res))

		self.resize(imagex, imagey)

	def forward(self, images, batches):
		# Check that the size of the input matches what we expect from a batch size.
		#nBatches = len(images)
		#assert(batches == nBatches)
		#xlen = len(images[0])
		#print(f"Xlen is {xlen}.")
		#assert(xlen == self.x_in_res)
		#ylen = len(images[0][0])
		#print(f"Ylen is {ylen}.")
		#assert(ylen == self.y_in_res)

		# TODO: Perform pooling, interpolation, and nothing depending on the size ratios of in to out resolution.
		#print(images)
		out = self.pool(images)
		return out
	def output_size(self):
		return self.x_out_res * self.y_out_res
	def resize(self, imagex, imagey):
		# Must check that image being passed in is a power of 2, else pooling will fail.
		assert(utils.is_power2(imagex))
		assert(utils.is_power2(imagey))

		self.x_in_res, self.y_in_res = imagex, imagey
		x_ratio, y_ratio = self.x_in_res//self.x_out_res, self.y_in_res//self.y_out_res
		#print(x_ratio, y_ratio)
		self.pool = nn.AvgPool2d((x_ratio, y_ratio))

		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

# A class that takes an arbitrary number of inputs and useses it to perform class recognition on the data.
# The output dimension should match the number of classes in the data.
class OutputLabeler(nn.Module):
	def __init__(self, config, input_dimensions, output_dimension):
		super(OutputLabeler, self).__init__()
		self.input_dimensions = input_dimensions
		self.output_dimension = -1
		self.resize(output_dimension)
	
	def forward(self, inputs, batches):
		inputs = inputs.view(batches, -1)
		output = self.output_layer(inputs)

		return output

	def resize(self, new_output_dimension):
		# Don't re-allocate output linear layer unless necessary
		if self.output_dimension == new_output_dimension:
			return

		# Create output layer
		self.output_dimension = new_output_dimension
		self.output_layer = nn.Linear(self.input_dimensions, self.output_dimension)

		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

# A container that holds the three portions of our Neural Network.
# The first segment rescales the images to be a standard size (which is input_image_x,input_image_y).
# The second segment uses a CNN to perform image recognition on the input image.
# The third segment selects the best fitting label for that data presented to it.
class BallotRecognizer(nn.Module):
	def __init__(self, config, input_image_x, input_image_y):
		super(BallotRecognizer, self).__init__()
		

		rescaler = ImageRescaler(config, input_image_x, input_image_y)
		recognizer = ImageRecognitionCore(config, rescaler.output_size())
		labeler = OutputLabeler(config, recognizer.output_size(), config['output_layers'])

		self.module_list = nn.ModuleDict({
			'rescaler':rescaler,
			'recognizer':recognizer,
			'labeler':labeler
		})

		
	def forward(self, inputs):
		batches = len(inputs)
		outputs = self.module_list['rescaler'](inputs, batches)
		outputs = self.module_list['recognizer'](outputs, batches)
		outputs = self.module_list['labeler'](outputs, batches)
		outputs = outputs.view(batches, -1)
		return outputs
	
	def resize_for_contest(self, contenst_info: Election.ContestDefinition,
	 contenst_phys_info: Election.ContestLocation ):
		self.modules['rescaler'].resize(contenst_phys_info.bound_rect[2], contenst_phys_info.bound_rect[3])
		self.modules['labeler'].resize(len(contenst_info.options))


def train_single_contest(model, config, train_data, test_data):
	if config['cuda']:
		model = utils.cuda(model, config)
	
	# Create loss function(s) for task
	criterion = settings.get_criterion(config)

	# Choose an optimizer
	optimizer = settings.get_optimizer(config, model)
	for epoch in range(config['epochs']):
		epoch_train_images, epoch_train_loss = 0, 0
		batch_train_images, batch_train_loss = 0, 0
		
		model.train()
		for batch_idx, batch in enumerate(train_data):
			# Must extract images, labels from "batch" variable. Move to CUDA device.
			images = utils.cuda(torch.tensor([x.image for x in batch], dtype=torch.float32), config)
			labels = utils.cuda(torch.tensor([x.actual_vote_index for x in batch], dtype=torch.long), config)
			#print(images)
			optimizer.zero_grad()
			output = model(images)

			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max value
			batch_cor = pred.eq(labels.view_as(pred)).sum().item() # count correct items

			# Perform optimization
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()

			# Accumulate losses
			batch_train_images += len(images)
			batch_train_loss += loss.data.item()

			if False:
				# Optionally perform logging within a batch.
				pass

			# Clean up memory, since CUDA seems to leak memory when running for a long time.
			del images
			del labels
			del loss
			del output
			torch.cuda.empty_cache()
		
		batch_test_images, batch_test_loss = 0,0
		batch_test_correct = 0

		model.eval()
		with torch.no_grad():
			for batch_idx, batch in enumerate(test_data):
				# Must extract images, labels from "batch" variable. Move to CUDA device.
				images = utils.cuda(torch.tensor([x.image for x in batch], dtype=torch.float32), config)
				labels = utils.cuda(torch.tensor([x.actual_vote_index for x in batch], dtype=torch.long), config)
				# Evaluate and compute statistics
				(output, images, loss, correct) = evaluate_one_batch(model, criterion, images, labels)
				batch_test_images += images
				batch_test_loss += loss
				batch_test_correct += correct

				# Clean up memory, since CUDA seems to leak memory when running for a long time.
				del images
				del labels
				del output
				torch.cuda.empty_cache()

		print(f"Guessed {batch_test_correct} ballots out of {batch_test_images} total for {100*batch_test_correct/batch_test_images}% accuracy")
	return model

# Helper method to compute one development / testing data run.
def evaluate_one_batch(model, criterion, images, labels):
	output = model(images)

	loss = criterion(output, labels)

	batch_test_images = len(images)
	batch_test_loss = loss.data.item()

	# Compute the number of correctly computed election outcomes
	batch_test_correct = 0
	pred = output.argmax(dim=1, keepdim=True)  # get the index of the max value
	batch_test_correct += pred.eq(labels.view_as(pred)).sum().item() # count correct items
	
	return (output, batch_test_images, batch_test_loss, batch_test_correct)