# This is where our conv neural net model will go.

import time, os, sys, random, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import collections

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import Reco.Settings as Settings
import utils as utils
import Ballot.BallotDefinitions, Ballot.MarkedBallots
import math

class ImageRecognitionCore(nn.Module):
	def __init__(self, config, input_dimensions):
		print(input_dimensions)
		super(ImageRecognitionCore, self).__init__()
		self.input_dimensions = input_dimensions

		# Construct convolutional layers.
		H = input_dimensions[0]
		W = input_dimensions[1]
		conv_list = []
		non_linear = Settings.get_nonlinear(config['recog_conv_nlo'])
		self.in_channels = in_channels = config['target_channels']

		# Iterate over all pooling/convolutional layer configurations.
		# Construct all items as a (name, layer) tuple so that the layers may be loaded into
		# an ordered dictionary. Ordered dictionaries respect the order in which items were inserted,
		# and are the least painful way to construct a nn.Sequential object.
		for index, item in enumerate(config['recog_conv_layers']):
			# Next item is a convolutional layer, so construct one and re-compute H,W, channels.
			if isinstance(item, Settings.conv_def):
				conv_list.append((f'conv{index}', nn.Conv2d(in_channels, item.out_channels, item.kernel,
				 stride=item.stride, padding=item.padding, dilation=item.dilation)))
				H = utils.resize_convolution(H, item.kernel, item.dilation, item.stride, item.padding)
				W = utils.resize_convolution(W, item.kernel, item.dilation, item.stride, item.padding)
				in_channels = item.out_channels
			# Next item is a pooling layer, so construct one and re-compute H,W.
			elif isinstance(item, Settings.pool_def):
				if item.pool_type.lower() == 'avg':
					conv_list.append((f'avgpool{index}',nn.AvgPool2d(item.kernel, stride=item.stride, padding=item.padding)))
					H = utils.resize_convolution(H, item.kernel, 1, item.stride, item.padding)
					W = utils.resize_convolution(W, item.kernel, 1, item.stride, item.padding)
				elif item.pool_type.lower() == 'max':
					conv_list.append((f'maxpool{index}', nn.MaxPool2d(item.kernel, stride=item.stride, padding=item.padding, dilation=item.dilation)))
					H = utils.resize_convolution(H, item.kernel, item.dilation, item.stride, item.padding)
					W = utils.resize_convolution(W, item.kernel, item.dilation, item.stride, item.padding)
				else:
					raise NotImplementedError(f"{item.pool_type.lower()} is not an implemented form of pooling.")

			# Add a non-linear operator if specified by item. Non linear operators also pair with dropout
			# in all the examples I've seen
			if item.non_linear_after:
				conv_list.append((f"{config['recog_conv_nlo']}{index}", non_linear))
				conv_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
		
		# Group all the convolutional layers into a single callable object.
		self.conv_layers = nn.Sequential(collections.OrderedDict(conv_list))


		# Construct fully connected layers
		#print(H, W, in_channels, H*W*in_channels) #Print the size of the
		layer_list =[]
		# The input dimension of the fully connected layers in the product the parameters of the last convolutional/pooling layer
		last_size = H*W*in_channels
		non_linear = Settings.get_nonlinear(config['recog_full_nlo'])

		# Iterate over list of fully connected layer definitions.
		# Construct all items as a (name, layer) tuple so that the layers may be loaded into
		# an ordered dictionary. Ordered dictionaries respect the order in which items were inserted,
		# and are the least painful way to construct a nn.Sequential object.
		for index, layer_size in enumerate(config['recog_full_layers']):
			layer_list.append((f"fc{index}", nn.Linear(last_size, layer_size)))
			layer_list.append((f"{config['recog_full_nlo']}{index}", non_linear))
			layer_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
			last_size = layer_size

		self.output_layer_size = last_size
		self.linear = nn.Sequential(collections.OrderedDict(layer_list)) #nn.Linear(input_dimensions, self.output_layer_size)
		
		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

	def forward(self, contest_number, batches, images):

		H,W = self.input_dimensions
		# Magic line of code needed to cast image vector into correct dimensions?
		images = images.view(batches, self.in_channels, H, W)
		outputs = self.conv_layers(images)
		# Flatten output of convolutional layers to be used by fully connected layers.
		outputs = outputs.view(batches, -1)
		outputs = self.linear(outputs)
		return outputs

	# Return the number of dimensions in the output size
	def output_size(self):
		return self.output_layer_size


class ImageRescaler(nn.Module):
	def __init__(self, config, ballot):
		super(ImageRescaler, self).__init__()
		# Init self from parameters
		self.x_out_res, self.y_out_res = config['target_resolution']
		# Don't allow an output size that is not a power of 2.
		assert(utils.is_power2(self.x_out_res))
		assert(utils.is_power2(self.y_out_res))

		self.pad = nn.ModuleList()
		self.pool = nn.ModuleList()
		self.x_in_res, self.y_in_res = [], []
		self.resize(ballot)

	def forward(self, contest_number, batches, images):

		# TODO: Perform pooling, interpolation, and nothing depending on the size ratios of in to out resolution.
		out = self.pad[contest_number](images)
		out = self.pool[contest_number](out)
		return out

	def output_size(self):
		return self.x_out_res * self.y_out_res

	# Return raw dimensions of output images, needed when performing convolutional filtering
	# in image recognition core.
	def output_dimensions(self):
		return (self.x_out_res, self.y_out_res)

	def resize(self, ballot):
		for contest in ballot.contests:
			# Determine the height, width of each image by subtracting the bounding rectangles from each other.
			imagex = contest.bounding_rect[2] - contest.bounding_rect[0]
			imagey = contest.bounding_rect[3] - contest.bounding_rect[1]
			# Must check that image being passed in is a power of 2, else pooling will fail.
			x_in_res, pad_top, pad_bottom = utils.pad_nearest_pow2(imagex, self.x_out_res)
			y_in_res, pad_left, pad_right = utils.pad_nearest_pow2(imagey, self.y_out_res)
			
			#print(pad_left, pad_right, pad_bottom, pad_top)
			self.pad.append(nn.ZeroPad2d((pad_left, pad_right, pad_bottom, pad_top)))
			# Since we've picked input resolutions, output resolutions that are powers of 2,
			# ratios should be whole numbers (even without rounding).
			x_ratio, y_ratio = x_in_res//self.x_out_res, y_in_res//self.y_out_res
			self.pool.append(nn.AvgPool2d((x_ratio, y_ratio)))

			# Need input resolution to properly view(...) input tensor
			self.x_in_res.append(x_in_res)
			self.y_in_res.append(y_in_res)

		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

# A class that takes an arbitrary number of inputs and useses it to perform class recognition on the data.
# The output dimension should match the number of classes in the data.
class OutputLabeler(nn.Module):
	def __init__(self, config, input_dimensions, ballot):
		super(OutputLabeler, self).__init__()
		self.input_dimensions = input_dimensions
		self.output_dimension = []
		self.output_layers = nn.ModuleList()
		self.resize(ballot)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, contest_number, batches, inputs):
		inputs = inputs.view(batches, -1)
		output = self.output_layers[contest_number](inputs)
		# Map outputs into range [0,1], with 1 being a vote for an option
		# and 0 being the absence of a vote.
		output = self.sigmoid(output)
		return output

	def resize(self, ballot):
		for contest in ballot.contests:
			this_dim = len(contest.options)
			self.output_dimension.append(this_dim)
			self.output_layers.append(nn.Linear(self.input_dimensions, this_dim))

		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

# A container that holds the three portions of our Neural Network.
# The first segment rescales the images to be a standard size (which is input_image_x,input_image_y).
# The second segment uses a CNN to perform image recognition on the input image.
# The third segment selects the best fitting label for that data presented to it.
class BallotRecognizer(nn.Module):
	def __init__(self, config, ballot):
		super(BallotRecognizer, self).__init__()
		
		rescaler = ImageRescaler(config, ballot)
		recognizer = ImageRecognitionCore(config, rescaler.output_dimensions())
		labeler = OutputLabeler(config, recognizer.output_size(), ballot)

		# Use an ordered dict so printing out the model prints in the correct order.
		self.module_list = nn.ModuleDict(collections.OrderedDict([
			('rescaler', rescaler),
			('recognizer', recognizer),
			('labeler', labeler)])
			)

		
	def forward(self, contest_number, inputs):
		batches = len(inputs)
		outputs = self.module_list['rescaler'](contest_number, batches, inputs)
		outputs = self.module_list['recognizer'](contest_number, batches, outputs)
		outputs = self.module_list['labeler'](contest_number, batches, outputs)
		outputs = outputs.view(batches, -1)
		return outputs
	
	def resize_for_election(self, ballot: Ballot.BallotDefinitions.Ballot,):
		self.modules['rescaler'].resize(ballot)
		self.modules['labeler'].resize(ballot)

def train_single_contest(model, config, train_data, test_data, number_candidates):
	raise NotImplementedError()

def train_single_ballot(model, config, ballot, train_data, test_data):
	if config['cuda']:
		model = utils.cuda(model, config)
	
	# Create loss function(s) for task.
	criterion = Settings.get_criterion(config)

	# Choose an optimizer.
	optimizer = Settings.get_optimizer(config, model)

	# Train the network for a given number of epochs.
	for epoch in range(config['epochs']):
		epoch_train_images, epoch_train_loss = 0, 0
		batch_train_images, batch_train_loss = 0, 0
		
		model.train()
		for marked_ballots in train_data:
			for contest_idx in range(len(ballot.contests)):
				number_candidates = len(ballot.contests[contest_idx].options)
				# Must extract images, labels from "batch" variable. Move to CUDA device.
				images = utils.ballot_images_to_tensor(marked_ballots, contest_idx, config)
				labels = utils.ballot_labels_to_tensor(marked_ballots, contest_idx, config, number_candidates)
				# TODO: Ensure all votes in a batch have the same index.
				#print(images)
				optimizer.zero_grad()
				(output, images, loss, correct) = evaluate_one_batch(model, contest_idx, criterion, images, labels)
				batch_train_images += images
				batch_train_loss += loss
				#batch_test_correct += correct

				# Perform optimization
				loss.backward()
				optimizer.step()

				# Accumulate losses
				batch_train_images += images
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

		model.eval()
		(batch_test_images, batch_test_loss, batch_test_correct) = evaluate_ballots(model, ballot, test_data, config, criterion)

		print(f"Guessed {batch_test_correct} ballots out of {batch_test_images} total for {100*batch_test_correct/batch_test_images}% accuracy")
	return model

# Helper method to compute one development / testing data run.
def evaluate_one_batch(model, contest_number, criterion, images, labels):
	output = model(contest_number, images)

	loss = criterion(output, labels)

	batch_test_loss = loss
	batch_test_images, batch_test_correct = 0,0

	# Compute the number of options that were reported correctly
	if False:
		for (index, value) in enumerate(labels):
			#print(value, output[index]) #Print out the tensors being evaluated, useful when debugging output errors.
			for inner_index, inner_value in enumerate(value):
				batch_test_images+=1
				# If the difference between the output and labels is greater than half of the range (i.e. .5),
				# the network correctly chose the label for THIS option. No inference may be made about the whole contest.
				if inner_value - output[index][inner_index] < .5:
					batch_test_correct += 1

	# Compute the number of contests where every option was selected correctly
	else:
		batch_test_images = len(images)

		for (index, value) in enumerate(labels):
			#print(value, output[index]) #Print out the tensors being evaluated, useful when debugging output errors.
			correct_so_far = True
			for inner_index, inner_value in enumerate(value):
				#print(value, output[index])
				# Labels are eof {0,1}. 1 indicates a vote for an option, 0 is not.
				# The output of the network is a vector of floats in the range [0,1], created by a sigmoid.
				# If the output and label are close in value, the network correctly chose the label.
				# If the difference between the output and labels is greater than half of the range (i.e. .5),
				# then the network made a mistake on this contest, and the contest should be marked as incorrect. 
				if inner_value - output[index][inner_index] > .5:
					correct_so_far = False
					break

			if correct_so_far:
				batch_test_correct += 1
	
	return (output, batch_test_images, batch_test_loss, batch_test_correct)

# Evaluate a list of marked ballots against an already trained model with a particular configuration
def evaluate_ballots(model, ballot, marked_ballot_list, config, criterion=None, add_to_ballots=False):
	if criterion is None:
		criterion = Settings.get_criterion(config)
	(test_images, test_loss, test_correct) = (0,0,0)
	with torch.no_grad():
		for marked_ballots in marked_ballot_list:
			for contest_idx in range(len(ballot.contests)):
				number_candidates = len(ballot.contests[contest_idx].options)
				# Must extract images, labels from "batch" variable. Move to CUDA device.
				images = utils.ballot_images_to_tensor(marked_ballots, contest_idx, config)
				labels = utils.ballot_labels_to_tensor(marked_ballots, contest_idx, config, number_candidates)
				# TODO: Ensure all votes in a batch have the same index.
				(output, images, loss, correct) = evaluate_one_batch(model, contest_idx, criterion, images, labels)
				if add_to_ballots:
					#TODO: Add any items that were voted for to MarkedContest.computed_vote_index
					pass
				test_images += images
				test_loss += loss.data.item()
				test_correct += correct

				# Clean up memory, since CUDA seems to leak memory when running for a long time.
				del images
				del labels
				del loss
				del output
				torch.cuda.empty_cache()
	return (test_images, test_loss, test_correct)