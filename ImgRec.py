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
		print(input_dimensions)
		super(ImageRecognitionCore, self).__init__()
		self.input_dimensions = input_dimensions
		self.output_layer_size = config['recog_out_dim']
		self.linear = nn.Linear(input_dimensions, self.output_layer_size)
		# Init self from parameters
		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

	def forward(self, contest_number, batches, images):
		images = images.view(batches, -1)
		return self.linear(images)

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

	def resize(self, ballot):
		for contest in ballot.contests:
			imagex = contest.bounding_rect[2] - contest.bounding_rect[0]
			imagey = contest.bounding_rect[3] - contest.bounding_rect[1]
			# Must check that image being passed in is a power of 2, else pooling will fail.
			x_in_res, pad_top, pad_bottom = utils.pad_nearest_pow2(imagex, self.x_out_res)
			y_in_res, pad_left, pad_right = utils.pad_nearest_pow2(imagey, self.y_out_res)
			print(pad_left, pad_right, pad_bottom, pad_top)
			self.pad.append(nn.ZeroPad2d((pad_left, pad_right, pad_bottom, pad_top)))
			x_ratio, y_ratio = x_in_res//self.x_out_res, y_in_res//self.y_out_res
			#print(x_ratio, y_ratio)
			self.pool.append(nn.AvgPool2d((x_ratio, y_ratio)))
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
	
	def forward(self, contest_number, batches, inputs):
		inputs = inputs.view(batches, -1)
		output = self.output_layers[contest_number](inputs)

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
		recognizer = ImageRecognitionCore(config, rescaler.output_size())
		labeler = OutputLabeler(config, recognizer.output_size(), ballot)

		self.module_list = nn.ModuleDict({
			'rescaler':rescaler,
			'recognizer':recognizer,
			'labeler':labeler
		})

		
	def forward(self, contest_number, inputs):
		batches = len(inputs)
		outputs = self.module_list['rescaler'](contest_number, batches, inputs)
		outputs = self.module_list['recognizer'](contest_number, batches, outputs)
		outputs = self.module_list['labeler'](contest_number, batches, outputs)
		outputs = outputs.view(batches, -1)
		return outputs
	
	def resize_for_election(self, ballot: Election.Ballot,):
		self.modules['rescaler'].resize(ballot)
		self.modules['labeler'].resize(ballot)

def train_single_contest(model, config, train_data, test_data, number_candidates):
	return model

def train_single_ballot(model, config, ballot, train_data, test_data):
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
		for batch_idx, marked_ballots in enumerate(train_data):
			for contest_idx in range(len(ballot.contests)):
				number_candidates = len(ballot.contests[contest_idx].options)
				# Must extract images, labels from "batch" variable. Move to CUDA device.
				images = utils.cuda(torch.tensor([x.marked_contest[contest_idx].image for x in marked_ballots], dtype=torch.float32), config)
				labels = utils.cuda(torch.tensor([label_to_one_hot(x.marked_contest[contest_idx].actual_vote_index, number_candidates) for x in marked_ballots], dtype=torch.float32), config)
				# TODO: Ensure all votes in a batch have the same index.
				contest_number = contest_idx
				#print(images)
				optimizer.zero_grad()
				output = model(contest_number, images)

				#pred = output.argmax(dim=1, keepdim=True)  # get the index of the max value
				#batch_cor = pred.eq(labels.view_as(pred)).sum().item() # count correct items

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
			for batch_idx, marked_ballots in enumerate(test_data):
				for contest_idx in range(len(ballot.contests)):
					number_candidates = len(ballot.contests[contest_idx].options)
					# Must extract images, labels from "batch" variable. Move to CUDA device.
					images = utils.cuda(torch.tensor([x.marked_contest[contest_idx].image for x in marked_ballots], dtype=torch.float32), config)
					labels = utils.cuda(torch.tensor([label_to_one_hot(x.marked_contest[contest_idx].actual_vote_index, number_candidates) for x in marked_ballots], dtype=torch.float32), config)
					# TODO: Ensure all votes in a batch have the same index.
					contest_number = contest_idx
					(output, images, loss, correct) = evaluate_one_batch(model, contest_number, criterion, images, labels)
					batch_test_images += images
					batch_test_loss += loss
					batch_test_correct += correct

					# Clean up memory, since CUDA seems to leak memory when running for a long time.
					del images
					del labels
					del loss
					del output
					torch.cuda.empty_cache()

		print(f"Guessed {batch_test_correct} ballots out of {batch_test_images} total for {100*batch_test_correct/batch_test_images}% accuracy")
	return model
	
# Helper method to compute one development / testing data run.
def evaluate_one_batch(model, contest_number, criterion, images, labels):
	output = model(contest_number, images)

	loss = criterion(output, labels)

	batch_test_images = len(images)
	batch_test_loss = loss.data.item()

	# Compute the number of correctly computed election outcomes
	batch_test_correct = 0
	for (index, value) in enumerate(labels):
		correct_so_far = True
		for inner_index, inner_value in enumerate(value):
			#print(value, output[index])
			if inner_value - output[index][inner_index] > 1:
				correct_so_far = False
				break

		if correct_so_far:
			batch_test_correct += 1
		
	#pred = output.argmax(dim=1, keepdim=True)  # get the index of the max value
	#batch_test_correct += pred.eq(labels.view_as(pred)).sum().item() # count correct items
	
	return (output, batch_test_images, batch_test_loss, batch_test_correct)

def label_to_one_hot(label, length):
	ret = [0]*length
	if label != None:
		 ret[label] = 1
	return ret