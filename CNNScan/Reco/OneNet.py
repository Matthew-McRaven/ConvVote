# This is where our conv neural net model will go.

import time, os, sys, random, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import typing

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import CNNScan.Reco.Settings as Settings
import CNNScan.utils as utils
from CNNScan.Ballot import BallotDefinitions, MarkedBallots
import CNNScan.utils
import math

class ImageRecognitionCore(nn.Module):
	def __init__(self, config, input_dimensions):
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
			self.output_layer_size = H * W * in_channels
			# Add a non-linear operator if specified by item. Non linear operators also pair with dropout
			# in all the examples I've seen
			if item.non_linear_after:
				conv_list.append((f"{config['recog_conv_nlo']}{index}", non_linear))
				conv_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
		
		# Group all the convolutional layers into a single callable object.
		self.conv_layers = nn.Sequential(collections.OrderedDict(conv_list))
		
		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

	def forward(self, _, contest_number, batches, images):

		H,W = self.input_dimensions
		# Magic line of code needed to cast image vector into correct dimensions?
		images = images.view(batches, self.in_channels, H, W)
		outputs = self.conv_layers(images)
		return outputs

	# Return the number of dimensions in the output size
	def output_size(self):
		return self.output_layer_size


class ImageRescaler(nn.Module):
	def __init__(self, config, ballot_factory):
		super(ImageRescaler, self).__init__()
		# Init self from parameters
		self.x_out_res, self.y_out_res = config['target_resolution']
		# Don't allow an output size that is not a power of 2.
		assert(utils.is_power2(self.x_out_res))
		assert(utils.is_power2(self.y_out_res))

		self.translate = []
		self.pad = nn.ModuleList()
		self.pool = nn.ModuleList()
		self.x_in_res, self.y_in_res = [], []
		self.resize(ballot_factory)

	def forward(self, ballot_number, contest_number, batches, images):
		out_tens =[]
		#print(ballot_number, contest_number, images)
		assert len(ballot_number) == len(contest_number)
		assert len(contest_number) == len(images)
		assert len(images) == batches
		for index in range(len(ballot_number)):
			ballot = ballot_number[index]
			contest = contest_number[index]
			lookup_index = self.translate[ballot][contest]
			# TODO: Perform pooling, interpolation, and nothing depending on the size ratios of in to out resolution.
			#print(images.shape)
			out = self.pad[lookup_index](images[index])
			#print(out.shape)
			out = self.pool[lookup_index](out)
			out_tens.append(out)
			#print(out.shape)
		return torch.cat(out_tens)

	def output_size(self):
		return self.x_out_res * self.y_out_res

	# Return raw dimensions of output images, needed when performing convolutional filtering
	# in image recognition core.
	def output_dimensions(self):
		return (self.x_out_res, self.y_out_res)

	def resize(self, ballot_factory):
		assert isinstance(ballot_factory, BallotDefinitions.BallotFactory)
		self.translate = len(ballot_factory.ballots) * [None]
		for outer, ballot in enumerate(ballot_factory.ballots):
			start = len(self.pad)
			self.translate[outer] = [i+start for i in range(len(ballot.contests))]
			for inner, contest in enumerate(ballot.contests):
				# Determine the height, width of each image by subtracting the bounding rectangles from each other.
				imagex = contest.abs_bounding_rect.lower_right.x - contest.abs_bounding_rect.upper_left.x
				imagey = contest.abs_bounding_rect.lower_right.y - contest.abs_bounding_rect.upper_left.y
				# Must check that image being passed in is a power of 2, else pooling will fail.
				x_in_res, pad_left, pad_right = utils.pad_nearest_pow2(imagex, self.x_out_res)
				y_in_res, pad_top, pad_bottom = utils.pad_nearest_pow2(imagey, self.y_out_res)
				#print(imagex, imagey)
				#print(pad_left, pad_right, pad_bottom, pad_top)
				self.pad.append(nn.ZeroPad2d((pad_left, pad_right, pad_bottom, pad_top)))
				# Since we've picked input resolutions, output resolutions that are powers of 2,
				# ratios should be whole numbers (even without rounding).
				x_ratio, y_ratio = x_in_res//self.x_out_res, y_in_res//self.y_out_res
				self.pool.append(nn.AvgPool2d((x_ratio, y_ratio)))

				# Need input resolution to properly view(...) input tensor
				self.x_in_res.append(x_in_res)
				self.y_in_res.append(y_in_res)
		# Print out the table that converts from 2d (ballot, contest) to the 1d module objects.
		print(self.translate)

		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)

# A class that takes an arbitrary number of inputs and useses it to perform class recognition on the data.
# The output dimension should match the number of classes in the data.
class OutputLabeler(nn.Module):
	def __init__(self, config, input_dimensions, ballot_factory):
		super(OutputLabeler, self).__init__()
		self.input_dimensions = input_dimensions
		self.tx_table = nn.Embedding(ballot_factory.num_contests(), config['recog_embed'])
		layer_list =[]
		# The input dimension of the fully connected layers in the product the parameters of the last convolutional/pooling layer
		last_size = self.input_dimensions+config['recog_embed']
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
		layer_list.append((f"Output", nn.Linear(last_size, ballot_factory.max_options())))
		self.output_layer_size = ballot_factory.max_options()

		self.linear = nn.Sequential(collections.OrderedDict(layer_list)) #nn.Linear(input_dimensions, self.output_layer_size)

		self.sigmoid = nn.Sigmoid()
		self.config = config
		start = 0
		self.index_of = len(ballot_factory.ballots) * [None]
		for outer, ballot in enumerate(ballot_factory.ballots):
			self.index_of[outer] = [i+start for i in range(len(ballot.contests))]
			start += len(self.index_of[outer])
		print(self.index_of)

		# Randomize initial parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.kaiming_normal_(p)
	
	def forward(self, ballot_number, contest_number, batches, inputs):
		#print(inputs)
		inputs = inputs.view(batches, -1)
		#print(ballot_number, contest_number)
		#print(torch.stack([ballot_number, contest_number]))
		wanted_embeds = []
		for index in range(len(ballot_number)):
			ballot = ballot_number[index]
			contest = contest_number[index]
			index = self.index_of[ballot][contest]
			wanted_embeds.append(index)
		tens = utils.cuda(torch.tensor(wanted_embeds, dtype=torch.long), self.config)
		#print(tens)
		emb = utils.cuda(self.tx_table(tens), self.config)
		#print(emb)
		#print(emb.shape, inputs.shape)
		inputs = torch.cat((emb, inputs), 1)
		#print(inputs.shape)
		output = self.linear(inputs)
		# Map outputs into range [0,1], with 1 being a vote for an option
		# and 0 being the absence of a vote.
		#print(output)
		#output = self.sigmoid(output)
		return output

# A container that holds the three portions of our Neural Network.
# The first segment rescales the images to be a standard size (which is input_image_x,input_image_y).
# The second segment uses a CNN to perform image recognition on the input image.
# The third segment selects the best fitting label for that data presented to it.
class BallotRecognizer(nn.Module):
	def __init__(self, config, ballot_factory):
		super(BallotRecognizer, self).__init__()
		assert isinstance(ballot_factory, BallotDefinitions.BallotFactory)
		rescaler = ImageRescaler(config, ballot_factory)
		recognizer = ImageRecognitionCore(config, rescaler.output_dimensions())
		labeler = OutputLabeler(config, recognizer.output_size(), ballot_factory)
		self.sigmoid = nn.Sigmoid()

		# Use an ordered dict so printing out the model prints in the correct order.
		self.module_list = nn.ModuleDict(collections.OrderedDict([
			('rescaler', rescaler),
			('recognizer', recognizer),
			('labeler', labeler)])
			)

		
	def forward(self, ballot_number, contest_number, inputs):
		batches = len(inputs)
		outputs = self.module_list['rescaler'](ballot_number, contest_number, batches, inputs)
		outputs = self.module_list['recognizer'](ballot_number, contest_number, batches, outputs)
		outputs = self.module_list['labeler'](ballot_number, contest_number, batches, outputs)
		outputs = outputs.view(batches, -1)
		return outputs
	

def train_single_contest(model, config, train_data, test_data, number_candidates):
	raise NotImplementedError()

def train_election(model, config, ballot_factory, train_loader, test_loader):
	if config['cuda']:
		model = utils.cuda(model, config)
	
	# Create loss function(s) for task.
	criterion = Settings.get_criterion(config)

	# Choose an optimizer.
	optimizer = Settings.get_optimizer(config, model)

	# Train the network for a given number of epochs.
	for epoch in range(config['epochs']):

		
		print(f"\n\nEpoch {epoch}")
		print("From 0 wrong to n wrong.")
		print("None selected, idx0 to idxn")
		model.train()
		optimizer.zero_grad()
		(batch_images, batch_loss, batch_correct, batch_select) = iterate_loader_once(config, model, ballot_factory, train_loader, criterion=criterion, optimizer=optimizer, train=True, annotate_ballots=False)
		for i, row in enumerate(batch_correct):
			if sum(row) == 0:
				continue
			print(f"Accuracy with {i+1} options is: {row[0:i+2]}")
			print(f"Selection distribution is: batch_select{batch_select[i][0:i+2]}\n")
		#print(f"Guessed {batch_correct} options out of {batch_images} total for {100*batch_correct[0]/batch_images}% accuracy. Loss of {batch_loss}.")
		
		model.eval()
		with torch.no_grad():
			(batch_images, batch_loss, batch_correct, batch_select) = iterate_loader_once(config, model, ballot_factory, test_loader, criterion=criterion, train=False)
			#print(f"Guessed {batch_correct} options out of {batch_images} total for {100*batch_correct[0]/batch_images}% accuracy. Loss of {batch_loss}.")

	return model


# Iterate over all the data in a loader one time.
# Account for varying number of ballots, as well annotating the data in the data loader with recorded votes.
# Works for both training and development. Will probably not work with real data, since no labels/loss will be available.
# TODO: Handle multiple "middle layer" CNN's.
def iterate_loader_once(config, model, ballot_factory, loader, criterion=None, optimizer=None, train=True, annotate_ballots=True, count_options=False):
	batch_images, batch_loss, batch_correct = 0,0,[None]*ballot_factory.max_options()
	batch_select = [None]*(ballot_factory.max_options())
	for i in range(ballot_factory.max_options()):
		batch_correct[i]=[0]*ballot_factory.max_options()
	for i in range(ballot_factory.max_options()):
		batch_select[i]=[0]*(ballot_factory.max_options()+1)
	#print(len(batch_correct))
	ballot_types = [i for i in range(len(ballot_factory))]
	random.shuffle(ballot_types)
	for ballot_type in ballot_types:
		loader.dataset.freeze_ballot_definiton_index(ballot_type)
		for dataset_index, ballot_numbers, labels, images in loader:
			ballot_numbers = ballot_numbers.type(torch.LongTensor)
			for contest_idx in range(len(ballot_factory.ballots[ballot_type].contests)):
			#for i in range(1):
				#contest_idx = 0
				tensor_images = utils.cuda(images[contest_idx], config)
				tensor_labels = labels[contest_idx]
				tensor_labels = utils.cuda(tensor_labels.type(torch.FloatTensor), config)

				#print(contest_idx)
				tensor_contest_idx = utils.cuda(torch.full( (len(dataset_index),), contest_idx, dtype=torch.long), config)
				output = model(ballot_numbers, tensor_contest_idx, tensor_images)

				output = output.narrow(-1, 0, tensor_labels.shape[-1])

				loss = criterion(output, tensor_labels)

				# Perform optimization
				if train:
					loss.backward()
					optimizer.step()
				output = model.sigmoid(output)

				# Possibly annotate ballots with the list of recorded votes.
				if True:
					for (index, output_labels) in enumerate(output):
						#print(value, output[index]) #Print out the tensors being evaluated, useful when debugging output errors.
						marked_one = False
						#print(output_labels)
						for inner_index, inner_value in enumerate(output_labels):
							# If the difference between the output and labels is greater than half of the range (i.e. .5),
							# the network correctly chose the label for THIS option. No inference may be made about the whole contest.
							if inner_value > .5:
								batch_select[len(output_labels)-1][inner_index+1]+=1
								marked_one = True
								#ballot.marked_contest[contest_idx].computed_vote_index.append(inner_index)
						if not marked_one:
							batch_select[len(output_labels)-1][0]+=1
									
				# Compute the number of options determined correctly
				for (index, contest_options) in enumerate(output):
					num_wrong=0
					num_total = 0
					#print(labels[contest_idx][index], output[index]) #Print out the tensors being evaluated, useful when debugging output errors.
					for inner_index, option_value in enumerate(contest_options):
						num_total+=1
						# If the difference between the output and labels is greater than half of the range (i.e. .5),
						# the network correctly chose the label for THIS option. No inference may be made about the whole contest.
						if abs(tensor_labels[index][inner_index] - option_value) > .5:
							num_wrong += 1
					# TODO: Every entry in batch_correct is identical.... that doesn't seem right....
					batch_images+=num_total
					batch_correct[num_total-1][num_wrong]+=1
					

				# Accumulate losses
				batch_loss += loss

				# Clean up memory, since CUDA seems to leak memory when running for a long time.
				del tensor_images
				del tensor_labels
				#del loss
				#del output
				torch.cuda.empty_cache()
	return (batch_images, batch_loss, batch_correct, batch_select)
