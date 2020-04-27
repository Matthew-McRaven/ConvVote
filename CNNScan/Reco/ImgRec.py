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

import CNNScan.Settings, CNNScan.Reco.Settings
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
		non_linear = CNNScan.Settings.get_nonlinear(config['recog_conv_nlo'])
		self.in_channels = in_channels = config['target_channels']

		# Iterate over all pooling/convolutional layer configurations.
		# Construct all items as a (name, layer) tuple so that the layers may be loaded into
		# an ordered dictionary. Ordered dictionaries respect the order in which items were inserted,
		# and are the least painful way to construct a nn.Sequential object.
		for index, item in enumerate(config['recog_conv_layers']):
			# Next item is a convolutional layer, so construct one and re-compute H,W, channels.
			if isinstance(item, CNNScan.Settings.conv_def):
				conv_list.append((f'conv{index}', nn.Conv2d(in_channels, item.out_channels, item.kernel,
				 stride=item.stride, padding=item.padding, dilation=item.dilation)))
				H = utils.resize_convolution(H, item.kernel, item.dilation, item.stride, item.padding)
				W = utils.resize_convolution(W, item.kernel, item.dilation, item.stride, item.padding)
				in_channels = item.out_channels
			# Next item is a pooling layer, so construct one and re-compute H,W.
			elif isinstance(item, CNNScan.Settings.pool_def):
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
		non_linear = CNNScan.Settings.get_nonlinear(config['recog_full_nlo'])

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

	def forward(self, _, contest_number, batches, images):

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

		index = self.translate[ballot_number][contest_number]
		# TODO: Perform pooling, interpolation, and nothing depending on the size ratios of in to out resolution.
		#print(images.shape)
		out = self.pad[index](images)
		#print(out.shape)
		out = self.pool[index](out)
		#print(out.shape)
		return out

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
		self.output_dimension = []
		self.output_layers = nn.ModuleList()
		self.translate = []
		self.resize(ballot_factory)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, ballot_number, contest_number, batches, inputs):
		index = self.translate[ballot_number][contest_number]
		inputs = inputs.view(batches, -1)
		output = self.output_layers[index](inputs)
		# Map outputs into range [0,1], with 1 being a vote for an option
		# and 0 being the absence of a vote.
		output = self.sigmoid(output)
		return output

	def resize(self, ballot_factory):
		assert isinstance(ballot_factory, BallotDefinitions.BallotFactory)
		self.translate = len(ballot_factory.ballots) * [None]
		for outer, ballot in enumerate(ballot_factory.ballots):
			start = len(self.output_layers)
			self.translate[outer] = [i+start for i in range(len(ballot.contests))]
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
	def __init__(self, config, ballot_factory):
		super(BallotRecognizer, self).__init__()
		assert isinstance(ballot_factory, BallotDefinitions.BallotFactory)
		rescaler = ImageRescaler(config, ballot_factory)
		recognizer = nn.ModuleList([ImageRecognitionCore(config, rescaler.output_dimensions()) for i in range(config['recog_copies'])])
		labeler = OutputLabeler(config, recognizer[0].output_size(), ballot_factory)
		self.run_all = True
		self.recognizer_copies = config['recog_copies']
		

		# Use an ordered dict so printing out the model prints in the correct order.
		self.module_list = nn.ModuleDict(collections.OrderedDict([
			('rescaler', rescaler),
			('recognizer', recognizer),
			('labeler', labeler)])
			)
		self.recognizer_list = len(ballot_factory)*[None]
		for ballot in range(len(ballot_factory)):
			self.recognizer_list[ballot] = [random.randint(0, self.recognizer_copies - 1) for i in range(len(ballot_factory.ballots[ballot].contests))]
		print(self.recognizer_list)

		
	def forward(self, ballot_number, contest_number, inputs):
		batches = len(inputs)
		outputs = self.module_list['rescaler'](ballot_number, contest_number, batches, inputs)
		if not self.run_all:
			outputs = self.module_list['recognizer'][self.recognizer_list[ballot_number][contest_number]](ballot_number, contest_number, batches, outputs)
			outputs = self.module_list['labeler'](ballot_number, contest_number, batches, outputs)
			outputs = outputs.view(1, batches, -1)
		else:
			stack_output = []
			for way in self.module_list.recognizer:
				inter_output = way(ballot_number, contest_number, batches, outputs)
				inter_output = self.module_list['labeler'](ballot_number, contest_number, batches, inter_output)
				stack_output.append(inter_output)
			#print(f"Pre stacked{stack_output}")
			stack_output = torch.stack(stack_output)
			#print(f"Post-stacked {stack_output}")
			outputs = stack_output.view(self.recognizer_copies, batches, -1)
		return outputs

		def update_CNN_table(self, ballot_number, contest_number, which_CNN):
			self.recognizer_list[ballot_number][contest_number] = which_CNN
	

def train_single_contest(model, config, train_data, test_data, number_candidates):
	raise NotImplementedError()

def train_election(model, config, ballot_factory, train_loader, test_loader):
	if config['cuda']:
		model = utils.cuda(model, config)
	
	# Create loss function(s) for task.
	criterion = CNNScan.Settings.get_criterion(config)

	# Choose an optimizer.
	optimizer = CNNScan.Settings.get_optimizer(config, model)

	# Train the network for a given number of epochs.
	for epoch in range(config['epochs']):

		
		model.train()
		optimizer.zero_grad()
		(batch_images, batch_loss, batch_correct) = iterate_loader_once(config, model, ballot_factory, train_loader, criterion=criterion, optimizer=optimizer, train=True, annotate_ballots=False, count_options=True)
		print(f"Guessed {batch_correct} options out of {batch_images} total for {100*batch_correct/batch_images}% accuracy. Loss of {batch_loss}.")
		
		model.eval()
		with torch.no_grad():
			(batch_images, batch_loss, batch_correct) = iterate_loader_once(config, model, ballot_factory, test_loader, criterion=criterion, train=False, count_options=True)
			print(f"Guessed {batch_correct} options out of {batch_images} total for {100*batch_correct/batch_images}% accuracy. Loss of {batch_loss}.")
			(batch_images, batch_loss, batch_correct) = iterate_loader_once(config, model, ballot_factory, test_loader, criterion=criterion, train=False, count_options=False)
			print(f"Guessed {batch_correct} contests out of {batch_images} total for {100*batch_correct/batch_images}% accuracy. Loss of {batch_loss}.")
		#raise NotImplementedError("Can't test ballots yet")

		#print(f"Guessed {batch_test_correct} ballots out of {batch_test_images} total for {100*batch_test_correct/batch_test_images}% accuracy")
	return model


# Iterate over all the data in a loader one time.
# Account for varying number of ballots, as well annotating the data in the data loader with recorded votes.
# Works for both training and development. Will probably not work with real data, since no labels/loss will be available.
# TODO: Handle multiple "middle layer" CNN's.
def iterate_loader_once(config, model, ballot_factory, loader, criterion=None, optimizer=None, train=True, annotate_ballots=True, count_options=False):
	batch_images, batch_loss, batch_correct = 0,0,0
	ballot_types = [i for i in range(len(ballot_factory))]
	random.shuffle(ballot_types)
	for ballot_type in ballot_types:
		loader.dataset.freeze_ballot_definiton_index(ballot_type)
		for dataset_index, ballot_number, labels, images in loader:
			for contest_idx in range(len(ballot_factory.ballots[ballot_type].contests)):
				tensor_images = utils.cuda(images[contest_idx], config)
				tensor_labels = utils.cuda(labels[contest_idx], config)
				tensor_labels = tensor_labels.type(torch.FloatTensor)

				output = model(ballot_type, contest_idx, tensor_images)
				#print(output)
				#print(f"Dis is {ballot_type}{contest_idx}")
				#print(f"Ve hav ze labils {len(labels[contest_idx])} {len(output)}")
				losses = []
				loss = float("inf")
				best_output_index = 0
				#print(output)
				for outer, one_output in enumerate(output):
					#print("Batch element is", one_output)
					inner_loss = []
					for inner, row in enumerate(one_output):
						inner_loss.append(criterion(row, tensor_labels[inner]))
					
					numeric_loss = [loss.data.item() for loss in inner_loss]
					#print("Numeric is ", numeric_loss)
					# If all NN's returned the right result, then we should weight all losses equally (i.e. not at all)
					if max(numeric_loss) == 0 or (max(numeric_loss) == min(numeric_loss)):
						scaled_coef = [1 for co in numeric_loss]
						#print("1/n")
					# Otherwise weight the losses towards those "close" to the best.
					else:
						#print(numeric_loss, max(numeric_loss), min(numeric_loss))
						coef = [(max(numeric_loss) - i )/(max(numeric_loss)-min(numeric_loss)) for i in numeric_loss]
						#print("Coefs are ", coef)
						scaled_coef = [len(coef)*co/sum(coef) for co in coef]
						#print("Scaled coefs are ", scaled_coef)
					
					loss = sum([scaled_coef[i]*item for i,item in enumerate(inner_loss)])
					#print("Loss is ", loss)
					losses.append(loss)

				#print(losses)

				# Perform optimization
				if train:
					"""all_losses = []
					for i, lossy in enumerate(output):
						loss_i = criterion(lossy, tensor_labels)
						all_losses.append(loss_i)"""
					real_loss = sum(losses)
					real_loss.backward()
					optimizer.step()

				output = output[best_output_index]

				# Possibly annotate ballots with the list of recorded votes.
				if annotate_ballots:
					for (index, output_labels) in enumerate(output):
						#print(value, output[index]) #Print out the tensors being evaluated, useful when debugging output errors.
						for inner_index, inner_value in enumerate(output_labels):
							# If the difference between the output and labels is greater than half of the range (i.e. .5),
							# the network correctly chose the label for THIS option. No inference may be made about the whole contest.
							if inner_value > .5:
								# selected_ballots contains a list of all selected ballot indicies.
								# Must subscript with the "current" ballot.
								location = dataset_index[index]
								ballot = loader.dataset.at(location, ballot_number=ballot_type)
								if contest_idx >= len(ballot.marked_contest):
									#print(f"We have {len(ballot.marked_contest)} contests, but asked for {contest_idx}")
									ballot.marked_contest[contest_idx].computed_vote_index.append(inner_index)

				# Compute the number of options determined correctly
				if count_options:
					for (index, contest_options) in enumerate(output):
						#print(labels[contest_idx][index], output[index]) #Print out the tensors being evaluated, useful when debugging output errors.
						for inner_index, option_value in enumerate(contest_options):
							batch_images+=1
							# If the difference between the output and labels is greater than half of the range (i.e. .5),
							# the network correctly chose the label for THIS option. No inference may be made about the whole contest.
							if abs(labels[contest_idx][index][inner_index] - option_value) < .5:
								batch_correct += 1

				# Compute the number of contests where every option was selected correctly
				else:
					batch_images += len(tensor_images)
					
					for (index, contest_values) in enumerate(output):
						#print(contest_values)
						#print(value)
						#print(labels[contest_idx], output[index]) #Print out the tensors being evaluated, useful when debugging output errors.
						correct_so_far = True
						for inner_index, inner_value in enumerate(contest_values):
							# Labels are eof {0,1}. 1 indicates a vote for an option, 0 is not.
							# The output of the network is a vector of floats in the range [0,1], created by a sigmoid.
							# If the output and label are close in value, the network correctly chose the label.
							# If the difference between the output and labels is greater than half of the range (i.e. .5),
							# then the network made a mistake on this contest, and the contest should be marked as incorrect.
							#print(index, inner_index)
							#print(inner_value, output[index], "    ", end=" ")
							if abs(labels[contest_idx][index][inner_index] - inner_value) > .5:
								#print("wrong!", end="")
								correct_so_far = False
								break

						if correct_so_far:
							batch_correct += 1
							#batch_images += len(tensor_images)

				# Accumulate losses
				batch_loss += loss

				# Clean up memory, since CUDA seems to leak memory when running for a long time.
				del tensor_images
				del tensor_labels
				#del loss
				#del output
				torch.cuda.empty_cache()
	return (batch_images, batch_loss, batch_correct)