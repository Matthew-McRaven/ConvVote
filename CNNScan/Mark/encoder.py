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
import CNNScan.Mark.gan
class Encoder(nn.Module):
	def __init__(self, config, input_size, output_size):
		super(Encoder, self).__init__()
		self.config = config
		self.input_size = input_size
		self.output_size = config['gen_seed_image']
		# Create a convolutional network from the settings file definition
		conv_layers = config['enc_conv_layers']
		nlo_name = config['nlo']
		tup = CNNScan.Settings.create_conv_layers(conv_layers, self.input_size[1:], self.input_size[0], nlo_name, config['dropout'], True)
		conv_list, cnn_output_size, _, _, _ = tup

		# Group all the convolutional layers into a single callable object.
		self.conv_layers = nn.Sequential(collections.OrderedDict(conv_list))
		
		last_size = cnn_output_size
		layer_list =[]
		# Create FC layers based on configuration.
		# Append layer to layer list that is the correct "size" for the network output.
		for index, layer_size in enumerate(config['enc_full_layers']+[output_size]):
			layer_list.append((f"fc{index}", nn.Linear(last_size, layer_size)))
			layer_list.append((f"{config['nlo']}{index}", CNNScan.Settings.get_nonlinear(config['nlo'])))
			layer_list.append((f"dropout{index}", nn.Dropout(config['dropout'])))
			last_size = layer_size
		self.linear = nn.Sequential(collections.OrderedDict(layer_list))
		

	def forward(self, images):
		output = images.view(len(images), self.input_size[0], self.input_size[1], self.input_size[2])
		output = self.conv_layers(images)
		output = output.view(len(images), -1)
		output = self.linear(output)
		return output

class AutoEncoder(nn.Module):
	def __init__(self, config):
		super(AutoEncoder, self).__init__()
		self.config = config
		self.encoder = Encoder(config, config['im_size'], config['gen_seed_len'])
		self.decoder = CNNScan.Mark.gan.MarkGenerator(config, config['gen_seed_len'])

	def forward(self, images):
		output = self.encoder(images)
		output = self.decoder(output)
		return output

def iterate_loader_once(config, model, loader, criterion, do_train=False, optimizer=None):
		batch_loss, batch_count = 0, 0
		for images, _ in loader:
			images = utils.cuda(images, config)
			# Feed all data through auto encoder.
			out_images = model(images)

			loss = criterion(out_images, images)
				
			if do_train:
				loss.backward()
				optimizer.step()

			batch_count+=len(images)
			batch_loss += loss.data.item()

			# Free allocated memory to prevent crashes.
			del loss
			del out_images
			del images

			torch.cuda.empty_cache()

		return (batch_count, batch_loss)

def train_autoencoder(config, model, train_loader, test_loader):
	model = utils.cuda(model, config)	
	# Choose a criterion.
	criterion = CNNScan.Settings.get_criterion(config)

	# Choose an optimizer.
	optimizer = CNNScan.Settings.get_optimizer(config, model)

	for epoch in range(config['epochs']):
		# Perform training
		model.train()
		optimizer.zero_grad()
		count, loss = iterate_loader_once(config, model, train_loader, criterion, True, optimizer)
		print(f"Saw {count} images in epoch {epoch} with loss of {loss}.")
		# Perform evaluation.
		with torch.no_grad():
			pass
