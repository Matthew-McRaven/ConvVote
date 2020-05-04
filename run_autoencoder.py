import argparse

import torch

from CNNScan.Mark import gan
import CNNScan.Mark.Settings

	
if __name__ == "__main__":
	# Parse command line arguments passed to the program.
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=10, help="Number of times all images are to be shown to the neural net.",  type=int)
	parser.add_argument("--batch-size", default=10, help="Number of items presented to the neural net at a time.", type=int)
	parser.add_argument("--learning-rate", default=0.0001, help="Learning rate of the neural net.", type=float)
	parser.add_argument("--dropout", default=0.1, help="Dropout rate in the neural net.", type=float)
	parser.add_argument("--l2-lambda", default=0.01, help="L2 regularization for the neural net.", type=float)
	parser.add_argument("--outdir", help="Directory where output images from the auto-encoder should be stored.", required=True, type=str)
	parser.add_argument("--show-results", help="Open up the created images after execution.", action='store_true')
	args = parser.parse_args()
	
	config = CNNScan.Mark.Settings.generate_default_settings()


	# Store hyperparameters from the command liner arguments
	config['epochs'] = args.epochs
	config['batch_size'] = args.batch_size
	config['learning_rate'] = args.learning_rate
	config['dropout'] = args.dropout
	config['l2_lambda'] = args.l2_lambda

	# Create encoder/decoder neural networks.

	# Determine neural network configuration of the "encoder" portion of the auto-encoder
	# Construct the convolutional layers of the neural network.
	# See CNNScan.Settings for descriptions of the convolutional layers.
	config['enc_conv_layers']		= [
		CNNScan.Settings.conv_def(4, 64, 1, padding=3, non_linear_after=False, padding_type='reflect'),
		CNNScan.Settings.conv_def(4, 64, 2, padding=0, non_linear_after=True),
		CNNScan.Settings.conv_def(6, 64, 1, padding=5, non_linear_after=False, padding_type='reflect'),
		CNNScan.Settings.conv_def(6, 64, 2, padding=0, non_linear_after=True),
	]

	# Describe the number of neurons in each fully-connected layer.
	# A final layer of size `gen_seed_len` will be appended to this list automatically.
	config['enc_full_layers']		= [8192]

	# Size of the value used to "seed" the decoder section of the network.
	# This value is also the output size of 
	config['gen_seed_len']			= 4*32*32

	# Determine neural network configuration of the "decoder" portion of the auto-encoder
	# List of fully connected layers that convert the input seed into an image seed (see 'gen_seed_image')
	# A layer of size `gen_seed_image` will be appended to this list automatically
	config['gen_full_layers']		= []

	# How large an image is fed from the fully connected layers to the convolutional layers?
	config['gen_seed_image']		= (4, 32, 32)

	# Debugging "good" or working configurations for this portion of the auto-encoder may be difficult, since
	# working with transpose layers is hard. The output of this network must exactly be 4x128x128, where the 4 channels are RGBA.
	config['gen_conv_layers'] 	= [
		CNNScan.Settings.conv_transpose_def(4, 64, stride=2, non_linear_after=True, padding=1),
		CNNScan.Settings.conv_transpose_def(4, 4,  stride=2, non_linear_after=True, padding=1),
	]


	# Create dataset from marks on the hard disk and create a neural net from the configuration.
	data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark)
	loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)
	encoder = CNNScan.Mark.encoder.AutoEncoder(config)

	# Print a description of the neural net's layers to the console.
	print(encoder)

	# Train the auto-encoder based on the command line parameters.
	encoder = CNNScan.Mark.encoder.train_autoencoder(config, encoder, loader, loader)

	# Read `batch_size` images, feed them through the neural net, and save the original and new images to the disk.
	images,_ = next(iter(loader))
	CNNScan.Mark.raster_images(images, args.outdir, base_name="original", show_images=args.show_results)
	CNNScan.Mark.raster_images(encoder(images), args.outdir, base_name="decoded", show_images=args.show_results)
