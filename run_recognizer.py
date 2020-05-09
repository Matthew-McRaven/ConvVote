
import argparse
import os
import tempfile

import torch

from CNNScan.Mark import gan
import CNNScan.Reco.Settings
import CNNScan.Samples

if __name__ == "__main__":
	# Parse command line arguments passed to the program.
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-dir", help="Directory where pre-generated ballots are store.", type=str)
	parser.add_argument("--include-oregon", help="Create marked up ballots for Oregon. Do not use with --data-dir.", action='store_true')
	parser.add_argument("--include-montana", help="Create marked up ballots for Montana. Do not use with --data-dir.", action='store_true')
	parser.add_argument("--include-random", default=0, help="Create a number of ballots that look like random noise. Do not use with --data-dir.", type=int)
	parser.add_argument("--ballot-count", default=100, help="The number of marked ballots to create from each class. Do not use with --data-dir.", type=int)
	parser.add_argument("--ballot-dpi", default=30, help="The DPI at which to render. Do not use with --data-dir.", type=int)
	# DO NOT enable this flag. It monotonically decreases the performance of the network
	parser.add_argument("--force-downsample", help="Enable image downsamoling in puts (not recommended!)",  action='store_true')
	parser.add_argument("--epochs", default=10, help="Number of times all images are to be shown to the neural net.",  type=int)
	parser.add_argument("--aggressive-crop", help="Crop the contest images so that only the option rectangles remain. Reduces portability, but speeds training time.", action='store_true')
	parser.add_argument("--batch-size", default=20, help="Number of items presented to the neural net at a time.", type=int)
	parser.add_argument("--learning-rate", default=0.0001, help="Learning rate of the neural net.", type=float)
	parser.add_argument("--dropout", default=0.1, help="Dropout rate in the neural net.", type=float)
	parser.add_argument("--l2-lambda", default=0.01, help="L2 regularization for the neural net.", type=float)
	parser.add_argument("--unique-outputs", help="Does each contest get its own fully connected layers?", action='store_true')
	parser.add_argument("--contest-embedding-size", default=10, help="Size of the embedding entry that encodes (ballot, contest) information.", type=int)
	args = parser.parse_args()
	
	config = CNNScan.Reco.Settings.generate_default_settings()


	# Store hyperparameters from the command liner arguments
	config['epochs'] = args.epochs
	config['batch_size'] = args.batch_size
	config['learning_rate'] = args.learning_rate
	config['dropout'] = args.dropout
	config['l2_lambda'] = args.l2_lambda
	config['rescale_pooling'] = args.force_downsample

	# Determine the configuration of the middle networks convolution layers.
	config['recog_conv_layers'] 	= [
		CNNScan.Settings.conv_def(4, 4, 1, 0, 1, False),
		CNNScan.Settings.conv_def(4, 4, 1, 0, 1, False),
		CNNScan.Settings.pool_def(1, 1, 0, 1, True, 'max'),
		]

	# Determine the configuration of the output layers
	config['unique_outputs'] = args.unique_outputs
	config['recog_embed'] = args.contest_embedding_size
	# Determine the number of neurons in each fully-connected layer of the output network
	config['recog_full_layers']	= [100]

	if not args.data_dir is None:
		if not os.path.isdir(args.data_dir):
			raise ValueError(f"Data directory {args.data_dir} does not exist.")
		data = CNNScan.Reco.Load.DirectoryDataSet(args.data_dir, CNNScan.Reco.Load.def_trans, False)
		factory = data.ballot_factory
	else:
		# Modify the marks applied to the ballot.
		markdb = CNNScan.Mark.MarkDatabase()
		markdb.insert_mark(CNNScan.Mark.BoxMark())
		markdb.insert_mark(CNNScan.Mark.InvertMark())
		markdb.insert_mark(CNNScan.Mark.XMark())


		factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
		# Insert the Oregon Ballot.
		if args.include_oregon:
			factory.Ballot(CNNScan.Samples.Oregon.contests, ballot_file=CNNScan.Samples.Oregon.ballot_file)
		# Insert the Montana Ballot.
		if args.include_montana:
			factory.Ballot(CNNScan.Samples.Montana.contests, ballot_file=CNNScan.Samples.Montana.ballot_file)
		# Insert the randomly generated ballots.
		[CNNScan.Samples.Random.get_sample_ballot(factory) for i in range(args.include_random)]
		# Must rasterize PDF's to PNG's. Use temporary directory to avoid polluting FS.
		with tempfile.TemporaryDirectory() as path:
			CNNScan.Mark.main(factory=factory, dpi=args.ballot_dpi, outdir=path, count=args.ballot_count, crop_contests=args.aggressive_crop)
		data = CNNScan.Reco.Load.GeneratingDataSet(factory, markdb, args.ballot_count)

	# Make all ballots of all types available to the len() builtin.
	data.freeze_ballot_definiton_index(None)
	# Datasets without ballots are boring. Abort if boring.
	if len(data) == 0:
		raise ValueError("Dataset must include some ballots. Please add ballots by selecting the --data-dir flag or one of the --include-* options.")

	# Create an iterable pytorch dataloader from either of our data sources.
	loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)
	model = CNNScan.Reco.ContestRec.BallotRecognizer(config, factory)
	# Print the model configuration to the harddisk.
	print(model)
	# Train the model for a given number of epochs.
	model = CNNScan.Reco.ContestRec.train_election(model, config, factory, loader, loader)
