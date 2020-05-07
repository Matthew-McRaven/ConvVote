import argparse
import torch
import torchvision
import numpy as np
import pickle
from CNNScan.Reco import Settings
import CNNScan.Samples
	
if __name__ == "__main__":
	# Parse command line arguments passed to the program.
	parser = argparse.ArgumentParser()
	parser.add_argument("--outdir",help="Directory in which to store ballot files.", required=True, type=str)
	parser.add_argument("--ballot", help=".", required=True)
	parser.add_argument("--pdf", help=" ", required=True)
	parser.add_argument("--count", default=100, help="Number of ballots to create.", type=int)
	parser.add_argument("--dpi", default=400, help="DPI at which to write the ballots.", type=int)
	args = parser.parse_args()

	# Create a factory, which contains all of the ballot definitions for a particular election.
	factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()

	# Insert the Oregon Ballot.
	# if args.include_oregon:
	# 	factory.Ballot(CNNScan.Samples.Oregon.contests, ballot_file=CNNScan.Samples.Oregon.ballot_file)
	# # Insert the Montana Ballot.
	# if args.include_montana:
		# factory.Ballot(CNNScan.Samples.Montana.contests, ballot_file=CNNScan.Samples.Montana.ballot_file)
	picklefile = open(args.ballot, 'rb')
	myballot = pickle.load(picklefile)
	picklefile.close() 
	# load the pickle file for contests
	factory.Ballot(myballot,ballot_file=args.pdf)
	# Create rasterized ballot templates
	for i,ballot in enumerate(factory.ballots):
		factory.ballots[i] = CNNScan.Raster.Raster.rasterize_ballot_image(ballot, args.dpi)

	

	# Determine marking parameters
	mark_db = CNNScan.Mark.MarkDatabase()

	# Feel free to modify the following lines of code to add new / different kinds of marks.
	# If you want to create an entirely new class of marks see the file CNNScan.mark.Marks.py
	mark_db.insert_mark(CNNScan.Mark.BoxMark())
	mark_db.insert_mark(CNNScan.Mark.BoxMark(color='green'))
	# Pass in any valid PIL color to create a box with that  color.
	# 	See https://pillow.readthedocs.io/en/4.0.x/reference/ImageColor.html
	#mark_db.insert_mark(CNNScan.Mark.BoxMark(color='blue'))
	#mark_db.insert_mark(CNNScan.Mark.BoxMark(color='#ffaadd'))
	mark_db.insert_mark(CNNScan.Mark.XMark())
	mark_db.insert_mark(CNNScan.Mark.InvertMark())

	# Pytorch image transformations that convert image to Tensor.
	transforms=CNNScan.Reco.Load.def_trans
	# Create an iterable data set containing `count` marked copies of each ballot type from the ballot factory.
	marked_ballots = CNNScan.Reco.Load.GeneratingDataSet(factory, mark_db, args.count, transforms, True)
	# Iterate over all marked ballots and write them to the disk.
	marked_ballots.save_to_directory(args.outdir)
