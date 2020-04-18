
import pickle
import os.path
import argparse
import copy

from PIL import Image
import torch, torchvision
import numpy as np

import CNNScan.Samples.Oregon
import CNNScan.Raster
def main(factory, **kwargs):
	# Determine number of images / marking conditions
	# Load the ballot definition
	# Create output directory.
	output_directory = os.path.abspath(kwargs['outdir'])

	transforms=CNNScan.Reco.Load.def_trans
	#print(os.path.abspath(ballot.ballot_file))

	for i,ballot in enumerate(factory.ballots):
		factory.ballots[i] = CNNScan.Raster.Raster.rasterize_ballot_image(ballot, kwargs['dpi'])

	# Determine marking parameters
	mark_db = CNNScan.Mark.MarkDatabase()
	mark_db.insert_mark(CNNScan.Mark.BoxMark())
	# Create marked ballots
	marked_ballots = CNNScan.Reco.Load.GeneratingDataSet(factory, mark_db, kwargs['count'], transforms, True )
	marked_ballots.save_to_directory(output_directory)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--outdir",help="Directory in which to store files", required=True, nargs=1)
	parser.add_argument("--count", default=100, help="Directory in which to store files", type=int)
	parser.add_argument("--dpi", default=400, help="DPI at which to write the ballots", type=int)
	#ballot = copy.deepcopy(CNNScan.Samples.Oregon.ballot)
	factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
	factory.Ballot(CNNScan.Samples.Oregon.contests, ballot_file=CNNScan.Samples.Oregon.ballot_file)
	main(factory, **vars(parser.parse_args()))