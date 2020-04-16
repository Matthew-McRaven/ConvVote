
import pickle
import os.path
import argparse

from PIL import Image
import torch, torchvision
import numpy as np

import CNNScan.Samples.Oregon
import CNNScan.Raster
def main(parser):
	options = parser.parse_args()
	# Determine output directory
	output_base_path = os.path.abspath(options.outdir[0])
	# Determine number of images / marking conditions
	count = options.count
	dpi = options.dpi
	# Load the ballot definition
	ballot = CNNScan.Samples.Oregon.ballot
	# Create output directory.
	output_directory = output_base_path

	if not os.path.exists(output_directory+ "/ballot_template"):
		os.makedirs(output_directory+ "/ballot_template")

	transforms=torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x: np.average(x, axis=-1, weights=[1,1,1,0],returned=True)[0]),
					                           torchvision.transforms.ToTensor(),
											   torchvision.transforms.Lambda(lambda x: x.float()),
											   torchvision.transforms.Normalize((1,),(127.5,))
											   #torchvision.transforms.Lambda(lambda x: (1.0 - (x / 127.5)).float())
											   ])
	print(os.path.abspath(ballot.ballot_file))
	ballot = CNNScan.Raster.Raster.rasterize_ballot_image(ballot, output_directory+"/ballot_template", dpi)

	# Determine marking parameters
	mark_db = CNNScan.Mark.MarkDatabase()
	mark_db.insert_mark(CNNScan.Mark.BoxMark())
	# Create marked ballots
	marked_ballots = CNNScan.Reco.Load.GeneratingDataSet(ballot, mark_db, count, transforms, True )
	marked_ballots.save_to_directory(output_directory)
		
		
	# Open the directory for 
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--outdir",help="Directory in which to store files", required=True, nargs=1)
	parser.add_argument("--count", default=100, help="Directory in which to store files", type=int)
	parser.add_argument("--dpi", default=400, help="DPI at which to write the ballots", type=int)
	main(parser)