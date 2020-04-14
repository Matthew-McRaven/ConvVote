
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
	print(options.outdir)
	output_base_path = os.path.abspath(options.outdir[0])
	# Determine number of images / marking conditions

	ballot = CNNScan.Samples.Oregon.percents.ballot
	output_directory = output_base_path
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
	if not os.path.exists(output_directory+ "/ballot_template"):
		os.mkdir(output_directory+ "/ballot_template")
	transforms=torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x: np.average(x, axis=-1, weights=[1,1,1,0],returned=True)[0]),
					                           torchvision.transforms.ToTensor(),
											   torchvision.transforms.Lambda(lambda x: x.float()),
											   torchvision.transforms.Normalize((1,),(127.5,))
											   #torchvision.transforms.Lambda(lambda x: (1.0 - (x / 127.5)).float())
											   ])
	print(os.path.abspath(ballot.ballot_file))
	ballot = CNNScan.Raster.Raster.rasterize_ballot_template(ballot, output_directory+"/ballot_template", 400)

	# Create marked ballots
	marked_ballots = CNNScan.Mark.mark_dataset(ballot, count=1000, transform=transforms)
	# TODO: use generator expression to create ballots in batches, to reduce memory pressure on host machine
	with open(output_directory+"/ballot-template.p", "wb") as file:
		pickle.dump(ballot, file)
	for i in range(len(marked_ballots)):
		marked_ballot = marked_ballots.at(i)
		print(marked_ballot)
		# Serialize all contests to PNG's so they may be inspected.
		# Null out reference to image, so it does not get pickled.
		ballot_dir = output_directory+"/ballot%s/" %i
		if not os.path.exists(ballot_dir):
			os.mkdir(ballot_dir)
		for j, contest in enumerate(marked_ballot.marked_contest):
			print(ballot_dir+f"c{j:06}.png")
			contest.image.save(ballot_dir+f"c{j:06}.png")
			contest.clear_data()
			
		# Serialize ballot (without images!) to object in directory structure
		with open(ballot_dir+f"ballot{i}.p", "wb") as file:
			pickle.dump(marked_ballot, file)
		
		
	# Open the directory for 
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--outdir",help="Directory in which to store files", required=True, nargs=1)
	parser.add_argument("--count", default=100, help="Directory in which to store files", required=True)
	main(parser)