import typing
import os

import numpy
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PIL import Image
import matplotlib.pyplot as plt

from CNNScan.Raster import Raster
import CNNScan.Samples.utils
import CNNScan.Samples.Oregon
import CNNScan.Reco.Load
from CNNScan.Ballot import BallotDefinitions, MarkedBallots
import CNNScan.Mark
import CNNScan.Reco

if __name__=="__main__":
	# test rasterizing a ballot


	# create the oregon ballotdefinition object
	rasterize = CNNScan.Raster.Raster.rasterize_ballot_image
	bd = CNNScan.Ballot.BallotDefinitions 

	# Manually create contests for each entry on the Multnomah county ballot.


	# Wrap contests in a ballot definition
	ballot = bd.Ballot(contests=CNNScan.Samples.Oregon.contests, ballot_file="CNNScan/Samples/Oregon/or2018ballot.pdf")
	# print("ballot:",ballot,"@",ballot.ballot_file)
	# print("contests:",len(ballot.contests))
	# # for con in ballot.contests:
	# # 	print("con",con,"@",con.bounding_rect)

	# print("contest 24 ",ballot.contests[24].bounding_rect)

	# bf = convert_from_path(ballot.ballot_file, output_folder="../test",output_file="oregon")

	# # print("ballot size",bf.size())
	# temp = os.listdir("../test")
	# imgs = []
	# for img in temp:
	# 	if "oregon" in img:
	# 		imgs.append(img)
	# imgs.sort()
	# page1 = Image.open(f"../test/{imgs[0]}")
	# print(page1.size)

	# br = ballot.contests[24].bounding_rect
	# print("contest dimensions",br)
	# print("contest page",ballot.contests[24].bounding_rect.page)
	# w,h=page1.size
	# qaud = (w*br.upper_left.x,h*br.upper_left.y,w*br.lower_right.x,h*.49)
	# c24test=page1.crop(qaud)
	# print(c24test.size)
	# c24test.show()

	direct="../test"
	if not os.path.exists(direct):
		os.mkdir(direct)
	ballot = rasterize(ballot, direct , 400)
	#print(value)
	for contest in ballot.contests:
		pass#print(contest.bounding_rect)
	config = CNNScan.Reco.Settings.generate_default_settings()
	# Display a single sample ballot to visualize if training was succesful.
	render_data = CNNScan.Reco.Driver.get_test(config, ballot, CNNScan.Samples.Oregon)

	CNNScan.utils.show_ballot(render_data.dataset.ballot_definition(0), render_data.dataset.at(0))
	"""for i, contest in enumerate(render_data.dataset.at(0).marked_contest):
		print(contest.index)
		if contest.index in [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]:
			continue
		fig = plt.figure()
		ax = fig.add_subplot( 1, 1, 1)
		ax.set_title(f'Contest {contest.index}')
		ax.set_xlabel(f'Vote for {contest.actual_vote_index}. Recorded as {contest.computed_vote_index}')
		ax.imshow(contest.image, interpolation='nearest')
	plt.show()"""
	
	

	# pass BallotDefinition and directory of contest .png's into rasterize_ballot_template()
	# rasterize ballot, pick out each contest and save to directory
	# return a ballot with absolute pixel contest locations  