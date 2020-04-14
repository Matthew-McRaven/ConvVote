import typing
import os
import numpy
from CNNScan.Raster import Raster
import CNNScan.Samples.utils
import CNNScan.Samples.Oregon
import CNNScan.Reco.Load
from CNNScan.Ballot import BallotDefinitions, MarkedBallots
import CNNScan.Mark
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PIL import Image
import CNNScan.Reco

if __name__=="__main__":
	# test rasterizing a ballot


	# create the oregon ballotdefinition object
	rasterize = CNNScan.Raster.Raster.rasterize_ballot_template
	bd = CNNScan.Ballot.BallotDefinitions 

	# Manually create contests for each entry on the Multnomah county ballot.


	# Wrap contests in a ballot definition
	ballot = bd.Ballot(contests=CNNScan.Samples.Oregon.percents.contests, ballot_file="CNNScan/Samples/Oregon/or2018ballot.pdf")
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

	CNNScan.utils.show_ballot(render_data.dataset.ballot, render_data.dataset.at(0))

	# pass BallotDefinition and directory of contest .png's into rasterize_ballot_template()
	# rasterize ballot, pick out each contest and save to directory
	# return a ballot with absolute pixel contest locations  