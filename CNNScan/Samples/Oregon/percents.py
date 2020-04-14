"""
This module returns a ballot based on Mulnomah County's (Oregon) 2018 election.
The ballot is available from:
	https://multco.us/elections/sample-ballots-november-2018-general-election
"""
import os

import torchvision

import CNNScan.Ballot.BallotDefinitions  
import CNNScan.Samples.utils
import CNNScan.Samples.Oregon

import CNNScan.Reco.Load
bd = CNNScan.Ballot.BallotDefinitions 
to_pos = CNNScan.Ballot.Positions.to_percent_pos
contests = []

# Manually create contests for each entry on the Multnomah county ballot.

# Contest 00
c00_opt = []
c00_opt.append(bd.Option(0, "A", to_pos(.6797, .6200, .7010, .6300,1)))
c00_opt.append(bd.Option(1, "B", to_pos(.6797, .6369, .7010, .6468,1)))
c00 = bd.Contest(0, "c00", "", c00_opt, to_pos(0.6487, .4355, .9673, .6538,1), "c00.png")
contests.append(c00)


# Contest 01
c01_opt = []
c01_opt.append(bd.Option(0, "A", to_pos(.6797, .3681, .7059, .3780,1)))
c01_opt.append(bd.Option(1, "B", to_pos(.6797, .3879, .7059, .3978,1)))
c01 = bd.Contest(1, "c01", "", c01_opt, to_pos(0.6487, .245, .967, .3988,1), "c01.png")
contests.append(c01)

# Contest 02
c02_opt = []
c02_opt.append(bd.Option(0, "A", to_pos(.6797, .2083, .7059, .2183,1)))
c02_opt.append(bd.Option(1, "B", to_pos(.6797, .2282, .7059, .2381,1)))
c02 = bd.Contest(2, "c02", "", c02_opt, to_pos(0.6485, .0734, .9706, .2421,1), "c02.png")
contests.append(c02)

# Contest 03
c03_opt = []
c03_opt.append(bd.Option(0, "A", to_pos(.3562, .7262, .3775, .7361,1)))
c03_opt.append(bd.Option(1, "B", to_pos(.3562, .7460, .3775, .7560,1)))
c03 = bd.Contest(3, "c03", "", c03_opt, to_pos(.3235, .5129, .6438, .7589,1), "c03.png")
contests.append(c03)

# Contest 04
c04_opt = []
c04_opt.append(bd.Option(0, "A", to_pos(.3562, .4762, .3775, .4861,1)))
c04_opt.append(bd.Option(1, "B", to_pos(.3562, .4960, .3775, .5060,1)))
c04 = bd.Contest(4, "c04", "", c04_opt, to_pos(.3235, .3006, .6389, .5079,1), "c04.png")
contests.append(c04)

# Contest 05
c05_opt = []
c05_opt.append(bd.Option(0, "A", to_pos(.3562, .2629, .3775, .2728,1)))
c05_opt.append(bd.Option(1, "B", to_pos(.3562, .2808, .3775, .2907,1)))
c05 = bd.Contest(5, "c05", "", c05_opt, to_pos(.3235, 0.0734, .6389, .2937,1), "c05.png")
contests.append(c05)

# Contest 06
c06_opt = []
c06_opt.append(bd.Option(0, "A", to_pos(.0621, .8155, .0833, .8254,1)))
c06_opt.append(bd.Option(1, "B", to_pos(.0621, .8352, .0833, .8452,1)))
c06 = bd.Contest(6, "c06", "", c06_opt, to_pos(.0294, .6042, .3170, .8472,1), "c06.png")
contests.append(c06)

# Contest 07
c07_opt = []
c07_opt.append(bd.Option(0, "A", to_pos(.0621, .5665, .0833, .5764,1)))
c07_opt.append(bd.Option(1, "B", to_pos(.0621, .5843, .0833, .5942,1)))
c07 = bd.Contest(7, "c07", "", c07_opt, to_pos(.0294, .3323, .3170, .5972,1), "c07.png")
contests.append(c07)

# Contest 08
c08_opt = []
c08_opt.append(bd.Option(0, "A", to_pos(.0621, .2262, .0833, .2361,1)))
c08_opt.append(bd.Option(1, "B", to_pos(.0621, .2460, .0833, .2560,1)))
c08 = bd.Contest(8, "c08", "", c08_opt, to_pos(.0294, .1712, .3170, .2579,1), "c08.png")
contests.append(c08)

# Contest 09
c09_opt = []
c09_opt.append(bd.Option(0, "A", to_pos(.0621, .1379, .0833, .1478,1)))
c09_opt.append(bd.Option(1, "B", to_pos(.0621, .1558, .0833, .1657,1)))
c09 = bd.Contest(9, "c09", "", c09_opt, to_pos(.0294, .0842, .3170, .1706,1), "c09.png")
contests.append(c09)

# Contest 10
c10_opt = []
c10_opt.append(bd.Option(0, "A", to_pos(.6797, .7798, .7010, .7897,0)))
c10_opt.append(bd.Option(1, "B", to_pos(.6797, .7996, .7010, .8095,0)))
c10 = bd.Contest(10, "c10", "", c10_opt, to_pos(.6438, .7312, .9657, .8125,0), "c10.png")
contests.append(c10)

# Contest 11
c11_opt = []
c11_opt.append(bd.Option(0, "A", to_pos(.6797, .6369, .7010, .6468,0)))
c11_opt.append(bd.Option(1, "B", to_pos(.6797, .6567, .7010, .6667,0)))
c11_opt.append(bd.Option(2, "C", to_pos(.6797, .6736, .7010, .6835,0)))
c11 = bd.Contest(11, "c11", "", c11_opt, to_pos(.6438, .5972, .9657, .6885,0), "c11.png")
contests.append(c11)

# Contest 12
c12_opt = []
c12_opt.append(bd.Option(0, "A", to_pos(.6797, .5129, .7010, .5228,0)))
c12_opt.append(bd.Option(1, "B", to_pos(.6797, .5308, .7010, .5407,0)))
c12_opt.append(bd.Option(2, "C", to_pos(.6797, .5486, .7010, .5585,0)))
c12 = bd.Contest(12, "c12", "", c12_opt, to_pos(.6438, .4702, .9657, .5615,0), "c12.png")
contests.append(c12)

# Contest 13
c13_opt = []
c13_opt.append(bd.Option(0, "A", to_pos(.6797, .4048, .7010, .4147,0)))
c13_opt.append(bd.Option(1, "B", to_pos(.6797, .4236, .7010, .4335,0)))
c13 = bd.Contest(13, "c13", "", c13_opt, to_pos(.6438, .3552, .9657, .4485,0), "c13.png")
contests.append(c13)

# Contest 14
c14_opt = []
c14_opt.append(bd.Option(0, "A", to_pos(.6797, .3155, .7010, .3254,0)))
c14_opt.append(bd.Option(1, "B", to_pos(.6797, .3353, .7010, .3452,0)))
c14 = bd.Contest(14, "c14", "", c14_opt, to_pos(.6438, .2629, .9657, .3462,0), "c14.png")
contests.append(c14)

# Contest 15
c15_opt = []
c15_opt.append(bd.Option(0, "A", to_pos(.3563, .7986, .3775, .8085,0)))
c15_opt.append(bd.Option(1, "B", to_pos(.3563, .8165, .3775, .8264,0)))
c15 = bd.Contest(15, "c15", "", c15_opt, to_pos(.3252, .7490, .6356,.8323, 0), "c15.png")
contests.append(c15)

# Contest 16
c16_opt = []
c16_opt.append(bd.Option(0, "A", to_pos(.3563, .6915, .3775, .7014,0)))
c16_opt.append(bd.Option(1, "B", to_pos(.3563, .7093, .3775, .7192,0)))
c16_opt.append(bd.Option(2, "C", to_pos(.3563, .7272, .3775, .7371,0)))
c16 = bd.Contest(16, "c16", "", c16_opt, to_pos(.3252, .6399, .6356, .7421 ,0), "c16.png")
contests.append(c16)

# Contest 17
c17_opt = []
c17_opt.append(bd.Option(0, "A", to_pos(.3563, .6012, .3775, .6111,0)))
c17_opt.append(bd.Option(1, "B", to_pos(.3563, .6210, .3775, .6310,0)))
c17 = bd.Contest(17, "c17", "", c17_opt, to_pos(.3252, .5675, .6356, .6399, 0), "c17.png")
contests.append(c17)

# Contest 18
c18_opt = []
c18_opt.append(bd.Option(0, "A", to_pos(.3563, .5317, .3775, .5417,0)))
c18_opt.append(bd.Option(1, "B", to_pos(.3563, .5486, .3775, .5585,0)))
c18 = bd.Contest(18, "c18", "", c18_opt, to_pos(.3252, .4970, .6356, .5635, 0), "c18.png")
contests.append(c18)

# Contest 19
c19_opt = []
c19_opt.append(bd.Option(0, "A", to_pos(.3563, .4593, .3775, .4692,0)))
c19_opt.append(bd.Option(1, "B", to_pos(.3563, .4772, .3775, .4871,0)))
c19 = bd.Contest(19, "c19", "", c19_opt, to_pos(.3252, .4256, .6356, .5010, 0), "c19.png")
contests.append(c19)

# Contest 20
c20_opt = []
c20_opt.append(bd.Option(0, "A", to_pos(.3563, .3869, .3775, .3968,0)))
c20_opt.append(bd.Option(1, "B", to_pos(.3563, .4067, .3775, .4167,0)))
c20 = bd.Contest(20, "c20", "", c20_opt, to_pos(.3252, .3532, .6356, .4196, 0), "c20.png")
contests.append(c20)

# Contest 21
c21_opt = []
c21_opt.append(bd.Option(0, "A", to_pos(.3563, .3165, .3775, .3264,0)))
c21_opt.append(bd.Option(1, "B", to_pos(.3563, .3343, .3775, .3442,0)))
c21 = bd.Contest(21, "c21", "", c21_opt, to_pos(.3252, .2708, .6356, .3482, 0), "c21.png")
contests.append(c21)

# Contest 22
c22_opt = []
c22_opt.append(bd.Option(0, "A", to_pos(.0621, .7262, .0833, .7361,0)))
c22_opt.append(bd.Option(1, "B", to_pos(.0621, .7460, .0833, .7560,0)))
c22_opt.append(bd.Option(2, "C", to_pos(.0621, .7629, .0833, .7728,0)))
c22_opt.append(bd.Option(3, "D", to_pos(.0621, .7808, .0833, .7897,0)))
c22 = bd.Contest(22, "c22", "", c22_opt, to_pos(.0310, .6915, .3185, .7907,0), "c22.png")
contests.append(c22)

# Contest 23
c23_opt = []
c23_opt.append(bd.Option(0, "A", to_pos(.0621, .5665, .0833, .5764,0)))
c23_opt.append(bd.Option(1, "B", to_pos(.0621, .5843, .0833, .5942,0)))
c23_opt.append(bd.Option(2, "C", to_pos(.0621, .6012, .0833, .6111,0)))
c23_opt.append(bd.Option(3, "D", to_pos(.0621, .6190, .0833, .6290,0)))
c23_opt.append(bd.Option(4, "E", to_pos(.0621, .6379, .0833, .6478,0)))
c23_opt.append(bd.Option(5, "F", to_pos(.0621, .6548, .0833, .6647,0)))
c23_opt.append(bd.Option(6, "G", to_pos(.0621, .6736, .0833, .6835,0)))
c23 = bd.Contest(23, "c23", "", c23_opt, to_pos(.0310, .5258, .3185, .6865,0), "c23.png")
contests.append(c23)

# Contest 24
c24_opt = []
c24_opt.append(bd.Option(0, "A", to_pos(.0621, .3879, .0833, .3978,0)))
c24_opt.append(bd.Option(1, "B", to_pos(.0621, .4048, .0833, .4147,0)))
c24_opt.append(bd.Option(2, "C", to_pos(.0621, .4236, .0833, .4335,0)))
c24_opt.append(bd.Option(3, "D", to_pos(.0621, .4415, .0833, .4514,0)))
c24_opt.append(bd.Option(4, "E", to_pos(.0621, .4593, .0833, .4692,0)))
c24_opt.append(bd.Option(5, "F", to_pos(.0621, .4772, .0833, .4871,0)))
c24 = bd.Contest(24, "c24", "", c24_opt, to_pos(.0310, .3423, .3185, .4901,0), "c24.png")
contests.append(c24)

# Wrap contests in a ballot definition
ballot = bd.Ballot(contests=contests, ballot_file="CNNScan/Samples/Oregon/or2018ballot.pdf")
# Provide interface to access ballot.
def get_sample_ballot():
	global ballot
	output_directory = "temp"
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
	return ballot
	
del bd
