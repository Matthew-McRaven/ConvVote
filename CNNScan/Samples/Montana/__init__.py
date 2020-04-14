"""
This module returns a ballot based on Mulnomah County's (Oregon) 2018 election.
The ballot is available from:
	https://multco.us/elections/sample-ballots-november-2018-general-election

This ballot took 3 hours to markup from start to finish.
This included creating the numbers file & measuring the 33 contests,
creating this file from numbers spreadsheet, and debugging the sizes
of contests and options.
"""

import os

import torchvision
import numpy as np

import CNNScan.Ballot.BallotDefinitions  
import CNNScan.Samples.utils
import CNNScan.Samples.Oregon

import CNNScan.Reco.Load
bd = CNNScan.Ballot.BallotDefinitions 
to_pos = CNNScan.Ballot.Positions.to_percent_pos
contests = []

# Manually create contests for each entry in the Montana Ballot.

# Contest 0
c0_opt = []
c0_opt.append(bd.Option(0, "A", to_pos(.0845, .1574, .1212, .1680, 0)))
c0_opt.append(bd.Option(1, "B", to_pos(.0845, .1858, .1212, .1964, 0)))
c0_opt.append(bd.Option(2, "C", to_pos(.0845, .2152, .1212, .2258, 0)))
c0_opt.append(bd.Option(3, "D", to_pos(.0845, .2447, .1212, .2554, 0)))
c0_opt.append(bd.Option(4, "E", to_pos(.0845, .2723, .1212, .2829, 0)))
c0 = bd.Contest(0, "c0", "", c0_opt, to_pos(.0763, .1201, .3492, .3031, 0), "c0.png")
contests.append(c0)

# Contest 1
c1_opt = []
c1_opt.append(bd.Option(0, "A", to_pos(.0845, .3514, .1212, .3620, 0)))
c1_opt.append(bd.Option(1, "B", to_pos(.0845, .3806, .1212, .3912, 0)))
c1_opt.append(bd.Option(2, "C", to_pos(.0845, .4085, .1212, .4191, 0)))
c1_opt.append(bd.Option(3, "D", to_pos(.0845, .4380, .1212, .4486, 0)))
c1_opt.append(bd.Option(4, "E", to_pos(.0845, .4668, .1212, .4774, 0)))
c1 = bd.Contest(1, "c1", "", c1_opt, to_pos(.0763, .3039, .3492, .4967, 0), "c1.png")
contests.append(c1)

# Contest 2
c2_opt = []
c2_opt.append(bd.Option(0, "A", to_pos(.0845, .5347, .1212, .5453, 0)))
c2_opt.append(bd.Option(1, "B", to_pos(.0845, .5634, .1212, .5740, 0)))
c2_opt.append(bd.Option(2, "C", to_pos(.0845, .5926, .1212, .6032, 0)))
c2_opt.append(bd.Option(3, "C", to_pos(.0845, .6212, .1212, .6318, 0)))
c2 = bd.Contest(2, "c2", "", c2_opt, to_pos(.0763, .4967, .3492, .6528, 0), "c2.png")
contests.append(c2)

# Contest 3
c3_opt = []
c3_opt.append(bd.Option(0, "A", to_pos(.3664, .1571, .4031, .1677, 0)))
c3_opt.append(bd.Option(1, "B", to_pos(.3664, .1864, .4031, .1970, 0)))
c3_opt.append(bd.Option(2, "C", to_pos(.3664, .2153, .4031, .2259, 0)))
c3 = bd.Contest(3, "c3", "", c3_opt, to_pos(.3595, .1225, .6258, .2451, 0), "c3.png")
contests.append(c3)

# Contest 4
c4_opt = []
c4_opt.append(bd.Option(0, "A", to_pos(.3664, .2721, .4031, .2827, 0)))
c4_opt.append(bd.Option(1, "B", to_pos(.3664, .3113, .4031, .3219, 0)))
c4_opt.append(bd.Option(2, "C", to_pos(.3664, .3399, .4031, .3505, 0)))
c4_opt.append(bd.Option(3, "D", to_pos(.3664, .3791, .4031, .3897, 0)))
c4_opt.append(bd.Option(4, "E", to_pos(.3664, .4189, .4031, .4295, 0)))
c4 = bd.Contest(4, "c4", "", c4_opt, to_pos(.3595, .2467, .6258, .4485, 0), "c4.png")
contests.append(c4)

# Contest 5
c5_opt = []
c5_opt.append(bd.Option(0, "A", to_pos(.3664, .4758, .4031, .4864, 0)))
c5_opt.append(bd.Option(1, "B", to_pos(.3664, .5046, .4031, .5152, 0)))
c5_opt.append(bd.Option(2, "C", to_pos(.3664, .5329, .4031, .5435, 0)))
c5_opt.append(bd.Option(3, "D", to_pos(.3664, .5624, .4031, .5730, 0)))
c5 = bd.Contest(5, "c5", "", c5_opt, to_pos(.3595, .4510, .6258, .5923, 0), "c5.png")
contests.append(c5)

# Contest 6
c6_opt = []
c6_opt.append(bd.Option(0, "A", to_pos(.3664, .6190, .4031, .6296, 0)))
c6_opt.append(bd.Option(1, "B", to_pos(.3664, .6470, .4031, .6577, 0)))
c6_opt.append(bd.Option(2, "C", to_pos(.3664, .6762, .4031, .6868, 0)))
c6_opt.append(bd.Option(3, "D", to_pos(.3664, .7053, .4031, .7160, 0)))
c6_opt.append(bd.Option(4, "E", to_pos(.3664, .7339, .4031, .7445, 0)))
c6 = bd.Contest(6, "c6", "", c6_opt, to_pos(.3595, .5921, .6258, .7631, 0), "c6.png")
contests.append(c6)

# Contest 7
c7_opt = []
c7_opt.append(bd.Option(0, "A", to_pos(.3664, .7912, .4031, .8018, 0)))
c7_opt.append(bd.Option(1, "B", to_pos(.3664, .8203, .4031, .8309, 0)))
c7_opt.append(bd.Option(2, "C", to_pos(.3664, .8490, .4031, .8596, 0)))
c7_opt.append(bd.Option(3, "D", to_pos(.3664, .8777, .4031, .8883, 0)))
c7 = bd.Contest(7, "c7", "", c7_opt, to_pos(.3595, .7655, .6258, .9077, 0), "c7.png")
contests.append(c7)

# Contest 8
c8_opt = []
c8_opt.append(bd.Option(0, "A", to_pos(.6487, .1573, .6854, .1679, 0)))
c8_opt.append(bd.Option(1, "B", to_pos(.6487, .1775, .6854, .1881, 0)))
c8_opt.append(bd.Option(2, "C", to_pos(.6487, .1979, .6854, .2086, 0)))
c8 = bd.Contest(8, "c8", "", c8_opt, to_pos(.6422, .1201, .9036, .2508, 0), "c8.png")
contests.append(c8)

# Contest 9
c9_opt = []
c9_opt.append(bd.Option(0, "A", to_pos(.6487, .2546, .6854, .2652, 0)))
c9_opt.append(bd.Option(1, "B", to_pos(.6487, .2752, .6854, .2858, 0)))
c9_opt.append(bd.Option(2, "C", to_pos(.6487, .2956, .6854, .3062, 0)))
c9 = bd.Contest(9, "c9", "", c9_opt, to_pos(.6422, .2296, .9036, .3268, 0), "c9.png")
contests.append(c9)

# Contest 10
c10_opt = []
c10_opt.append(bd.Option(0, "A", to_pos(.6487, .3525, .6854, .3631, 0)))
c10_opt.append(bd.Option(1, "B", to_pos(.6487, .3726, .6854, .3832, 0)))
c10_opt.append(bd.Option(2, "C", to_pos(.6487, .3928, .6854, .4034, 0)))
c10 = bd.Contest(10, "c10", "", c10_opt, to_pos(.6422, .3260, .9036, .4240, 0), "c10.png")
contests.append(c10)

# Contest 11
c11_opt = []
c11_opt.append(bd.Option(0, "A", to_pos(.6487, .8399, .6854, .8505, 0)))
c11_opt.append(bd.Option(1, "B", to_pos(.6487, .8596, .6854, .8702, 0)))
c11 = bd.Contest(11, "c11", "", c11_opt, to_pos(.6422, .4845, .9036, .8799, 0), "c11.png")
contests.append(c11)



# Page 2
# Column 1

# Contest 12
c12_opt = []
c12_opt.append(bd.Option(0, "A", to_pos(.0837, .1574, .1205, .1680, 1)))
c12_opt.append(bd.Option(1, "B", to_pos(.0837, .1881, .1205, .1988, 1)))
c12 = bd.Contest(12, "c12", "", c12_opt, to_pos(.0768, .1220, .3464, .2192, 1), "c12.png")
contests.append(c12)

# Contest 13
c13_opt = []
c13_opt.append(bd.Option(0, "A", to_pos(.0837, .2561, .1205, .2667, 1)))
c13_opt.append(bd.Option(1, "B", to_pos(.0837, .2870, .1205, .2976, 1)))
c13_opt.append(bd.Option(2, "C", to_pos(.0837, .3071, .1205, .3177, 1)))
c13 = bd.Contest(13, "c13", "", c13_opt, to_pos(.0768, .2196, .3464, .3381, 1), "c13.png")
contests.append(c13)

# Contest 14
c14_opt = []
c14_opt.append(bd.Option(0, "A", to_pos(.0837, .3751, .1205, .3857, 1)))
c14_opt.append(bd.Option(1, "B", to_pos(.0837, .4056, .1205, .4163, 1)))
c14 = bd.Contest(14, "c14", "", c14_opt, to_pos(.0768, .3388, .3464, .4369, 1), "c14.png")
contests.append(c14)

# Contest 15
c15_opt = []
c15_opt.append(bd.Option(0, "A", to_pos(.0837, .4739, .1205, .4845, 1)))
c15_opt.append(bd.Option(1, "B", to_pos(.0837, .5041, .1205, .5147, 1)))
c15 = bd.Contest(15, "c15", "", c15_opt, to_pos(.0768, .4377, .3464, .5357, 1), "c15.png")
contests.append(c15)

# Contest 16
c16_opt = []
c16_opt.append(bd.Option(0, "A", to_pos(.0837, .5723, .1205, .5830, 1)))
c16_opt.append(bd.Option(1, "B", to_pos(.0837, .5923, .1205, .6029, 1)))
c16_opt.append(bd.Option(2, "C", to_pos(.0837, .6229, .1205, .6335, 1)))
c16 = bd.Contest(16, "c16", "", c16_opt, to_pos(.0768, .5359, .3464, .6544, 1), "c16.png")
contests.append(c16)

# Contest 17
c17_opt = []
c17_opt.append(bd.Option(0, "A", to_pos(.0837, .6915, .1205, .7021, 1)))
c17_opt.append(bd.Option(1, "B", to_pos(.0837, .7219, .1205, .7325, 1)))
c17 = bd.Contest(17, "c17", "", c17_opt, to_pos(.0768, .6536, .3464, .7516, 1), "c17.png")
contests.append(c17)

# Contest 18
c18_opt = []
c18_opt.append(bd.Option(0, "A", to_pos(.0837, .7902, .1205, .8008, 1)))
c18_opt.append(bd.Option(1, "B", to_pos(.0837, .8206, .1205, .8313, 1)))
c18 = bd.Contest(18, "c18", "", c18_opt, to_pos(.0768, .7523, .3464, .8522, 1), "c18.png")
contests.append(c18)

# Contest 19
c19_opt = []
c19_opt.append(bd.Option(0, "A", to_pos(.0837, .8886, .1205, .8992, 1)))
c19_opt.append(bd.Option(1, "B", to_pos(.0837, .9188, .1205, .9294, 1)))
c19 = bd.Contest(19, "c19", "", c19_opt, to_pos(.0768, .8523, .3464, .9504, 1), "c19.png")
contests.append(c19)



# Page 2
# Column 2

# Contest 20
c20_opt = []
c20_opt.append(bd.Option(0, "A", to_pos(.3668, .1578, .4036, .1684, 1)))
c20_opt.append(bd.Option(1, "B", to_pos(.3668, .1879, .4036, .1985, 1)))
c20 = bd.Contest(20, "c20", "", c20_opt, to_pos(.3595, .1215, .6291, .2196, 1), "c20.png")
contests.append(c20)

# Contest 21
c21_opt = []
c21_opt.append(bd.Option(0, "A", to_pos(.3668, .2563, .4036, .2669, 1)))
c21_opt.append(bd.Option(1, "B", to_pos(.3668, .2865, .4036, .2971, 1)))
c21 = bd.Contest(21, "c21", "", c21_opt, to_pos(.3595, .2195, .6291, .3175, 1), "c21.png")
contests.append(c21)

# Contest 22
c22_opt = []
c22_opt.append(bd.Option(0, "A", to_pos(.3668, .3547, .4036, .3653, 1)))
c22_opt.append(bd.Option(1, "B", to_pos(.3668, .3855, .4036, .3961, 1)))
c22_opt.append(bd.Option(2, "C", to_pos(.3668, .4057, .4036, .4163, 1)))
c22 = bd.Contest(22, "c22", "", c22_opt, to_pos(.3595, .3186, .6291, .4376, 1), "c22.png")
contests.append(c22)

# Contest 23
c23_opt = []
c23_opt.append(bd.Option(0, "A", to_pos(.3668, .4737, .4036, .4843, 1)))
c23_opt.append(bd.Option(1, "B", to_pos(.3668, .4939, .4036, .5045, 1)))
c23_opt.append(bd.Option(2, "C", to_pos(.3668, .5142, .4036, .5248, 1)))
c23 = bd.Contest(23, "c23", "", c23_opt, to_pos(.3595, .4372, .6291, .5457, 1), "c23.png")
contests.append(c23)


# Contest 24
c24_opt = []
c24_opt.append(bd.Option(0, "A", to_pos(.3668, .5819, .4036, .5925, 1)))
c24_opt.append(bd.Option(1, "B", to_pos(.3668, .6025, .4036, .6132, 1)))
c24_opt.append(bd.Option(2, "C", to_pos(.3668, .6328, .4036, .6435, 1)))
c24 = bd.Contest(24, "c24", "", c24_opt, to_pos(.3595, .5461, .6291, .6642, 1), "c24.png")
contests.append(c24)


# Contest 25
c25_opt = []
c25_opt.append(bd.Option(0, "A", to_pos(.3668, .7007, .4036, .7113, 1)))
c25_opt.append(bd.Option(1, "B", to_pos(.3668, .7315, .4036, .7421, 1)))
c25_opt.append(bd.Option(2, "C", to_pos(.3668, .7515, .4036, .7622, 1)))
c25 = bd.Contest(25, "c25", "", c25_opt, to_pos(.3595, .6648, .6291, .7825, 1), "c25.png")
contests.append(c25)


# Contest 26
c26_opt = []
c26_opt.append(bd.Option(0, "A", to_pos(.3668, .8198, .4036, .8304, 1)))
c26_opt.append(bd.Option(1, "B", to_pos(.3668, .8402, .4036, .8508, 1)))
c26_opt.append(bd.Option(2, "C", to_pos(.3668, .8706, .4036, .8813, 1)))
c26 = bd.Contest(26, "c26", "", c26_opt, to_pos(.3595, .7834, .6291, .9010, 1), "c26.png")
contests.append(c26)


# Page 2
# Column 3




# Contest 27
c27_opt = []
c27_opt.append(bd.Option(0, "A", to_pos(.6490, .1571, .6858, .1677, 1)))
c27_opt.append(bd.Option(1, "B", to_pos(.6490, .1881, .6858, .1987, 1)))
c27 = bd.Contest(27, "c27", "", c27_opt, to_pos(.6422, .1217, .9118, .2191, 1), "c27.png")
contests.append(c27)

# Contest 28
c28_opt = []
c28_opt.append(bd.Option(0, "A", to_pos(.6490, .2563, .6858, .2669, 1)))
c28_opt.append(bd.Option(1, "B", to_pos(.6490, .2867, .6858, .2973, 1)))
c28 = bd.Contest(28, "c28", "", c28_opt, to_pos(.6422, .2192, .9118, .3173, 1), "c28.png")
contests.append(c28)

# Contest 29
c29_opt = []
c29_opt.append(bd.Option(0, "A", to_pos(.6490, .3548, .6858, .3654, 1)))
c29_opt.append(bd.Option(1, "B", to_pos(.6490, .3852, .6858, .3958, 1)))
c29 = bd.Contest(29, "c29", "", c29_opt, to_pos(.6422, .3189, .9118, .4169, 1), "c29.png")
contests.append(c29)

# Contest 30
c30_opt = []
c30_opt.append(bd.Option(0, "A", to_pos(.6490, .4535, .6858, .4642, 1)))
c30_opt.append(bd.Option(1, "B", to_pos(.6490, .4840, .6858, .4947, 1)))
c30 = bd.Contest(30, "c30", "", c30_opt, to_pos(.6422, .4147, .9118, .5153, 1), "c30.png")
contests.append(c30)

# Contest 31
c31_opt = []
c31_opt.append(bd.Option(0, "A", to_pos(.6490, .5518, .6858, .5624, 1)))
c31_opt.append(bd.Option(1, "B", to_pos(.6490, .5826, .6858, .5933, 1)))
c31 = bd.Contest(31, "c31", "", c31_opt, to_pos(.6422, .5151, .9118, .6132, 1), "c31.png")
contests.append(c31)

# Contest 32
c32_opt = []
c32_opt.append(bd.Option(0, "A", to_pos(.6490, .6506, .6858, .6613, 1)))
c32_opt.append(bd.Option(1, "B", to_pos(.6490, .6812, .6858, .6918, 1)))
c32 = bd.Contest(32, "c32", "", c32_opt, to_pos(.6422, .6154, .9118, .7135, 1), "c32.png")
contests.append(c32)

# contests = [c00]
# Wrap contests in a ballot definition
ballot = bd.Ballot(contests=contests, ballot_file="CNNScan/Samples/Montana/min2018.pdf")
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
