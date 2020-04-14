"""
This module returns a ballot based on Mulnomah County's (Oregon) 2018 election.
The ballot is available from:
	https://multco.us/elections/sample-ballots-november-2018-general-election
"""
import CNNScan.Ballot.BallotDefinitions  
import CNNScan.Samples.utils
import CNNScan.Samples.Oregon

import CNNScan.Reco.Load
from . import percents
bd = CNNScan.Ballot.BallotDefinitions 
to_pixel = CNNScan.Ballot.Positions.to_pixel_pos
contests = []

# Manually create contests for each entry on the Multnomah county ballot.

# Contest 00
c00_opt = []
c00_opt.append(bd.Option(0, "A", to_pixel(70, 667, 120, 699)))
c00_opt.append(bd.Option(1, "B", to_pixel(70, 732, 120, 764)))
c00 = bd.Contest(0, "c00", "", c00_opt, to_pixel(0, 0, 704, 782), "c00.png")
contests.append(c00)


# Contest 01
c01_opt = []
c01_opt.append(bd.Option(0, "A", to_pixel(70, 444, 120, 476)))
c01_opt.append(bd.Option(1, "B", to_pixel(70, 510, 120, 540)))
c01 = bd.Contest(1, "c01", "", c01_opt, to_pixel(0, 0, 704, 562), "c01.png")
contests.append(c01)

# Contest 02
c02_opt = []
c02_opt.append(bd.Option(0, "A", to_pixel(70, 486, 120, 515)))
c02_opt.append(bd.Option(1, "B", to_pixel(70, 552, 120, 582)))
c02 = bd.Contest(2, "c02", "", c02_opt, to_pixel(0, 0, 702, 596), "c02.png")
contests.append(c01)

# Contest 03
c03_opt = []
c03_opt.append(bd.Option(0, "A", to_pixel(70, 770, 120, 800)))
c03_opt.append(bd.Option(1, "B", to_pixel(70, 836, 120, 865)))
c03 = bd.Contest(3, "c03", "", c03_opt, to_pixel(0, 0, 696, 890), "c03.png")
contests.append(c03)

# Contest 04
c04_opt = []
c04_opt.append(bd.Option(0, "A", to_pixel(70, 641, 120, 669)))
c04_opt.append(bd.Option(1, "B", to_pixel(70, 706, 120, 735)))
c04 = bd.Contest(4, "c04", "", c04_opt, to_pixel(0, 0, 698, 756), "c04.png")
contests.append(c04)

# Contest 05
c05_opt = []
c05_opt.append(bd.Option(0, "A", to_pixel(70, 682, 120, 714)))
c05_opt.append(bd.Option(1, "B", to_pixel(70, 747, 120, 776)))
c05 = bd.Contest(5, "c05", "", c05_opt, to_pixel(0, 0, 704, 798), "c05.png")
contests.append(c05)

# Contest 06
c06_opt = []
c06_opt.append(bd.Option(0, "A", to_pixel(74, 772, 124, 803)))
c06_opt.append(bd.Option(1, "B", to_pixel(74, 837, 124, 866)))
c06 = bd.Contest(6, "c06", "", c06_opt, to_pixel(0, 0, 638, 884), "c06.png")
contests.append(c06)

# Contest 07
c07_opt = []
c07_opt.append(bd.Option(0, "A", to_pixel(70, 846, 120, 876)))
c07_opt.append(bd.Option(1, "B", to_pixel(70, 910, 120, 940)))
c07 = bd.Contest(7, "c07", "", c07_opt, to_pixel(0, 0, 638, 970), "c07.png")
contests.append(c07)

# Contest 08
c08_opt = []
c08_opt.append(bd.Option(0, "A", to_pixel(76, 181, 120, 216)))
c08_opt.append(bd.Option(1, "B", to_pixel(76, 250, 120, 280)))
c08 = bd.Contest(8, "c08", "", c08_opt, to_pixel(0, 0, 638, 294), "c08.png")
contests.append(c08)

# Contest 09
c09_opt = []
c09_opt.append(bd.Option(0, "A", to_pixel(74, 182, 124, 215)))
c09_opt.append(bd.Option(1, "B", to_pixel(74, 249, 124, 280)))
c09 = bd.Contest(9, "c09", "", c09_opt, to_pixel(0, 0, 644, 306), "c09.png")
contests.append(c09)

# Contest 10
c10_opt = []
c10_opt.append(bd.Option(0, "A", to_pixel(72, 173, 122, 205)))
c10_opt.append(bd.Option(1, "B", to_pixel(72, 240, 122, 280)))
c10 = bd.Contest(10, "c10", "", c10_opt, to_pixel(0, 0, 706, 294), "c10.png")
contests.append(c10)

# Contest 11
c11_opt = []
c11_opt.append(bd.Option(0, "A", to_pixel(69, 150, 119, 182)))
c11_opt.append(bd.Option(1, "B", to_pixel(69, 215, 119, 246)))
c11_opt.append(bd.Option(2, "C", to_pixel(69, 280, 119, 311)))
c11 = bd.Contest(11, "c11", "", c11_opt, to_pixel(0, 0, 702, 332), "c11.png")
contests.append(c11)

# Contest 12
c12_opt = []
c12_opt.append(bd.Option(0, "A", to_pixel(70, 152, 117, 182)))
c12_opt.append(bd.Option(1, "B", to_pixel(70, 216, 117, 247)))
c12_opt.append(bd.Option(2, "C", to_pixel(70, 280, 117, 313)))
c12 = bd.Contest(12, "c12", "", c12_opt, to_pixel(0, 0, 700, 332), "c12.png")
contests.append(c12)

# Contest 13
c13_opt = []
c13_opt.append(bd.Option(0, "A", to_pixel(70, 184, 120, 217)))
c13_opt.append(bd.Option(1, "B", to_pixel(70, 249, 120, 281)))
c13 = bd.Contest(13, "c13", "", c13_opt, to_pixel(0, 0, 716, 294), "c13.png")
contests.append(c13)

# Contest 14
c14_opt = []
c14_opt.append(bd.Option(0, "A", to_pixel(70, 184, 120, 216)))
c14_opt.append(bd.Option(1, "B", to_pixel(70, 250, 120, 280)))
c14 = bd.Contest(14, "c14", "", c14_opt, to_pixel(0, 0, 706, 300), "c14.png")
contests.append(c14)

# Contest 15
c15_opt = []
c15_opt.append(bd.Option(0, "A", to_pixel(71, 186, 121, 216)))
c15_opt.append(bd.Option(1, "B", to_pixel(71, 251, 121, 281)))
c15 = bd.Contest(15, "c15", "", c15_opt, to_pixel(0, 0, 706, 306), "c15.png")
contests.append(c15)

# Contest 16
c16_opt = []
c16_opt.append(bd.Option(0, "A", to_pixel(73, 187, 123, 220)))
c16_opt.append(bd.Option(1, "B", to_pixel(73, 254, 123, 284)))
c16_opt.append(bd.Option(2, "C", to_pixel(73, 319, 123, 350)))
c16 = bd.Contest(16, "c16", "", c16_opt, to_pixel(0, 0, 706, 368), "c16.png")
contests.append(c16)

# Contest 17
c17_opt = []
c17_opt.append(bd.Option(0, "A", to_pixel(69, 121, 119, 153)))
c17_opt.append(bd.Option(1, "B", to_pixel(69, 185, 119, 218)))
c17 = bd.Contest(17, "c17", "", c17_opt, to_pixel(0, 0, 696, 240), "c17.png")
contests.append(c17)

# Contest 18
c18_opt = []
c18_opt.append(bd.Option(0, "A", to_pixel(72, 122, 120, 152)))
c18_opt.append(bd.Option(1, "B", to_pixel(72, 185, 120, 217)))
c18 = bd.Contest(18, "c18", "", c18_opt, to_pixel(0, 0, 702, 236), "c18.png")
contests.append(c18)

# Contest 19
c19_opt = []
c19_opt.append(bd.Option(0, "A", to_pixel(70, 120, 120, 150)))
c19_opt.append(bd.Option(1, "B", to_pixel(70, 180, 120, 210)))
c19 = bd.Contest(19, "c19", "", c19_opt, to_pixel(0, 0, 704, 236), "c19.png")
contests.append(c19)

# Contest 20
c20_opt = []
c20_opt.append(bd.Option(0, "A", to_pixel(75, 128, 125, 158)))
c20_opt.append(bd.Option(1, "B", to_pixel(75, 191, 125, 225)))
c20 = bd.Contest(20, "c20", "", c20_opt, to_pixel(0, 0, 710, 248), "c20.png")
contests.append(c20)

# Contest 21
c21_opt = []
c21_opt.append(bd.Option(0, "A", to_pixel(72, 163, 122, 193)))
c21_opt.append(bd.Option(1, "B", to_pixel(72, 227, 122, 260)))
c21 = bd.Contest(21, "c21", "", c21_opt, to_pixel(0, 0, 700, 276), "c21.png")
contests.append(c21)

# Contest 22
c22_opt = []
c22_opt.append(bd.Option(0, "A", to_pixel(70, 118, 120, 150)))
c22_opt.append(bd.Option(1, "B", to_pixel(70, 183, 120, 214)))
c22_opt.append(bd.Option(2, "C", to_pixel(70, 246, 120, 281)))
c22_opt.append(bd.Option(3, "D", to_pixel(70, 313, 120, 344)))
c22 = bd.Contest(22, "c22", "", c22_opt, to_pixel(0, 0, 638, 364), "c22.png")
contests.append(c22)

# Contest 23
c23_opt = []
c23_opt.append(bd.Option(0, "A", to_pixel(71, 150, 121, 180)))
c23_opt.append(bd.Option(1, "B", to_pixel(71, 214, 121, 244)))
c23_opt.append(bd.Option(2, "C", to_pixel(71, 278, 121, 310)))
c23_opt.append(bd.Option(3, "D", to_pixel(71, 343, 121, 375)))
c23_opt.append(bd.Option(4, "E", to_pixel(71, 409, 121, 439)))
c23_opt.append(bd.Option(5, "F", to_pixel(71, 472, 121, 501)))
c23_opt.append(bd.Option(6, "G", to_pixel(71, 535, 121, 569)))
c23 = bd.Contest(23, "c23", "", c23_opt, to_pixel(0, 0, 646, 592), "c23.png")
contests.append(c23)

# Contest 24
c24_opt = []
c24_opt.append(bd.Option(0, "A", to_pixel(70, 165, 120, 195)))
c24_opt.append(bd.Option(1, "B", to_pixel(70, 234, 120, 263)))
c24_opt.append(bd.Option(2, "C", to_pixel(70, 298, 120, 327)))
c24_opt.append(bd.Option(3, "D", to_pixel(70, 360, 120, 390)))
c24_opt.append(bd.Option(4, "E", to_pixel(70, 425, 120, 455)))
c24_opt.append(bd.Option(5, "F", to_pixel(70, 487, 120, 521)))
c24 = bd.Contest(24, "c24", "", c24_opt, to_pixel(0, 0, 638, 544), "c24.png")
contests.append(c24)

# Wrap contests in a ballot definition
ballot = bd.Ballot(contests=contests, ballot_file="CNNScan/Samples/Oregon/or2018ballot.pdf")
# Provide interface to access ballot.
def get_sample_ballot():
	global ballot
	for contest in ballot.contests:
		if contest.image is None:
			contest.image = CNNScan.Samples.utils.load_template_image(CNNScan.Samples.Oregon, contest)
	return ballot
	
del bd
