import Ballot.BallotDefinitions as bd
_ = None

# Contest 00
c00_opt = []
c00_opt.append(bd.Option(0, "A", (70, 667, 120, 699)))
c00_opt.append(bd.Option(1, "B", (70, 732, 120, 764)))
c00 = bd.Contest(0, "00", "", c00_opt, (0, 0, _, _))

# Contest 01
c01_opt = []
c01_opt.append(bd.Option(0, "A", (70, 444, 120, 476)))
c01_opt.append(bd.Option(1, "B", (70, 510, 120, 540)))
c01 = bd.Contest(1, "01", "", c01_opt, (0, 0, _, _))

# Contest 02
c02_opt = []
c02_opt.append(bd.Option(0, "A", (70, 486, 120, 515)))
c02_opt.append(bd.Option(1, "B", (70, 552, 120, 582)))
c02 = bd.Contest(2, "02", "", c02_opt, (0, 0, _, _))

# Contest 03
c03_opt = []
c03_opt.append(bd.Option(0, "A", (70, 770, 120, 800)))
c03_opt.append(bd.Option(1, "B", (70, 836, 120, 865)))
c03 = bd.Contest(3, "03", "", c03_opt, (0, 0, _, _))

# Contest 04
c04_opt = []
c04_opt.append(bd.Option(0, "A", (70, 641, 120, 669)))
c04_opt.append(bd.Option(1, "B", (70, 706, 120, 735)))
c04 = bd.Contest(4, "04", "", c04_opt, (0, 0, _, _))

# Contest 05
c05_opt = []
c05_opt.append(bd.Option(0, "A", (70, 682, 120, 714)))
c05_opt.append(bd.Option(1, "B", (70, 747, 120, 776)))
c05 = bd.Contest(5, "05", "", c05_opt, (0, 0, _, _))