"""
This module returns a ballot based on Mulnomah County's (Oregon) 2018 election.
The ballot is available from:
	https://multco.us/elections/sample-ballots-november-2018-general-election
"""

import os
import copy
import pickle
import torchvision
import argparse
import torch
import torchvision
import numpy as np

import CNNScan.Ballot.BallotDefinitions  
import CNNScan.Samples.utils
import CNNScan.Samples.Oregon

import CNNScan.Reco.Load
bd = CNNScan.Ballot.BallotDefinitions 
to_pos = CNNScan.Ballot.Positions.to_percent_pos
	

if __name__=="__main__":
	# Manually create contests for each entry on the Multnomah county ballot.
	alph = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
	# print(pickle.__doc__)
	parser = argparse.ArgumentParser()
	parser.add_argument("--input",help="Contest and option data file.", required=True, type=str)
	parser.add_argument("--output", help="output file location.", required=True, type=str)
	args = parser.parse_args()

	file = open(args.input, "r") 
	lines = file.readlines()
	temp_contests=[]
	options=[]
	for line in lines:
		temp_dict={}
		temp_str = line.split(",")
		temp_str[len(temp_str)-1] = temp_str[len(temp_str)-1].strip()
		if temp_str[0]=="C":
			temp_dict['id']=int(temp_str[1])
			temp_dict['x1']=float(temp_str[3])
			temp_dict['y1']=float(temp_str[4])
			temp_dict['x2']=float(temp_str[5])
			temp_dict['y2']=float(temp_str[6])
			temp_dict['page']=int(temp_str[7])
			temp_dict['page-width']=int(temp_str[8])
			temp_dict['page-height']=int(temp_str[9])
			temp_dict['options']=[]
			temp_dict['ballot']=int(temp_str[10])
			temp_contests.append(temp_dict)
		else :
			temp_dict['id']=int(temp_str[1])
			temp_dict['x1']=float(temp_str[3])
			temp_dict['y1']=float(temp_str[4])
			temp_dict['x2']=float(temp_str[5])
			temp_dict['y2']=float(temp_str[6])
			temp_dict['contest_id']=int(temp_str[7])
			options.append(temp_dict)

	for op in options:

		for con in temp_contests:
			if op['contest_id']==con['id']:
				con['options'].append(op)
				break

	# contests
	contests=[]
	for contest in temp_contests:
		ops=[]

		i=0
		for op in contest['options']:
			ops.append(bd.Option(i, alph[i], to_pos(op['x1'],op['y1'],op['x2'],op['y2'],contest['page']-1)))
			i+=1

			x1=contest['x1']
			y1=contest['y1']
			x2=contest['x2']
			y2=contest['y2']

		con = bd.Contest(contest['id']-1, f"c{contest['id']}", "", ops, to_pos(contest['x1'],contest['y1'],contest['x2'],contest['y2'],contest['page']-1), f"c{contest['id']-1}.png")
		contests.append(con)
	
	# save contests and ballot file
	picklefile=open(args.output, 'wb')
	pickle.dump(contests, picklefile)
	picklefile.close()
	print(f"Done saving {len(contests)} contests to {args.output}")

	



