"""
This module randomly generates a ballot definition.
It conforms to the same interface as Samples.Oregon and other.
"""
from . import ElectionFaker
import CNNScan.utils

def get_sample_ballot():
	ballot = ElectionFaker.create_fake_ballot(3, 3)
	for contest in ballot.contests:
		contest.image = ElectionFaker.create_fake_contest_image(contest)
	return ballot

def create_marked_ballots(ballot, mark_database, count=0):
	rval = ElectionFaker.create_fake_marked_ballots(ballot, mark_database, count)
	return rval