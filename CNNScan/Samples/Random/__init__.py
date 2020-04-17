"""
This module randomly generates a ballot definition.
It conforms to the same interface as Samples.Oregon and other.
"""
from . import ElectionFaker
import CNNScan.utils

def get_sample_ballot(factory):
	ballot = ElectionFaker.create_fake_ballot(factory, 3, 3)
	for contest in ballot.contests:
		ballot.pages.append(ElectionFaker.create_fake_contest_image(contest))
	return ballot