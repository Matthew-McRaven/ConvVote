"""
This module randomly generates a ballot definition.
It conforms to the same interface as Samples.Oregon and other.
"""
import CNNScan.Reco.ElectionFaker

def get_sample_ballot():
	ballot = CNNScan.Reco.ElectionFaker.create_fake_ballot(3, 10)
	for contest in ballot.contests:
		contest.image = CNNScan.Reco.ElectionFaker.create_fake_contest_image(contest)
	return ballot