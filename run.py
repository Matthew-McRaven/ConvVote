# Begin training neural net using default data / parameters
import torch
import torchvision
import numpy as np

from CNNScan.Reco import Driver, Settings
import CNNScan.Samples


from CNNScan.Reco import Driver, Settings
import CNNScan.Samples

# Choose to use real Oregon data (on which the network performs poorly)
# Or choose randomly generate data, on which the network performs decently.
config = Settings.generate_default_settings()
config['epochs'] = 500
markdb = CNNScan.Mark.MarkDatabase()
markdb.insert_mark(CNNScan.Mark.BoxMark())
markdb.insert_mark(CNNScan.Mark.InvertMark())
markdb.insert_mark(CNNScan.Mark.XMark())


# Create fake data that can be used 
ballot_factory = CNNScan.Ballot.BallotDefinitions.BallotFactory()
#ballots = [CNNScan.Samples.Random.get_sample_ballot(ballot_factory) for i in range(2)]
CNNScan.Samples.Oregon.get_sample_ballot(ballot_factory)
#CNNScan.Samples.Montana.get_sample_ballot(ballot_factory)




# Create faked marked ballots from ballot factories.
data = CNNScan.Reco.Load.GeneratingDataSet(ballot_factory, markdb, 100)
load = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

# Attempts to train model.
model = CNNScan.Reco.OneNet.BallotRecognizer(config, ballot_factory)
print(model)
model = CNNScan.Reco.OneNet.train_election(model, config, ballot_factory, load, load)
