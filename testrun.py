# Begin training neural net using default data / parameters
from CNNScan.Reco import Driver, Settings
import CNNScan.Samples
if __name__ == "__main__":
	settings = Settings.generate_default_settings()
	# Choose to use real Oregon data (on which the network performs poorly)
	# Or choose randomly generate data, on which the network performs decently.
	#module = CNNScan.Samples.Oregon
	module = CNNScan.Samples.Montana
	#module = CNNScan.Samples.Random

	Driver.contest_entry_point(settings, module)