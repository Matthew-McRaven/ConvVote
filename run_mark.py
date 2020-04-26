# Begin training neural net using default data / parameters
import torch
import torchvision
import numpy as np

from CNNScan.Mark import gan
import CNNScan.Samples


from CNNScan.Reco import Driver, Settings
import CNNScan.Samples

# Choose to use real Oregon data (on which the network performs poorly)
# Or choose randomly generate data, on which the network performs decently.
config = Settings.generate_default_settings()
config['epochs'] = 500

transforms = torchvision.transforms.Compose([
 											 torchvision.transforms.ToTensor()
											])

data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark, transforms)
loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

disc_model = CNNScan.Mark.gan.MarkDiscriminator(config, 4*128*128)
gen_model = CNNScan.Mark.gan.MarkGenerator(config, 10, 4*128*128)

print(disc_model)
print(gen_model)

CNNScan.Mark.gan.train_once(config, gen_model, disc_model, loader, loader)
