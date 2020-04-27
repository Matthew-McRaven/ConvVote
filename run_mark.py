# Begin training neural net using default data / parameters
import torch
import torchvision
import numpy as np

from CNNScan.Mark import gan
import CNNScan.Mark.Settings

# Choose to use real Oregon data (on which the network performs poorly)
# Or choose randomly generate data, on which the network performs decently.
config = CNNScan.Mark.Settings.generate_default_settings()
config['epochs'] = 50

transforms = torchvision.transforms.Compose([
 											 torchvision.transforms.ToTensor()
											])

data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark, transforms)
loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

disc_model = CNNScan.Mark.gan.MarkDiscriminator(config)
gen_model = CNNScan.Mark.gan.MarkGenerator(config, config['gen_seed_len'])

print(disc_model)
print(gen_model)

CNNScan.Mark.gan.train_once(config, gen_model, disc_model, loader, loader)
