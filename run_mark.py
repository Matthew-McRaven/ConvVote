import tempfile
import os

import torch
import torchvision
import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt 

from CNNScan.Mark import gan
import CNNScan.Mark.Settings

config = CNNScan.Mark.Settings.generate_default_settings()

config['epochs'] = 0
config['batch_size'] = 5

data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark)
loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

disc_model = CNNScan.Mark.gan.MarkDiscriminator(config)
gen_model = CNNScan.Mark.gan.MarkGenerator(config, config['gen_seed_len'])

print(disc_model)
print(gen_model)

CNNScan.Mark.gan.train_once(config, gen_model, disc_model, loader, loader)

count=4
images = CNNScan.Mark.gan.generate_images(gen_model, count, config)

#im,labels = next(iter(loader))
with tempfile.TemporaryDirectory() as path:
	CNNScan.Mark.raster_images(images, path, show_images=True)
#raster_images(im)