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


config['epochs'] = 400
config['batch_size'] = 5
config['cuda'] = False 
data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark)
loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

encoder = CNNScan.Mark.encoder.AutoEncoder(config)

print(encoder)

CNNScan.Mark.encoder.train_autoencoder(config, encoder, loader, loader)

count=4

images,_ = next(iter(loader))
CNNScan.Mark.raster_images(images, "./temp/here1", show_images=True)
CNNScan.Mark.raster_images(encoder(images), "./temp/here2", show_images=True)
