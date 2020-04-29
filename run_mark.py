import tempfile
import os

import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

from CNNScan.Mark import gan
import CNNScan.Mark.Settings

# Choose to use real Oregon data (on which the network performs poorly)
# Or choose randomly generate data, on which the network performs decently.
config = CNNScan.Mark.Settings.generate_default_settings()

transforms = torchvision.transforms.Compose([
											 #torchvision.transforms.Grayscale(),
 											 torchvision.transforms.ToTensor(),
											 #torchvision.transforms.Normalize((1,),(127.5,))
											 #torchvision.transforms.Lambda(lambda x: (x[0] + x[1] + x[2])/3)
											])

data = CNNScan.Mark.gan.get_marks_dataset(CNNScan.Mark, transforms)
loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True)

disc_model = CNNScan.Mark.gan.MarkDiscriminator(config)
gen_model = CNNScan.Mark.gan.MarkGenerator(config, config['gen_seed_len'])

print(disc_model)
print(gen_model)

config['epochs'] = 1
CNNScan.Mark.gan.train_once(config, gen_model, disc_model, loader, loader)

count=4
images = CNNScan.Mark.gan.generate_images(gen_model, count, config)

# Convert a 2 channel tensor to an image.
asLA= torchvision.transforms.Compose([
	torchvision.transforms.Normalize((-1/127.5,),(1/127.5,)),
	torchvision.transforms.ToPILImage(mode='LA')
])
# Convert a 1 channel tensor to an image.
asL= torchvision.transforms.Compose([
	torchvision.transforms.Normalize((-1/127.5,),(1/127.5,)),
	torchvision.transforms.ToPILImage(mode='L')
])

# Perform channel decomposition and analysis on each of the generated images
with tempfile.TemporaryDirectory() as path:
	for i,image in enumerate(images):
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		fig.suptitle(f'Image {i} Channel Decomposition')

		im1 = asLA(images[0]).convert('RGBA')
		im2 = asL( torch.stack((images[0][0],)) ).convert('RGBA')
		im3 = asL( torch.stack((images[0][1],))  ).convert('RGBA')
		im4 = asL( torch.stack((images[0][0]-images[0][1],)) ).convert('RGBA')
		ax1.imshow(im1)

		# Add axis titles and images.
		ax1.set_title("Original Image")
		ax2.imshow(im2)
		ax2.set_title("Only L Channel")
		ax3.imshow(im3)
		ax3.set_title("Only A Channel")
		ax4.imshow(im4)
		ax4.set_title("L-A")
		for ax in fig.get_axes():
			ax.label_outer()

		fig.savefig(path+f"/file{i}.png", dpi=400)

		Image.open(path+f"/file{i}.png").show()