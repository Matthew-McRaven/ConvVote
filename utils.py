import math

import torch
import pylab as plt

# Our classes.
import Election as Election

# Determine if cuda is available on the machine
def cuda(arr, config):
    if config['cuda']:
        return arr.cuda()
    return arr

# Determine if a number is a power of 2 or not and the number is non-zero.
def is_power2(number):
	return number > 0 and math.ceil(math.log(number, 2)) == math.floor(math.log(number, 2)) 

# Convert images to tensors, and apply normalization if necessary
def image_to_tensor(image):
	#TODO: apply image normalization.
	return torch.from_numpy(image)
	
# Visualize marked ballots.
def show_ballot(marked: Election.MarkedContest):
	plt.imshow(marked.image, cmap='gray', interpolation='nearest')
	plt.show()