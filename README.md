# Getting Started
Please `pip install -r requirements.txt` to make sure you have the same python environment as we do.

# Finding contests on ballots
## Annotating a Ballot by Hand
Here we include a tutorial on how to use the ballot marking tool.
See the script `my_script.py` for details.

## Using a Neural Net to Markup a Ballot
TL;DR See the script *my_script.py* and download *my_pretrained_model* for an example on using our neural network to recognize contests bounding rectangles and option bounding rectangle

## Built-in Annotated Ballotes
TL;DR See CNNScan.Samples for built-in ballot definitions that may be operated on.

# Marking Ballots
We are planning on implementing two different methods of creating marks to apply to ballots.
One method will use a GAN trained on a corpus of marks.
This method is a work in progress, and does not work correctly yet.
The other method applies geometric shapes (currently X's and Boxes) to a contest.
This method is currently working, with working sample programs.

## Creating Marks
Generative Adversarial Nets (or [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network)) are a form of unsupervised machine learning.
They are [particularly apt](https://arxiv.org/abs/1812.04948) at image generation tasks.
We hope to train a GAN to generate realistic marks from
These marks would be applied to ballots, hopefully simulating real voters marking up a ballot using a pen.

### Theory of GANs

GAN's consist of two separate networks.
The first network, the generator, takes in a small number of noise values and using fully-connected and convolutional-transpose layers it creates a fixed-sized image.
The second network, the discriminator takes in images from both the generator and from a corpus of "real" images.
The discriminator attempts to classify if a a particular image is real or generated.

Training the network is done is steps.
For each epoch, divide the real images into `n` batches, and create an equal number of generated batches using the discriminator.
Shuffle the order of the real and generated batches
Batches fed through the discriminator should only contain one class of images--all images should either be all real images or all images should be generated.
Mixing classes within batches is not recommended, since the distributions of real and generated datasets are different.

If the number of batches seen mod `k + 1` (a hyperparameter usually set as 1) equals 0, then train the generator while holding the discriminator constant.
Otherwise, train the discriminator.

### Using GANs to generate Marks (WIP)
We have created a corpus of images that look like marks that might be applied to ballots.
More examples are provided in `CNNScan/Marks/marks`.
![Sample marks](/docs/sample_marks.png).

Using this corpus, we have designed a GAN that is trained to create marks similar to the original corpus.
When working, this gives us the ability to generate an unlimited number of unique marks, much like marks applied by real voters to real ballots.


We include a script that automates the creation, training, and evaluation of our GAN, called `run_gan.py`.
This script allows you to configure various hyperparameters (such as the number of random values to be fed to the generator) from a command-line interface.
After training for a number of epochs, the script will save--and display--a number of images created by the network.

A sample invocation of the script is `python run_gan.py --outdir "temp/gan" --gen-count 10`.
A `--help` flag is provided to describe hyperparameters of the network.
If you wish to change the neural network configraton (i.e. changing the number of neurons in a layer), see lines `48-61` of the script.

The images being produced by the GAN are fairly low quality (see below), which indicates that there in some problem with the network.
These images show the original RGBA outputs in the first column, the image as grayscale in the second column, the image's alpha channel rendered as greyscale in the third, and a subtractive combination of the grayscale and alpha values in the forth channel.
The second row uses a traditional edge-finding algorithm on the image above it.
We discuss the poor performance of our network as well as methods for debugging in the next section.

![Output from Gan](/docs/gan_output.png)



### Debugging the GANs (WIP)
Since our GAN is not producing high quality images
* Insufficient training data and few epochs. Even at 400 epochs, both networks are only presented with 24,000 images in total. NN's typically see 100k's of images if not more. For example, the MNIST dataset has 10,000 images presented each epoch compared to our 60 images.
* Poor input normalization.  When we map pixels from [0,255] to [-1.0,+1.0], we may be creating a dataset with mean≉0 and stdev≉1.
* Poor output normalization. Since our data is mostly black / white, using tanh to normalize outputs of the NN may mostly yield shades of gray.
* PIL image conversion issues. We may have a conversion error between float32 outputs of the neural network and the uint8's expected by an image.


One method of debugging a GAN is to create an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder).
An auto-encoder uses two neural networks.
The first network, called an encoder, uses successive layers that compress an input.
The compressed inputs are fed into a second decoder network that expands the compressed input back to its original size.
The goal of both networks is to learn a configuration such that the input to the first network is identical to the output of the second network.

The generator portion of a GAN takes in a small number of input values and expands them into a full-size image.
This behaves very much like a decoding network.
So, to create a true auto-encoder, it is only necessary to supply the encoding portion of a network and a training function that properly optimizes the network.
Since auto-encoders are fairly well understood, if the entire GAN is misbehaving, isolating the generator from the discriminator aid debugging of existing components.
If the auto-encoder doesn't work given a particular configuration, then it is likely that the generator needs to be tuned further.
If the auto-encoder works correctly, then the discriminator needs further tuning.

We provide a script that accomplishes these tasks, called `run_autoencoder.py`.
This script allows configuration of various hyperparameters from command line, and trains the auto-encoding network for a number of epochs.
After training, a batch of original images are written to the output directory.
In addition, the original images are run through the network and the resulting images are written to the disk.

A sample invocation of the script is `python run_autoencoder.py --outdir "temp/encode"`.
A `--help` flag is provided to describe hyperparameters of the network.
If you wish to change the neural network configraton (i.e. changing the number of neurons in a layer), see lines `44-71` of the script.



## Creating Marked Ballots
We include a script, `create_marked_ballots.py`, which allows you to create a large number of marked ballots at a time.
A sample invocation would be `python create_marked_ballots.py --outdir "temp/mark_test" --include-orgegon --count 20 --dpi 40`.
If you would like to change the kinds of marks being applied to ballots, you will need to go to lines `37-46` of the script, and directions are included for chaning the applied markes.
Call the script with a `--help` flag to be given a description of how each of the flags works.

For now, we are targeting built-in ballot types (like the Oregon and Montanna ballots) as a proof of concept, but we hope to extend the system to handle novel ballots with minimal amounts of Python coding required.
A marked-up sample ballot is shown below.
![A marked up ballot generated by our tool](/docs/sample_ballot.jpeg)

# Recognizing Outcomes of Contests
The goal of our project is to train a contest evaluation [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) to evaluate how marked ballots.

## Recognizing Synthetic ballots.
The overall archiecture of the contest evaluation neural network is shown below.
![NN architecture for ballot recognition](/docs/conv-net.png)

Multiple differnt ballot types may be fed into the neural network at once.
Please note that as the number of ballot types evaluated by the network increase, the size of the embedding tables and number of neurons will likely need to increase.

The output layers may be designed so that all of the weights are shared between all (ballot, contest) pairs. 
This configuration uses far less memory (no redundant neural nets), but requires more epochs to train than a unshared configuration.
It is possible to have unique fully-connected and output layers for each (ballot, contest) pair, using the `--unique-outputs` flag.
This flag reduces weight sharing between output layers, and should be faster to train since weights aren't being "fought over" by multiple contest recognizers.
In practice, the shared configuration trains faster, since there are fewer parameters being updated each epoch.
Both configurations asymptotically approach 100% accuracy with the current data.


Here are two sample invocations:
* First, execute `python create_marked_ballots.py --outdir "temp/mark_test" --include-orgegon --count 20 --dpi 40` to create a dataset. Then, execute this script with `python run_recognizer.py --data-dir "temp/mark_test"`.
* Alternatively, you may create an in-memory dataset that is deleted after the program closes. A sample invocation in this mode is `python run_recognizer.py --include-oregon --ballot-count 20 --ballot-dpi 40`.
Call the script with a `--help` flag to be given a description of how each of the flags works.
If you wish to change the neural network configraton (i.e. changing the number of neurons in each layer), see lines `41-51` of the script.

For datasets with many kinds of marks and a wide variety in ballots, 10's to 100's of epochs may be required to train to high accuracy.
In practice, with 3 kinds of marks, in a shared configuration, we are capable of 98% accuracy.

If training time is slow, consider adding the `--aggressive-crop` flag, which will crop contest images to only contain the option bubbles and nothing else.
This will miss marks outside of the option boxes (such as putting a check next to a candidates name to indicate a vote), but training speed will be multiples faster with this flag enabled.

Use of the ballot marking tool speeds up the process of creating marked ballots. Here is an invocation that creates a pickle file from a ballot annotation made by our markup tool and uses it to create 10  marked up ballot images. The file `create_from_file.py` follows exactly the same structure as `create_marked_ballots.py`. The file `full_ballot_annotation.txt` should be created by marking up every contest and option on a ballot.

`python save_contests.py --input full_ballot_annotation.txt --output or_ballot_2_pickle`
`python create_from_file.py --outdir temp/mark_test --ballot or_ballot_2_pickle --count 20` 

Future work includes creating an end-to-end pipeline for automatically detecting contests and options in the ballot images. The following is a working example of using the ballot marking tool to create training data for the FRCNN object detection model (COCO files and a directory of images) as well as a directory of evaluation contests. 

	`annotations.txt` contains data for all contests on two ballots.
	`test_ann.txt` contains data for the contests that contain options (all but 2 contests have been removed).

The following lines can be executed from the `fcrnn/` directory to populate `contests/` with every contest from the ballots to evaluate after training and `../test2/contests/` with the two contests to use during training. The model should be capable of detecting option bubbles in all images in `contests/`. Evaluated images with red bounding boxes will be output to `../test2/eval/`

`python make_coco.py --input ../annotation.txt --directory test/ --dest2 contests/`
`python make_coco.py --input ../test_ann.txt --directory test/ --dest2 ../test2/contests --coco2 options_coco.json`
`python run.py --train_data_dir ../test2/contests/ --train_coco ../test2/options_coco.json --eval_dir contests --eval_dest ../test2/eval/ --num_epochs 25`

## Recognizing Real Ballots (WIP)
We have not yet been able to recognize real ballots.
We would like to add this ability in the future, but we aren't sure if our synthetic data is of high quality.
Poor data quality limits transfer to real ballots.
Succesfully implementingt the GAN will increase synthetic data quality, and at this point we would be comfortable working with real ballots.

# Unit & Regression Testing
We have included several unit tests and regression tests to verify that our tool works correctly.
These may be executed by `cd`'ing to the root of our project, and executing `python -m unittest discover`.
Testing may take several minutes, but is designed to test every major component of the project.

# Hyperparameter Sweeping
Files beginning with sweep_*.py are meant to perform distributed hyperparameter sweeping for our neural networks.
We accomplish this by using the [ray](https://pypi.org/project/ray/) project.

These files require that you manually install ray by executing `pip install ray`.
We do not include this in our `requirements.txt`, since it may cause pip to fail on platforms without a ray backend.

Hypereparameter configuration is not controlled through a commandline interface, but rather through modifying the sampling code.
