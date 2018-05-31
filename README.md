# Riva tutorial - Introduction to Deep learning and CNNs

This part of the tutorial will introduce a simple CNN to classify the [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Therefore we build a CNN to solve a classification task with 10 classes using the RGB images as direct input features.

## Getting Started

Clone this repository to your local machine with:

```
git clone git@gitlab.phys.ethz.ch:nlang/DL_tutorial.git
```

Download the required data from this link:

> https://drive.google.com/open?id=138Bsd5pa44gpQiyVEQrUCKgUIHRni6TV

Move the two directories `data/` and `pretrained_models_imageNet/` into the `DL_tutorial/` directory.

> DL_tutorial/
> 	data/
> 	pretrained_models_imageNet/


## Prerequisites

The tutorial will use keras with a tensorflow backend. We are going to write and execute the code in a jupyter notebook.

Therefore, we need to install:

* python2 or python3
* jupyter
* tensorflow
* keras

Further we will need the python packages/modules:

* numpy
* matplotlib
* random

Our import section looks like this:

```
import numpy as np
import matplotlib.pyplot as plt
import random

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils
from keras import initializers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
```

## Installing
For windows: Tensorflow only exists for python3

### python with pip
* MacOS: Installation with homebrew follow the instructions from this [link](http://docs.python-guide.org/en/latest/starting/install/osx/)
* Windows: [link](https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation)
* Ubuntu: [link](https://www.rosehosting.com/blog/how-to-install-pip-on-ubuntu-16-04/)

Make sure that pip2 or pip3 is installed. To check open a terminal and type:

```
pip2 help
pip3 help
```

Use pip2 or pip3 instead of pip in the following installations.

### jupyter
Install jupyter with pip [link](https://jupyter.readthedocs.io/en/latest/install.html#id4)

Type the following in your terminal:

```
pip install --upgrade pip
pip install jupyter
```

### tensorflow
Follow the official installation instructions:

[official tensorflow installation site](https://www.tensorflow.org/install/)


#### keras

To install keras on your system without a virtualenv: 

[official keras installation site](https://keras.io/#installation)

```
sudo pip install keras
```

## Verify your installation
open a terminal and go to the location of the file: `installation_check.ipynb`

Then open the jupyter notebook with:

```
jupyter notebook installation_check.ipynb
```

If this does not automatically open a browser showing the notebook, then open a browser (Firefox, Chrome) and type: 

`http://localhost:8889/notebooks/installation_check.ipynb`

Then select the first cell containing the imports and click on the `> Run` Button.
If your installation was successful, the output should be like this:

```
successfully imported
keras version:  2.1.6
```

## Running the code

In your terminal, go to the git repository and open the notebook with this command:

```
jupyter notebook CNN_tutorial.ipynb
```

Follow the instructions in the notebook:

* Run cell by cell individually
* Complete the missing parts of the code
* Answer the questions

In the end: you can export the notebook as an html file: File/Download as/HTML

## Code inspirations

* https://github.com/tgjeon/Keras-Tutorials
* https://github.com/flyyufelix/cnn_finetune


## Authors

* Nico Lang 







