from datetime import datetime

# Desired image dimensions
WIDTH = 96
HEIGHT = 48
CHANNELS = 3

# Training and validation parameters
BATCH = 128
EPOCH = 1
START_LR = 0.001

# Licence plate's symbols
SYMBOLS = "0123456789ABEKMHOPCTYX"

# Model path
NAME = "../pretrained_models/Example"
SAVE_MODEL_PATH = NAME + "-{:%m_%d_%H_%M}".format(datetime.now())
