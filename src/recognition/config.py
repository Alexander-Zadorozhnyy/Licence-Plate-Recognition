from datetime import datetime

# Desired image dimensions
WIDTH = 96
HEIGHT = 48
CHANNELS = 1

# Training and validation parameters
BATCH = 128
EPOCH = 100
START_LR = 0.0001

# Licence plate's symbols
SYMBOLS = "0123456789ABEKMHOPCTYX"

# Model path
NAME = f"../pretrained_models/MRNET_150000_{EPOCH}ep_{BATCH}b"
SAVE_MODEL_PATH = f"{NAME}-{'{:%m_%d_%H_%M}'.format(datetime.now())}"
