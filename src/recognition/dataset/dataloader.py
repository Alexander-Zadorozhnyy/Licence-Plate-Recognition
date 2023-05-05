import os

from mltu.tensorflow.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen, RandomGaussianBlur


class Dataloader:
    def __init__(self, train_path, val_path, symbols, augment=False):
        self.train_path = train_path
        self.val_path = val_path
        self.augment = augment
        self.characters = sorted([i for i in symbols])
        self._data = dict()
        self._train_data_loader = None
        self._valid_data_loader = None

    def get_str_characters(self):
        return "".join(self.characters)

    def get_num_characters(self):
        return len(self.characters)

    def print_statistic(self):
        print("Number of train images found: ", len(self._data['x_train']))
        print("Number of train labels found: ", len(self._data['y_train']))

        print("Number of val images found: ", len(self._data['x_valid']))
        print("Number of val labels found: ", len(self._data['y_valid']))

        print("Number of unique characters: ", self.get_num_characters())
        print("Characters present: ", self.characters)

        print("Max/Min length of labels: {0}/{1}".format(*self.get_max_min_length_of_labels()))

    def apply_augment(self):
        # Augment training data with random brightness, rotation and erode/dilate, gaussian blur
        self._train_data_loader.augmentors = [
            RandomBrightness(),
            RandomErodeDilate(),
            RandomSharpen(),
            RandomGaussianBlur(),
        ]

    def get_max_min_length_of_labels(self):
        # Maximum length of any captcha in the dataset
        max_length = max([len(label) for label in self._data['y_train'] + self._data['y_valid']])
        min_length = min([len(label) for label in self._data['y_train'] + self._data['y_valid']])
        return max_length, min_length

    def build(self, image_width, image_height, batch):
        from pathlib import Path
        # Path to the data directories

        train_dir = Path(os.path.join(os.getcwd(), self.train_path))  #
        val_dir = Path(os.path.join(os.getcwd(), self.val_path))  #

        # Get dataset of all the images and labels in train folder
        x_train = sorted(list(map(str, list(train_dir.glob("*.png")))))
        y_train = [(img.split(os.path.sep)[-1].split(".png")[0]) for img in x_train]
        train_dataset = [[x_train[k], y_train[k]] for k in range(len(x_train))]
        self._data['x_train'] = x_train
        self._data['y_train'] = y_train

        # Get dataset of all the images and labels in valid folder
        x_valid = sorted(list(map(str, list(val_dir.glob("*.png")))))
        y_valid = [(img.split(os.path.sep)[-1].split(".png")[0]) for img in x_valid]
        val_dataset = [[x_valid[k], y_valid[k]] for k in range(len(x_valid))]
        self._data['x_valid'] = x_valid
        self._data['y_valid'] = y_valid

        # Create a data provider for the dataset
        self._train_data_loader = DataProvider(
            dataset=train_dataset,
            skip_validation=True,
            batch_size=batch,
            data_preprocessors=[ImageReader()],
            transformers=[
                ImageResizer(image_width, image_height, keep_aspect_ratio=True),
                LabelIndexer(self.characters),
                LabelPadding(max_word_length=self.get_max_min_length_of_labels()[0],
                             padding_value=self.get_num_characters()),
            ],
        )
        # train_data_provider, val_data_provider = data_provider.split(split=0.9)

        if self.augment:
            self.apply_augment()

        # Create a data provider for the dataset
        self._valid_data_loader = DataProvider(
            dataset=val_dataset,
            skip_validation=True,
            batch_size=batch,
            data_preprocessors=[ImageReader()],
            transformers=[
                ImageResizer(image_width, image_height, keep_aspect_ratio=True),
                LabelIndexer(self.characters),
                LabelPadding(max_word_length=self.get_max_min_length_of_labels()[0],
                             padding_value=self.get_num_characters()),
            ],
        )

    def get_train_dataloader(self):
        if self._train_data_loader is None:
            raise RuntimeError("Dataloader is not initialized!")

        return self._train_data_loader

    def get_valid_dataloader(self):
        if self._train_data_loader is None:
            raise RuntimeError("Dataloader is not initialized!")

        return self._valid_data_loader
