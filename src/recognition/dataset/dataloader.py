import os
from pathlib import Path

from mltu.preprocessors import ImageReader
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen, RandomGaussianBlur

from dataset.filters import GrayFilter, EdgeFilter
from dataset.data_providers import MyDataProvider


class Dataloader:
    def __init__(self, train_path, val_path, symbols, augment=False):
        self.train_path = train_path
        self.val_path = val_path
        self.augment = augment
        self.characters = sorted(list(symbols))
        self._data = {}
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

        print(f"Max/Min length of labels: {'/'.join(*self.get_max_min_length_of_labels())}")

    def apply_augment(self):
        # Augment training data with random brightness, rotation and erode/dilate, gaussian blur
        self._train_data_loader.augmentors = ([
                                                 RandomBrightness(),
                                                 RandomErodeDilate(),
                                                 RandomSharpen(),
                                                 RandomGaussianBlur()
                                             ] if self.augment else []) + [
                                                 GrayFilter(),
                                                 EdgeFilter(),
                                             ]
        self._valid_data_loader.augmentors = [
                                                 GrayFilter(),
                                                 EdgeFilter(),
                                             ]

    # def apply_grayscale(self):
    #     self._train_data_loader.augmentors += [
    #
    #     ]
    #
    #     self._valid_data_loader.augmentors += [
    #         GrayFilter(),
    #     ]
    #
    # def apply_canny(self):
    #     self._train_data_loader.augmentors += [
    #
    #     ]
    #
    #     self._valid_data_loader.augmentors += [
    #         EdgeFilter(),
    #     ]

    def get_max_min_length_of_labels(self):
        # Maximum length of any captcha in the dataset
        max_length = max(len(label) for label in self._data['y_train'] + self._data['y_valid'])
        min_length = min(len(label) for label in self._data['y_train'] + self._data['y_valid'])
        return max_length, min_length

    def create_dataset(self, dir_type):
        # Get dataset of all the images and labels in train folder
        images = []
        annotations = []
        ann_dir = Path(os.path.join(os.getcwd(),
                                    self.train_path, "ann")) if dir_type == "train" \
            else Path(os.path.join(os.getcwd(), self.val_path, "ann"))
        for i, js_file in enumerate(list(map(str, list(ann_dir.glob("*.txt"))))):
            with open(js_file, "r") as file:
                label = file.read()
                images.append(js_file.replace("ann", "img").replace(".txt", ".png"))
                annotations.append(label)
            if i % 2500 == 0:
                print(f"Data loaded: {i}")
        return images, annotations
        # y_train = [(img.split(os.path.sep)[-1].split(".png")[0]) for img in x_train]
        # train_dataset = [[x_train[k], y_train[k]] for k in range(len(x_train))]
        # self._data['x_train'] = x_train
        # self._data['y_train'] = y_train

    def build(self, image_width, image_height, batch):
        # from pathlib import Path
        # Path to the data directories

        # train_dir = Path(os.path.join(os.getcwd(), self.train_path, "img"))  #
        # val_dir = Path(os.path.join(os.getcwd(), self.val_path, "img"))  #
        # #
        # # print(self.create_dataset("train"))
        # #
        # # # Get dataset of all the images and labels in train folder
        # x_train = sorted(list(map(str, list(train_dir.glob("*.png")))))
        # y_train = [(img.split(os.path.sep)[-1].split(".png")[0]) for img in x_train]
        # train_dataset = [[x_train[k], y_train[k]] for k in range(len(x_train))]
        # self._data['x_train'] = x_train
        # self._data['y_train'] = y_train
        # print(x_train, y_train, sep="\n\n\n\n")
        images, annotations = self.create_dataset("train")
        # print(images, annotations, sep="\n\n\n\n")
        train_dataset = [[images[k], annotations[k]] for k in range(len(images))]
        self._data['x_train'] = images
        self._data['y_train'] = annotations

        # Get dataset of all the images and labels in valid folder
        # x_valid = sorted(list(map(str, list(val_dir.glob("*.png")))))
        # y_valid = [(img.split(os.path.sep)[-1].split(".png")[0]) for img in x_valid]
        # val_dataset = [[x_valid[k], y_valid[k]] for k in range(len(x_valid))]
        # self._data['x_valid'] = x_valid
        # self._data['y_valid'] = y_valid

        images, annotations = self.create_dataset("valid")
        val_dataset = [[images[k], annotations[k]] for k in range(len(images))]
        self._data['x_valid'] = images
        self._data['y_valid'] = annotations

        # Create a data provider for the dataset
        self._train_data_loader = MyDataProvider(
            dataset=train_dataset,
            shuffle=True,
            skip_validation=True,
            batch_size=batch // 2,
            use_cache=True,
            data_preprocessors=[ImageReader()],
            transformers=[
                ImageResizer(image_width, image_height, keep_aspect_ratio=True),
                LabelIndexer(self.characters),
                LabelPadding(max_word_length=self.get_max_min_length_of_labels()[0],
                             padding_value=self.get_num_characters()),
            ],
        )
        # train_data_provider, val_data_provider = data_provider.split(split=0.9)

        # Create a data provider for the dataset
        self._valid_data_loader = DataProvider(
            dataset=val_dataset,
            skip_validation=True,
            batch_size=batch,
            use_cache=True,
            data_preprocessors=[ImageReader()],
            transformers=[
                ImageResizer(image_width, image_height, keep_aspect_ratio=True),
                LabelIndexer(self.characters),
                LabelPadding(max_word_length=self.get_max_min_length_of_labels()[0],
                             padding_value=self.get_num_characters()),
            ],
        )

        self.apply_augment()

    def get_train_dataloader(self):
        if self._train_data_loader is None:
            raise RuntimeError("Dataloader is not initialized!")

        return self._train_data_loader

    def get_valid_dataloader(self):
        if self._train_data_loader is None:
            raise RuntimeError("Dataloader is not initialized!")

        return self._valid_data_loader
