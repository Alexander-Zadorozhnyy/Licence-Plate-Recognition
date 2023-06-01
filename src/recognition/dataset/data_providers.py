import copy
import logging

import tensorflow as tf
import numpy as np
from mltu.dataProvider import DataProvider

from dataset.filters import GrayFilter

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')


class MyDataProvider(DataProvider, tf.keras.utils.Sequence):
    def process_data(self, batch_data):
        """ Process data batch of data """
        if self._use_cache and batch_data[0] in self._cache:
            data, annotation = copy.deepcopy(self._cache[batch_data[0]])
        else:
            data, annotation = batch_data
            for preprocessor in self._data_preprocessors:
                data, annotation = preprocessor(data, annotation)

            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, marking for removal on epoch end.")
                self._on_epoch_end_remove.append(batch_data)
                return None, None

            if self._use_cache and batch_data[0] not in self._cache:
                self._cache[batch_data[0]] = (copy.deepcopy(data), copy.deepcopy(annotation))

        data_s, annotation_s = copy.deepcopy(data), copy.deepcopy(annotation)

        # Then augment, transform and postprocess the batch data
        for objects in [self._augmentors, self._transformers]:
            for _object in objects:
                data, annotation = _object(data, annotation)

        for _object in self._transformers:
            data_s, annotation_s = _object(data_s, annotation_s)

        to_gray_scale = GrayFilter()
        data_s, annotation_s = to_gray_scale(data_s, annotation_s)

        # Convert to numpy array if not already
        if not isinstance(data, np.ndarray):
            data = data.numpy()

        # Convert to numpy array if not already
        if not isinstance(annotation, (np.ndarray, int, float, str, np.uint8, np.float)):
            annotation = annotation.numpy()

        # Convert to numpy array if not already
        if not isinstance(data_s, np.ndarray):
            data_s = data_s.numpy()

        # Convert to numpy array if not already
        if not isinstance(annotation_s, (np.ndarray, int, float, str, np.uint8, np.float)):
            annotation_s = annotation_s.numpy()

        return [(data_s, annotation_s), (data, annotation)]

    def __getitem__(self, _index: int):
        """ Returns a batch of data by batch index"""
        dataset_batch = self.get_batch_annotations(_index)

        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for _index, batch in enumerate(dataset_batch):
            (data_s, annotation_s), (data, annotation) = self.process_data(batch)

            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, skipping.")
                continue

            batch_data.append(data_s)
            batch_annotations.append(annotation_s)

            batch_data.append(data)
            batch_annotations.append(annotation)

        return np.array(batch_data), np.array(batch_annotations)
