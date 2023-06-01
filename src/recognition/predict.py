import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer

from src.recognition.config import SYMBOLS


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.char_list = sorted(list(SYMBOLS))

    def get_list_char(self):
        return self.char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        image_pred = np.expand_dims(image_pred, axis=-1).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]

        return ctc_decoder(preds, self.char_list)[0]
