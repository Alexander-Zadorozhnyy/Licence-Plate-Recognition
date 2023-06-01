import logging
import typing

import cv2
from mltu import Image
from mltu.augmentors import Augmentor, randomness_decorator
import numpy as np


class GrayFilter(Augmentor):
    """ Adjust grayscale to image """

    def __init__(
            self,
            log_level: int = logging.INFO,
    ) -> None:
        """ Randomly adjust image brightness

        Args:
            random_chance (float, optional): Chance of applying the augmentor.
            Where 0.0 is never and 1.0 is always. Defaults to 0.5.
            delta (int, optional): Integer value for brightness adjustment. Defaults to 100.
            log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
        """
        super().__init__(1, log_level)

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Adjust grayscale to image

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation if necessary
        """

        img = np.array(image.RGB(), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        image.update(img)

        return image, annotation


class EdgeFilter(Augmentor):
    """ Adjust grayscale to image """

    def __init__(
            self,
            log_level: int = logging.INFO,
    ) -> None:
        """ Randomly adjust image brightness

        Args:
            random_chance (float, optional): Chance of applying the augmentor.
            Where 0.0 is never and 1.0 is always. Defaults to 0.5.
            delta (int, optional): Integer value for brightness adjustment. Defaults to 100.
            log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
        """
        super().__init__(1, log_level)

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Adjust grayscale to image

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation if necessary
        """

        img = np.array(image.RGB(), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.Canny(img_blur, 25, 40)

        image.update(img)

        return image, annotation
