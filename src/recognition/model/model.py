import os

import tensorflow as tf
from tensorflow.keras import layers, Model

try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU')]
except:
    pass

from model.model_utils import LightDepthwiseSeparableConvResidualBlock, FeatureFusionLayer, PredictionLayer


class MRNET():
    def __init__(self, symbol_count, input_shape=(48, 96, 1)):
        self.symbol_count = symbol_count
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.opt = None

    def build(self):
        input_img = layers.Input(shape=self.input_shape, name="image")

        # 3x3 DSConv
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_img)

        # Layer1: LDWB x1
        x = LightDepthwiseSeparableConvResidualBlock(x.shape[-1], name="LDWBx1")(x)

        # MaxPooling
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer2: LDWB x1
        l2 = LightDepthwiseSeparableConvResidualBlock(x.shape[-1], name="LDWBx2")(x)

        # MaxPooling
        x = layers.MaxPooling2D(pool_size=(2, 2))(l2)

        # Layer3: LDWB x1
        l3 = LightDepthwiseSeparableConvResidualBlock(x.shape[-1], name="LDWBx3")(x)

        # MaxPooling
        x = layers.MaxPooling2D(pool_size=(2, 2))(l3)

        # Layer4: LDWB x1
        l4 = LightDepthwiseSeparableConvResidualBlock(x.shape[-1], name="LDWBx4")(x)

        # Feature fusion
        fused_features = FeatureFusionLayer(filters=128)(l2, l3, l4)

        # Sequence decoder
        decoder_pre_result = PredictionLayer(self.symbol_count)(fused_features)
        res = layers.MaxPooling2D(pool_size=(1, 3))(decoder_pre_result)
        res = layers.Reshape(target_shape=(24, self.symbol_count))(res)

        # Define the model
        self.model = Model(
            inputs=input_img, outputs=res, name="ocr_model_mrnet"
        )

    def summary(self):
        print(self.model.summary())

    def predict(self, input_img):
        return self.model.predict(input_img)

    def compile(self, opt, loss, metrics=None):
        # Optimizer
        self.opt = opt
        # Compile the model and return
        self.model.compile(run_eagerly=False,
                           optimizer=self.opt,
                           metrics=metrics,
                           loss=loss,
                           )

    def train(self, train_data, validation_data, epochs, callbacks, workers):
        self.history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            workers=workers,
        )

    def save(self, path='trained_models', name='weights'):
        # Create folders for trained models
        if not os.path.isdir(os.path.join(os.getcwd(), path)):
            os.makedirs(os.path.join(os.getcwd(), path))
        if not os.path.isdir(os.path.join(os.getcwd(), path, name)):
            os.makedirs(os.path.join(os.getcwd(), path, name))

        # Save model weights
        self.model.save_weights(os.path.join(os.getcwd(), path, name, 'weights.h5'))
        self.model.save(os.path.join(os.getcwd(), path, name, 'MRNET.h5'))

    def load(self, path='trained_models', name='example'):
        #
        self.model.load_weights(os.path.join(os.getcwd(), path, name, 'weights.h5'))


if __name__ == "__main__":
    model = MRNET(symbol_count=23, input_shape=(48, 96, 3))
    model.build()
    tf.keras.utils.plot_model(model.model, to_file="model.png",
                              show_layer_names=True)
    model.summary()
