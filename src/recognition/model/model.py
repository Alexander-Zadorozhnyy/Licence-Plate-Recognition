import os

from tensorflow.keras import layers, Model

from model_utils import LightDepthwiseSeparableConvResidualBlock, \
    FeatureFusionLayer, PredictionLayer


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
        _x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_img)

        # Layer1: LDWB x1
        for i in range(1, 3):
            _x = LightDepthwiseSeparableConvResidualBlock(_x.shape[-1],
                                                          name=f"LDWBx1_{i}")(_x)

        # MaxPooling
        _x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(_x)

        # Layer2: LDWB x1
        l_two = LightDepthwiseSeparableConvResidualBlock(_x.shape[-1],
                                                         name="LDWBx2_1")(_x)
        for i in range(2, 5):
            l_two = LightDepthwiseSeparableConvResidualBlock(_x.shape[-1],
                                                             name=f"LDWBx2_{i}")(l_two)

        # MaxPooling
        _x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(l_two)

        # Layer3: LDWB x1
        l_three = LightDepthwiseSeparableConvResidualBlock(_x.shape[-1],
                                                           name="LDWBx3")(_x)
        for i in range(2, 5):
            l_three = LightDepthwiseSeparableConvResidualBlock(_x.shape[-1],
                                                               name=f"LDWBx3_{i}")(l_three)

        # MaxPooling
        _x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(l_three)

        # Layer4: LDWB x1
        l_four = LightDepthwiseSeparableConvResidualBlock(_x.shape[-1],
                                                          name="LDWBx4")(_x)
        for i in range(2, 3):
            l_four = LightDepthwiseSeparableConvResidualBlock(_x.shape[-1],
                                                              name=f"LDWBx4_{i}")(l_four)

        # Feature fusion
        fused_features = FeatureFusionLayer(filters=128)(l_two, l_three, l_four)

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
                           # jit_compile=True,
                           )

    def train(self, train_data, validation_data, epochs, callbacks, workers):
        # ct = train_data.__getitem__(0)
        # print(ct)
        self.history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            workers=workers,
        )

    def save(self, path='trained_models'):
        # Create folders for trained models
        if not os.path.isdir(os.path.join(os.getcwd(), path)):
            os.makedirs(os.path.join(os.getcwd(), path))

        # Save model weights
        self.model.save_weights(os.path.join(os.getcwd(), path, 'weights.h5'))

    def load(self, path='trained_models'):
        _model = os.path.join(path, 'weights.h5')
        self.model.load_weights(_model)


if __name__ == "__main__":
    model = MRNET(symbol_count=23, input_shape=(48, 96, 3))
    model.build()
    # tf.keras.utils.plot_model(model.model, to_file="model.png",
    #                           show_layer_names=True)
    model.summary()
