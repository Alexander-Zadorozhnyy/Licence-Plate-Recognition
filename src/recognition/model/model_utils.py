import tensorflow as tf
from tensorflow.keras import layers


class LightDepthwiseSeparableConvResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.branch1 = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        self.branch2 = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.conc = layers.Concatenate(axis=-1)
        self.activation = layers.Activation("relu")  # gelu

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    def call(self, inputs):
        x_one = self.branch1(inputs)
        x_one = self.bn1(x_one)
        x_two = self.branch2(inputs)
        x_two = self.bn2(x_two)

        # print("X1: ", x1.shape)
        _x = self.conc([x_one, x_two])
        # x = layers.Reshape((12, 24, 128))(x)
        _x = self.activation(_x)
        return _x


class MultiScaleFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.multiply = layers.Multiply()
        self.dense1 = layers.Dense(filters, activation="relu")
        self.dense2 = layers.Dense(filters * 3, activation="sigmoid")
        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=(1, 1),
                                  padding='same')

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    def call(self, inputs):
        _x = self.global_avg_pool(inputs)

        _x = self.dense1(_x)
        _x = self.dense2(_x)

        _x = self.multiply([inputs, _x])
        _x = self.conv(_x)
        return _x


class PredictionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.avg_pool = layers.MaxPooling2D(pool_size=(1, 4))
        self.d_conv = layers.SeparableConv2D(filters=filters,
                                             kernel_size=(3, 3),
                                             padding='same',
                                             activation="softmax")

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    def call(self, inputs):
        _x = self.avg_pool(inputs)
        _x = self.d_conv(_x)
        return _x


class FeatureFusionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.max_pool = layers.MaxPooling2D(name="fuse_pool", pool_size=(2, 2))
        self.p_two = layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   name="P2")

        self.p_three = layers.Conv2D(filters=filters,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     name="P3")

        self.up_sample = layers.UpSampling2D(size=(2, 2))
        self.p_four = layers.Conv2D(filters=filters,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    name="P4")

        self.conc = layers.Concatenate(axis=3)  # 3
        self.msf = MultiScaleFeatureFusion(filters=128)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    def call(self, l_two, l_three, l_four):
        # l2: MaxPooling and 1x1 convolution
        p_two = self.max_pool(l_two)
        p_two = self.p_two(p_two)  # p2.shape[-1] * 2

        # L3: 1x1 convolution
        p_three = self.p_three(l_three)  # L3.shape[-1]

        # L4: Transposed convolution
        p_four = self.up_sample(l_four)
        p_four = self.p_four(p_four)

        # Stack feature maps
        stack = self.conc([p_two, p_three, p_four])
        _x = self.msf(stack)
        return _x
