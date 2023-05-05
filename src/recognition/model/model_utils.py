import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


class LightDepthwiseSeparableConvResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(LightDepthwiseSeparableConvResidualBlock, self).__init__(**kwargs)
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
        x1 = self.branch1(inputs)
        x1 = self.bn1(x1)
        x2 = self.branch2(inputs)
        x2 = self.bn2(x2)

        # print("X1: ", x1.shape)
        x = self.conc([x1, x2])
        # x = layers.Reshape((12, 24, 128))(x)
        x = self.activation(x)
        return x


class MultiScaleFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(MultiScaleFeatureFusion, self).__init__(**kwargs)
        self.filters = filters
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.multiply = layers.Multiply()
        self.dense1 = layers.Dense(filters, activation="relu")
        self.dense2 = layers.Dense(filters * 3, activation="sigmoid")
        self.conv = layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same')

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    def call(self, inputs):
        x = self.global_avg_pool(inputs)

        x = self.dense1(x)
        x = self.dense2(x)

        x = self.multiply([inputs, x])
        x = self.conv(x)
        return x


class PredictionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(PredictionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.avg_pool = layers.MaxPooling2D(pool_size=(1, 4))
        self.d_conv = layers.SeparableConv2D(filters=filters, kernel_size=(3, 3), padding='same', activation="softmax")

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.d_conv(x)
        return x


class FeatureFusionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(FeatureFusionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.mp = layers.MaxPooling2D(name="fuse_pool", pool_size=(2, 2))
        self.p2 = layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', name="P2")

        self.p3 = layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', name="P3")

        self.us = layers.UpSampling2D(size=(2, 2))
        self.p4 = layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', name="P4")

        self.conc = layers.Concatenate(axis=3)  # 3
        self.msf = MultiScaleFeatureFusion(filters=128)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    def call(self, l2, l3, l4):
        # l2: MaxPooling and 1x1 convolution
        p2 = self.mp(l2)
        p2 = self.p2(p2)  # p2.shape[-1] * 2

        # L3: 1x1 convolution
        p3 = self.p3(l3)  # L3.shape[-1]

        # L4: Transposed convolution
        p4 = self.us(l4)
        p4 = self.p4(p4)

        # Stack feature maps
        p = self.conc([p2, p3, p4])
        x = self.msf(p)
        return x
