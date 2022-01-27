import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import pickle 


class Linear_Layer(tf.keras.layers.Layer):
    def __init__(self, 
                 name='Linear_Layer', 
                 **kwargs):
        super(Linear_Layer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(1,), 
                                 initializer='random_normal',
                                 trainable=True)
    def call(self, inputs):
        return self.b + tf.linalg.matvec(inputs, self.w)

class Linear_Model(tf.keras.Model):
    def __init__(self):
        super(Linear_Model, self).__init__()

    def build(self, input_shape):
        self.input_layer = tf.keras.layers.Masking(mask_value=0.0, 
                                                   input_shape=input_shape[-2:])
        self.linear_layer = Linear_Layer()

    def call(self, inputs):
        x = self.input_layer(inputs)
        return self.linear_layer(x)