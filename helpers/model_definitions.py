import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import pickle 

'''
EXISTING LAYERS AND MODELS
'''

def get_linear_model(X):
    '''
    functional API design of a simple linear regression model
    '''
    # input layer
    inputs = tf.keras.Input(shape=(X.shape[-1]))
    x = tf.keras.layers.Masking(mask_value=0.0, input_shape=inputs.shape[1:])(inputs)

    # linear layer
    outputs = tf.keras.layers.Dense(units=1)(x)

    # model 
    linear_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # compiling 
    linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                         loss=tf.keras.losses.MeanSquaredError())
    # metrics=tfa.metrics.RSquare()

    return linear_model

# SHAP IS NOT RECOGNIZING THIS DESIGN, IT'S WEIRD!!!!
class Linear_Model(tf.keras.Model):
    def __init__(self):
        super(Linear_Model, self).__init__()
        self.supports_masking = True

    def build(self, input_shape):
        self.input_layer = tf.keras.layers.Masking(mask_value=0.0, 
                                                   input_shape=input_shape[-2:])
        # self.linear_layer = Linear_Layer()
        self.linear_layer = tf.keras.layers.Dense(units=1)

    def call(self, inputs, mask=None):
        x = self.input_layer(inputs)
        return tf.squeeze(self.linear_layer(x))


'''
CUSTOM LAYERS AND MODELS
'''


class Linear_Layer(tf.keras.layers.Layer):
    def __init__(self, 
                 name='Linear_Layer', 
                 **kwargs):
        super(Linear_Layer, self).__init__(name=name, **kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(1,), 
                                 initializer='random_normal',
                                 trainable=True)
    def call(self, inputs, mask=None):
        # The mask associated with the inputs will be passed
        # to your layer whenever it is available.
        masked_inputs = inputs #* tf.expand_dims(tf.cast(mask, 'float32'), -1)
        return self.b + tf.linalg.matvec(masked_inputs, self.w)