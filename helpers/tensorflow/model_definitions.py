import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import pickle 

'''
EXISTING LAYERS AND MODELS
'''

def get_GRU_classifier_model(X, args, 
                             regularizer,
                             mask_value=0.0, 
                             return_sequences=True,
                             masking=True): # =tf.keras.regularizers.l2(l2=args.l2),
    '''
    functional API design of a basic RNN based classifier
    '''
    # tf.random.set_seed(args.SEED)

    # regularizer = tf.keras.regularizers.l2(l2=args.l2) 
    # optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # input layers
    inputs = tf.keras.Input(
        shape=(None, X.shape[-1]), 
        name='input')
    if masking==True:
        mask_layer = tf.keras.layers.Masking(
            mask_value=mask_value, 
            input_shape=[None, inputs.shape[-1]], 
            name='masking')
        x = mask_layer(inputs)
    else:
        x = inputs

    # hidden layers
    GRU_layer = tf.keras.layers.GRU(
        units=args.num_units, activation='tanh', recurrent_activation='sigmoid',
        use_bias=True, kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros', kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, 
        recurrent_constraint=None, bias_constraint=None,
        dropout=args.dropout, recurrent_dropout=args.dropout, 
        return_sequences=return_sequences, return_state=False,
        go_backwards=False, stateful=False, unroll=False, time_major=False,
        reset_after=True, name='gru') 
    x = GRU_layer(x)

    # output layers
    output_layer = tf.keras.layers.Dense(
        args.num_classes, 
        activation=None, 
        name='output')
    outputs = output_layer(x)

    # temp_layer = tf.keras.layers.TimeDistributed(
    #     tf.keras.layers.Lambda(
    #         lambda x: x / args.temp, 
    #         name='temperature'), 
    #     name='temp_temp')
    # x = temp_layer(x)

    # softmax_layer = tf.keras.layers.TimeDistributed(
    #     tf.keras.layers.Softmax(
    #         axis=-1, 
    #         name='softmax'), 
    #     name='temp_soft')
    # outputs = softmax_layer(x)

    # model 
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='GRU_classifier')
    
    return model

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

# SHAP IS NOT RECOGNIZING THIS DESIGN, DON'T KNOW WHY; IT'S WEIRD!!!!
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