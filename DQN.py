import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
#https://github.com/EvolvedSquid/tutorials/tree/master/dqn
def buildDQN(n_actions=4, learning_rate=0.00001, frame_height=84, frame_width=84, history_length=4):
    inputTensor=Input(shape=(frame_height, frame_width, history_length), dtype=tf.float32) 

    inputscaled = Lambda(lambda layer: layer / 255)(inputTensor)  # normalize by 255

    conv1 = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), 
    activation='relu', use_bias=False,name='conv1')(inputscaled)
    conv2 = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), 
    activation='relu', use_bias=False,name='conv2')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), 
    activation='relu', use_bias=False,name='conv3')(conv2)
    conv4 = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), 
    activation='relu', use_bias=False,name='conv4')(conv3)

    valuestream, advantagestream = Lambda(lambda w: tf.split(w, 2, 3))(conv4)

    val_stream = Flatten(name="valuestream_flatten")(valuestream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.),name="valuestream_dense")(val_stream)

    adv_stream = Flatten(name="advantagestream_flatten")(advantagestream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.),name="advantagestream_dense")(adv_stream)

    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True),name="reduce_mean")

    q_vals = Add(name="q_values")([val, Subtract()([adv, reduce_mean(adv)])])

    model = Model(inputTensor, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model 

