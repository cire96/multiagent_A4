import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
#https://github.com/EvolvedSquid/tutorials/tree/master/dqn
def buildDQN(n_actions=4, learning_rate=0.00001, frame_height=84, frame_width=84, sequence_length=4):
    input_tensor=Input(shape=(frame_height, frame_width, sequence_length), dtype=tf.float32) 

    input_scaled = Lambda(lambda input_unscaled: input_unscaled / 255)(input_tensor)  # normalize by 255

    conv1 = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), 
    activation='relu', use_bias=False,name='conv1')(input_scaled)
    conv2 = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), 
    activation='relu', use_bias=False,name='conv2')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), 
    activation='relu', use_bias=False,name='conv3')(conv2)
    conv4 = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), 
    activation='relu', use_bias=False,name='conv4')(conv3)

    value_stream, advantage_stream = Lambda(lambda w: tf.split(w, 2, 3))(conv4)

    value_stream = Flatten(name="value_stream_flatten")(value_stream)
    value = Dense(1, kernel_initializer=VarianceScaling(scale=2.),name="value_stream_dense")(value_stream)

    advantage_stream = Flatten(name="advantage_stream_flatten")(advantage_stream)
    advantage = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.),name="advantage_stream_dense")(advantage_stream)

    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True),name="reduce_mean")

    q_values = Add(name="q_values")([value, Subtract()([advantage, reduce_mean(advantage)])])

    model = Model(input_tensor, q_values)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model 


