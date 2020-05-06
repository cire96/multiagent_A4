import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
#https://github.com/EvolvedSquid/tutorials/tree/master/dqn
def buildDQN(n_actions, learning_rate=0.00001, frame_height=84, frame_width=84, history_length=4):