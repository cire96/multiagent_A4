import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class BreakNet():
    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001,
                 frame_height=84, frame_width=84, n_history_step=4):
        
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.n_history_step = n_history_step
        
        self.input = layers.Input(shape=( self.frame_height, self.frame_width, self.n_history_step), dtype=tf.float32) 
        #tf.placeholder(shape=[None, self.frame_height, self.frame_width, self.agent_history_length], dtype=tf.float32)
        print(self.input)
        # Normalizing the input for memory
        self.inputscaled = self.input/255

        # Convolutional layers
        self.conv1 = layers.Conv2D(
            filters=32, kernel_size=[8, 8], strides=4,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')(self.inputscaled)
        self.conv2 = layers.Conv2D(
            filters=64, kernel_size=[4, 4], strides=2, 
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')(self.conv1)
        self.conv3 = layers.Conv2D(
            filters=64, kernel_size=[3, 3], strides=1, 
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')(self.conv2)
        self.conv4 = layers.Conv2D(
            filters=hidden, kernel_size=[7, 7], strides=1, 
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')(self.conv3)

        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = layers.Flatten()(self.valuestream)
        self.advantagestream = layers.Flatten()(self.advantagestream)

        #Split dense layers
        self.advantage = layers.Dense(
            units=self.n_actions,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), name="advantage")(self.advantagestream)
        self.value = layers.Dense(
            units=1, 
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), name='value')(self.valuestream)

        # Combining value and advantage into Q-values as described above
        #self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.q_values = tf.keras.layers.Add(name='Qvalue')([self.value,tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))])
        self.best_action = tf.argmax(self.q_values, 1)

        # The next lines perform the parameter update. This will be explained in detail later.
        
        # targetQ according to Bellman equation: 
        # Q = r + gamma*max Q', calculated in the function learn()
        
        self.target_q = layers.Input(tensor=self.q_values, dtype=tf.float32)
        #self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = layers.Input(tensor=self.best_action,dtype=tf.int64)
        #self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        # Extract the Q value of the chosen action by multiplying with one hot encoding of the action.
        print("Hej")
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)), axis=1)
        
        # Parameter updates
        #self.loss = tf.reduce_mean(tf.keras.losses.Huber()(self.target_q,self.Q))
        self.model=tf.keras.Model(self.input,self.Q)
        self.model.compile(loss=self.lossfunc, optimizer='adam')
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        #self.update = self.optimizer.minimize(self.lossfunc(),var_list=None)

    def lossfunc(self):
        loss = tf.reduce_mean(tf.keras.losses.Huber()(self.target_q,self.Q))
        return loss