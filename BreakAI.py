import gym
import json
import tensorflow as tf
from DQN import *
from AtariWrapper import *
import numpy as np
import os

class BreakAI(object):
    def __init__(self, dqn, target_dqn, replay_memory, n_actions=4, frame_height=84, frame_width=84, batch_size=32, 
    sequence_length=4, eps_initial=1, eps_final=0.1, eps_final_frame=0.01, eps_evaluation=0.0,
    eps_annealing_frames=1000000, replay_mem_wait_frames=50000, max_frames=25000000, use_per=False):

        self.n_actions = n_actions
        self.input_shape = (frame_width,frame_height)
        self.sequence_length = sequence_length

        # Memory information
        self.replay_mem_wait_frames = replay_mem_wait_frames
        self.max_frames = max_frames
        self.batch_size = batch_size

        self.replay_memory = replay_memory
        self.use_per = use_per

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_mem_wait_frames
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_mem_wait_frames)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

        # DQN
        self.DQN = dqn
        self.target_dqn = target_dqn

    def get_epsilon(self, frame_number, evaluation=False):
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_mem_wait_frames:
            return self.eps_initial
        elif frame_number >= self.replay_mem_wait_frames and frame_number < self.replay_mem_wait_frames + self.eps_annealing_frames:
            return self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_mem_wait_frames + self.eps_annealing_frames:
            return self.slope_2*frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        eps = self.get_epsilon(frame_number, evaluation)

        # For exploring radomly action. Probe beggning..
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.sequence_length)))[0]
        return q_vals.argmax()

    ########   Remove its not used ##########3
    def get_intermediate_representation(self, state, layer_names=None, stack_state=True):

        # Prepare list of layers
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.DQN.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.DQN.get_layer(name=layer_names).output

        # Model for getting intermediate output
        temp_model = tf.keras.Model(self.DQN.inputs, layers)

        # Stack state 4 times
        if stack_state:
            if len(state.shape) == 2:
                state = state[:, :, np.newaxis]
            state = np.repeat(state, self.sequence_length, axis=2)

        # Put it all together
        return temp_model.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.sequence_length)))


    def update_target_network(self):
        """Update the target Q_value network"""
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, action, frame, reward, game_over, clip_reward=True):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_memory.add_experience(action, frame, reward, game_over)

    def learn(self, batch_size, gamma, frame_number, priority_scale=1.0):
        if self.use_per:
            (states, actions, rewards, new_states, terminal_flags), importance, indices = self.replay_memory.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)
            importance = importance ** (1-self.get_epsilon(frame_number))
        else:
            states, actions, rewards, new_states, terminal_flags = self.replay_memory.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)

        # Main DQN estimates best action in new states
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_values = self.target_dqn.predict(new_states)
        double_q = future_q_values[range(batch_size), arg_q_max]

        # Calculate targets (bellman equation)
        target_q = rewards + (gamma*double_q * (1-terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions, dtype=np.float32)  # using tf.one_hot causes strange errors
            Q_value = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            q_error = Q_value - target_q
            loss = tf.keras.losses.Huber()(target_q, Q_value)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

        gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(gradients, self.DQN.trainable_variables))

        if self.use_per:
            self.replay_memory.set_priorities(indices, q_error)

        return float(loss.numpy()), q_error
    
    def save(self, folder_name, **kwargs):

        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        # Save replay buffer
        self.replay_memory.save(folder_name + '/replay-buffer')

        # Save meta. Is this used? can be removed? 
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_memory.count, 'buff_curr': self.replay_memory.current}, **kwargs}))  # save replay_memory information and any other information

    def load(self, folder_name, load_replay_memory=True):

        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer

        # Load replay buffer
        if load_replay_memory:
            self.replay_memory.load(folder_name + '/replay-buffer')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_memory:
            self.replay_memory.count = meta['buff_count']
            self.replay_memory.current = meta['buff_curr']

        del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        return meta

