import gym
import tensorflow as tf

import numpy as np

class AtariWrapper():
    def __init__(self,envID,no_op_steps=10,n_history_step=4):
        self.env = gym.make(envID)
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.n_history_step = n_history_step

    def reset(self, evaluation=False):
        observation = self.env.reset()
        self.last_lives = 0
        simulate_loss=True
        if evaluation:
            for i in range(np.random.randint(1, self.no_op_steps)):
                self.env.step(1) # Action 'Fire'
        processedObs = self.ImageProcessor(observation)  
        self.state = np.repeat(processedObs, self.n_history_step, axis=2)
        
        return simulate_loss

    def step(self,action):
        new_observation, reward, real_loss, info = self.env.step(action)
        reward=self.clip_reward(reward)

        if info['ale.lives'] < self.last_lives:
            simulate_loss = True
        else:
            simulate_loss = real_loss
        self.last_lives = info['ale.lives']

        processedObs=self.ImageProcessor(new_observation)
        self.state = np.append(self.state[:, :, 1:], processedObs, axis=2)  

        return processedObs, reward, real_loss, simulate_loss

    def ImageProcessor(self, AleImage):
        #frame dimensions same as used in deepmind
        frame_height = 84
        frame_width = 84

        #grayScale
        grayImage = tf.image.rgb_to_grayscale(AleImage)
        #crop away unused part of the frame
        grayImage = tf.image.crop_to_bounding_box(grayImage, 34, 0, 160, 160)
        #downsize frame to 84x84
        grayImage = tf.image.resize(grayImage,[frame_height, frame_width],  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return grayImage
    
    # clip reward add this to atari wrapper(why not)
    def clip_reward(self,reward):
        if reward > 0:
            return 1
        elif reward == 0:
            return 0
        else:
            return -1
