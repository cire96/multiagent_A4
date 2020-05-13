import gym
import tensorflow as tf

import numpy as np

class AtariWrapper():
    def __init__(self,envID,wait_steps=10,sequence_length=4):
        self.env = gym.make(envID)
        self.state = None
        self.last_lives = 0
        self.wait_steps = wait_steps
        self.sequence_length = sequence_length

    def reset(self, evaluation=False):
        observation = self.env.reset()
        self.last_lives = 0
        simulate_game_over=True
        if evaluation:
            for i in range(np.random.randint(1, self.wait_steps)):
                self.env.step(1) # Action 'Fire'
        processedObs = self.ImageProcessor(observation)  
        self.state = np.repeat(processedObs, self.sequence_length, axis=2)
        
        return simulate_game_over

    def step(self,action,render=False):
        new_observation, reward, game_over, info = self.env.step(action)
        reward=self.clip_reward(reward)

        if info['ale.lives'] < self.last_lives:
            simulate_game_over = True
        else:
            simulate_game_over = game_over
        self.last_lives = info['ale.lives']

        processedObs=self.ImageProcessor(new_observation)
        self.state = np.append(self.state[:, :, 1:], processedObs, axis=2)  
        if(render):
            self.env.render()

        return processedObs, reward, game_over, simulate_game_over

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
        return np.sign(reward)
