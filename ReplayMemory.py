import numpy as np
import os


class ReplayMemory:
    def __init__(self, size=1000000, frame_height=84, frame_width=84, sequence_length=4, use_per=False):
        self.size = size
        self.sequence_length = sequence_length
        self.count = 0
        self.current = 0
        self.use_per = use_per

        # Memory
        
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, frame_width, frame_height), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

    def add_experience(self, action, frame, reward, terminal):
        
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.frames[self.current, ...] = frame
        self.terminal_flags[self.current] = terminal

        self.priorities[self.current] = max(self.priorities.max(), 1)
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32, priority_scale=0.0):
        if self.use_per:
            scaled_priorities = self.priorities[self.sequence_length:self.count -
                                                1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        indices = []
        for i in range(batch_size):
            not_valid_index=True
            while not_valid_index:
                # Get a random number from sequence_length to maximum frame written with probabilities based on priority weights
                if self.use_per:
                    index = np.random.choice(
                        np.arange(self.sequence_length, self.count-1), p=sample_probabilities)
                else:
                    index = np.random.randint(
                        self.sequence_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.  If either are True, the index is invalid.
                if (index >= self.current and index - self.sequence_length <= self.current) or (self.terminal_flags[index - self.sequence_length:index].any()):
                    continue
                else:
                    not_valid_index=False

            indices.append(index)


        states = []
        new_states = []
        for i in indices:
            states.append(self.frames[i-self.sequence_length:i, ...])
            new_states.append(self.frames[i-self.sequence_length+1:i+1, ...])

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        if self.use_per:
            # Get importance weights from probabilities calculated earlier
            importance = 1/self.count * 1/sample_probabilities[[index - 4 for index in indices]]
            importance = importance / importance.max()

            return (states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]), importance, indices
        else:
            return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def set_priorities(self, indices, errors, offset=0.1):#Remove
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


    def save(self, folder_name):
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')