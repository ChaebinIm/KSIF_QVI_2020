import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.idxs = np.zeros((self.mem_size, n_actions))
        self.codes = [['']*n_actions]*self.mem_size

    def store_transition(self, state, action, reward, state_, idxs, codes):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action.reshape(-1)
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.idxs[index] = idxs
        self.codes[index] = codes

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)[0]

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        idxs = self.idxs[batch]
        codes = self.codes[batch]

        return states, actions, rewards, states_, idxs, codes