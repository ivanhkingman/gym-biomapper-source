import numpy as np
import tensorflow.keras.backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
from cv2 import cv2
from gym import spaces
import matplotlib.pyplot as plt


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 10

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# Agent class
class DQNAgent:
    def __init__(self, env, input, pos0=None, speed=1.0, square_size=50, sample_time=120, n_auvs=1, render=True):
        if render:
            self.fig, self.ax = plt.subplots()
        self.g_pos = np.zeros(3)
        self.input = input
        
        # Main model
        self.model = self.create_model(env)

        # Target network
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        
    # Andreas' Agent
    def deliberate(self, obs):
        if self.input == 'dict':
            pos = obs['pos'][0]
            env = obs["env"]
        if self.input == 'flatten':
            pos = obs[-3:]
            env = obs[:-3].reshape(50, 50)
        if self.input == '3D-matrix':
            pos = np.zeros(3)
            pos[:2] = np.array(np.nonzero(obs[:, :, 1])).flatten()
            env = obs[:, :, 0]
        ind = np.unravel_index(np.argmax(env), env.shape)
        g_pos = np.array([ind[0], ind[1], 0])
        action = [0, 0, 0]
        for i in range(2):
            if pos[i] < g_pos[i]:
                action[i] = 1
            if pos[i] > g_pos[i]:
                action[i] = -1

        actions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
                  [-1, 0, 0], [-1, -1, 0], [0, -1, 0], [1, -1, 0]]
        index = actions.index(action)
        self.g_pos = g_pos
        return [index]

    def render(self, obs, next_pos):
        self.ax.clear()
        self.ax.set_title('Observation')
        if self.input == 'dict':
            self.ax.pcolormesh(obs["env"].T,
                               cmap=cmocean.cm.deep,
                               shading='auto')
            self.ax.plot(self.g_pos[0], self.g_pos[1], 'go')
        if self.input == '3D-matrix':
            self.ax.pcolormesh(obs[:, :, 1].T)
            self.ax.plot(self.g_pos[0], self.g_pos[1], 'go')
        self.fig.canvas.draw()

    def create_model(self, env):
        NB_FILTER = 64  # Number of convolutional filters to use
        NB_ROWS = 3     # Number of rows to use in kernel
        NB_COLS = 3     # Number of cols to use in kerlen


        model = Sequential()

        model.add(Conv2D(NB_FILTER, (NB_ROWS, NB_COLS), input_shape=env.STATE_SPACE_SHAPE))  # = (50, 50, 2) Plankton distribution
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(NB_FILTER, (NB_ROWS, NB_COLS)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.action_space.nvec[0], activation='linear'))  # env.action_space.nvec[0] = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        #self.writer = tf.summary.FileWriter(self.log_dir) // 
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
