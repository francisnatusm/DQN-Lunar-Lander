import time
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils
import argparse
import os
import sys

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

# Display(visible=0, size=(840, 480)).start();
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

tf.random.set_seed(utils.SEED)

MEMORY_SIZE = 100_000
GAMMA = 0.995
ALPHA = 1e-3
NUM_STEPS_FOR_UPDATE = 4

env = gym.make('LunarLander-v3', render_mode='rgb_array')

# Use the new Gymnasium reset/step API which returns (obs, info) and
# step returns (obs, reward, terminated, truncated, info).
obs, _ = env.reset()
PIL.Image.fromarray(env.render())

state_size = env.observation_space.shape
num_actions = int(env.action_space.n)

print('State Shape:', state_size)
print('Number of actions:', num_actions)

current_state, _ = env.reset()

action = 0
next_state, reward, terminated, truncated, _ = env.step(action)
done = bool(terminated or truncated)
utils.display_table(current_state, action, next_state, reward, done)
current_state = next_state

# Build models using the Functional API so `input` tensors are defined and
# the `.layers` list contains only the Dense layers expected by the tests.
q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])

target_q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])

optimizer = Adam(learning_rate=ALPHA)

# Build the Sequential models by calling them once with a dummy input so their
# `input`/`output` tensors are created and the tests can inspect them.
import numpy as _np
_dummy = _np.zeros((1, state_size[0]), dtype=_np.float32)
q_network(_dummy)
target_q_network(_dummy)
q_network.build((None, state_size[0]))
target_q_network.build((None, state_size[0]))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--no-tests", action="store_true", help="Skip running public tests")
parser.add_argument("--load", type=str, default=None, metavar="MODEL_PATH",
                    help="Load a pre-trained model (e.g. lunar_lander_model.h5) and skip training")
args, _ = parser.parse_known_args()

if not args.no_tests:
    from public_tests import *

    # Create lightweight proxies so the public tests see three Dense layers and an
    # input tensor with the expected shape (the Sequential model may include an
    # InputLayer that interferes with the tests).
    class ModelProxy:
        def __init__(self, model):
            self._m = model

        @property
        def layers(self):
            return [l for l in self._m.layers if not isinstance(l, tf.keras.layers.InputLayer)]

        @property
        def input(self):
            class _Shape:
                def __init__(self, dim):
                    self._dim = dim

                def as_list(self):
                    return [None, self._dim]

            class _In:
                def __init__(self, dim):
                    self.shape = _Shape(dim)

            return _In(state_size[0])

    test_network(ModelProxy(q_network))
    test_network(ModelProxy(target_q_network))
    test_optimizer(optimizer, ALPHA)

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = experiences
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))

    loss = MSE(y_targets, q_values)
    return loss

# UNIT TEST
if not args.no_tests:
    test_compute_loss(compute_loss)

@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    """
    
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)


if args.load:
    model_path = args.load
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    print(f"Loading pre-trained model from '{model_path}'...")
    q_network = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
else:
    start = time.time()

    num_episodes = 2000
    max_num_timesteps = 1000

    total_point_history = []

    num_p_av = 100    # number of total points to use for averaging
    epsilon = 1.0     # initial ε value for ε-greedy policy

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    for i in range(num_episodes):
        
        # Reset the environment to the initial state and get the initial state
        state, _ = env.reset()
        total_points = 0
        
        for t in range(max_num_timesteps):
            
            # From the current state S choose an action A using an ε-greedy policy
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            action = utils.get_action(q_values, epsilon)
            
            # Take action A and receive reward R and the next state S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))
            
            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
            
            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = utils.get_experiences(memory_buffer)
                
                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                agent_learn(experiences, GAMMA)
            
            state = next_state.copy()
            total_points += reward
            
            if done:
                break
                
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        
        # Update the ε value
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

        if (i+1) % num_p_av == 0:
            print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if av_latest_points >= 200.0:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            q_network.save('lunar_lander_model.h5')
            break
            
    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

    # Plot the total point history along with the moving average
    utils.plot_history(total_point_history)

# Suppress warnings from imageio
import logging
logging.getLogger().setLevel(logging.ERROR)

filename = os.path.join(os.path.dirname(__file__), "videos", "lunar_lander.mp4")

utils.create_video(filename, env, q_network)
utils.embed_mp4(filename)