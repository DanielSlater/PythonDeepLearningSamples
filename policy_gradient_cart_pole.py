# note must import tensorflow before gym
from collections import deque

import tensorflow as tf
import gym
import numpy as np

env = gym.make('CartPole-v0')

ACTIONS_COUNT = 2
FUTURE_REWARD_DISCOUNT = 0.9
LEARN_RATE = 0.01
STORE_SCORES_LEN = 100.
GAMES_PER_TRAINING = 3
INPUT_NODES = env.observation_space.shape[0]

HIDDEN_NODES = 20

session = tf.Session()

feed_forward_weights_1 = tf.Variable(tf.truncated_normal([INPUT_NODES, HIDDEN_NODES], stddev=0.01))
feed_forward_bias_1 = tf.Variable(tf.constant(0.0, shape=[HIDDEN_NODES]))

feed_forward_weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_NODES, ACTIONS_COUNT], stddev=0.01))
feed_forward_bias_2 = tf.Variable(tf.constant(0.1, shape=[ACTIONS_COUNT]))

input_placeholder = tf.placeholder("float", [None, INPUT_NODES])
hidden_layer = tf.nn.tanh(tf.matmul(input_placeholder, feed_forward_weights_1) + feed_forward_bias_1)
output_layer = tf.nn.softmax(tf.matmul(hidden_layer, feed_forward_weights_2) + feed_forward_bias_2)

action_placeholder = tf.placeholder("float", [None, ACTIONS_COUNT])
advantage_placeholder = tf.placeholder("float", [None, 1])

policy_gradient = tf.reduce_mean(advantage_placeholder * action_placeholder * tf.log(output_layer))
actor_train_operation = tf.train.AdamOptimizer(LEARN_RATE).minimize(-policy_gradient)

scores = deque(maxlen=STORE_SCORES_LEN)

# set the first action to do nothing
last_action = np.zeros(ACTIONS_COUNT)
last_action[1] = 1

time = 0

session.run(tf.initialize_all_variables())


def choose_next_action(state):
    probability_of_actions = session.run(output_layer, feed_dict={input_placeholder: [state]})[0]
    try:
        move = np.random.multinomial(1, probability_of_actions)
    except ValueError:
        # sometimes because of rounding errors we end up with probability_of_actions summing to greater than 1.
        # so need to reduce slightly to be a valid value
        move = np.random.multinomial(1, probability_of_actions / (sum(probability_of_actions) + 1e-5))
    return move


def train(states, actions_taken, advantages):
    # learn that these actions in these states lead to this reward
    session.run(actor_train_operation, feed_dict={
        input_placeholder: states,
        action_placeholder: actions_taken,
        advantage_placeholder: advantages})


last_state = env.reset()
total_reward = 0
current_game_observations = []
current_game_rewards = []
current_game_actions = []

episode_observation = []
episode_rewards = []
episode_actions = []
games = 0

while True:
    env.render()
    last_action = choose_next_action(last_state)
    current_state, reward, terminal, info = env.step(np.argmax(last_action))
    total_reward += reward

    if terminal:
        reward = -.1

    current_game_observations.append(last_state)
    current_game_rewards.append(reward)
    current_game_actions.append(last_action)

    if terminal:
        games += 1
        scores.append(total_reward)

        # get temporal difference values
        cumulative_reward = 0
        for i in reversed(range(len(current_game_observations))):
            cumulative_reward = current_game_rewards[i] + FUTURE_REWARD_DISCOUNT * cumulative_reward
            current_game_rewards[i] = [cumulative_reward]

        print("Time: %s reward %s average scores %s" %
              (time, total_reward,
               np.mean(scores)))

        episode_observation.extend(current_game_observations)
        episode_actions.extend(current_game_actions)
        episode_rewards.extend(current_game_rewards)

        total_reward = 0
        current_game_observations = []
        current_game_rewards = []
        current_game_actions = []

        if games % GAMES_PER_TRAINING == 0:
            episode_rewards = np.array(episode_rewards)
            normalized_rewards = episode_rewards - np.mean(episode_rewards)
            normalized_rewards /= np.std(normalized_rewards)

            train(episode_observation, episode_actions, normalized_rewards)

            episode_observation = []
            episode_actions = []
            episode_rewards = []

    time += 1
    # update the old values
    if terminal:
        last_state = env.reset()
    else:
        last_state = current_state
