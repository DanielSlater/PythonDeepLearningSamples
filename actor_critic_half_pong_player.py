import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from half_pong_player import HalfPongPlayer


class ActorCriticHalfPongPlayer(HalfPongPlayer):
    ACTIONS_COUNT = 3  # number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    CRITIC_MEMORY = 1000.  # time steps to observe before training
    STATE_FRAMES = 3  # number of frames to store in the state
    SAVE_EVERY_X_STEPS = 5000
    LEARN_RATE_ACTOR = 1e-6
    LEARN_RATE_CRITIC = 1e-6

    SCREEN_WIDTH = 40
    SCREEN_HEIGHT = 40

    STORE_SCORES_LEN = 1000
    GAMES_PER_TRAINING = 5

    def __init__(self, checkpoint_path="actor_critic_half_pong_1", playback_mode=True, verbose_logging=True):
        """
        Example of deep q network for pong

        :param checkpoint_path: directory to store checkpoints in
        :type checkpoint_path: str
        :param playback_mode: if true games runs in real time mode and demos itself running
        :type playback_mode: bool
        :param verbose_logging: If true then extra log information is printed to std out
        :type verbose_logging: bool
        """
        self._playback_mode = playback_mode
        super(ActorCriticHalfPongPlayer, self).__init__(force_game_fps=8, run_real_time=playback_mode)
        self.verbose_logging = verbose_logging
        self._checkpoint_path = checkpoint_path
        self._session = tf.Session()

        self._input_layer = tf.placeholder("float", [None, self.SCREEN_WIDTH, self.SCREEN_HEIGHT,
                                               self.STATE_FRAMES])
        actor_hidden_activation, actor_regularizer = self._create_network(self._input_layer)

        feed_forward_weights_actor = tf.Variable(tf.truncated_normal([256, self.ACTIONS_COUNT], stddev=0.01))
        feed_forward_bias_actor = tf.Variable(tf.constant(0.01, shape=[self.ACTIONS_COUNT]))

        self._actor_output_layer = tf.nn.softmax(tf.matmul(actor_hidden_activation, feed_forward_weights_actor) + feed_forward_bias_actor)

        self._actor_action_placeholder = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._actor_advantage_placeholder = tf.placeholder("float", [None, 1])

        policy_gradient = tf.reduce_mean(self._actor_advantage_placeholder * self._actor_action_placeholder * tf.log(self._actor_output_layer))
        self._actor_train_operation = tf.train.AdamOptimizer(self.LEARN_RATE_ACTOR).minimize(-policy_gradient)

        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._critic_target_placeholder = tf.placeholder("float", [None, 1])

        critic_hidden_activation, critic_regularizer = self._create_network(self._input_layer)

        feed_forward_weights_critic = tf.Variable(tf.truncated_normal([256, 1], stddev=0.01))
        feed_forward_bias_critic = tf.Variable(tf.constant(0.01, shape=[1]))

        self._critic_output_layer = tf.matmul(critic_hidden_activation, feed_forward_weights_critic) + feed_forward_bias_critic

        self._critic_cost = tf.reduce_mean(tf.square(self._critic_target_placeholder - self._critic_output_layer))
        self._critic_train_operation = tf.train.AdamOptimizer(self.LEARN_RATE_CRITIC).minimize(self._critic_cost)

        self._critic_advantages = self._critic_target_placeholder - self._critic_output_layer

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[1] = 1

        self._last_state = self._empty_iamge()
        self._time = 0

        self._total_reward = 0
        self._current_game_observations = []
        self._current_game_rewards = []
        self._current_game_actions = []

        self._episode_observation = []
        self._episode_rewards = []
        self._episode_actions = []
        self._games = 0
        self._scores = deque(maxlen=self.STORE_SCORES_LEN)

        self._critic_costs = deque(maxlen=self.CRITIC_MEMORY)

        self._session.run(tf.initialize_all_variables())

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)
        self._saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        elif playback_mode:
            raise Exception("Could not load checkpoints for playback")

    def _empty_iamge(self):
        return np.zeros((self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.STATE_FRAMES), dtype=np.float32)

    def get_keys_pressed(self, screen_array, reward, terminal):
        # images will be black or white
        ret, binary_image = cv2.threshold(cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY), 1, 255,
                                          cv2.THRESH_BINARY)

        binary_image = np.reshape(binary_image,
                                  (self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 1))
        current_state = np.append(self._last_state[:, :, 1:], binary_image, axis=2)

        next_action = self._choose_next_action(current_state)
        #current_state, reward, terminal, info = env.step(np.argmax(last_action))
        self._total_reward += reward

        if terminal:
            reward = -.10

        self._current_game_observations.append(self._last_state)
        self._current_game_rewards.append(reward)
        self._current_game_actions.append(self._last_action)

        if terminal:
            self._games += 1
            self._scores.append(self._total_reward)

            # get temporal difference values for critic
            cumulative_reward = 0
            for i in reversed(range(len(self._current_game_observations))):
                cumulative_reward = self._current_game_rewards[i] + self.FUTURE_REWARD_DISCOUNT * cumulative_reward
                self._current_game_rewards[i] = [cumulative_reward]

            _, cost, advantages = self._session.run([self._critic_train_operation, self._critic_cost, self._critic_advantages], {
                self._input_layer: self._current_game_observations,
                self._critic_target_placeholder: self._current_game_rewards})

            self._critic_costs.append(cost)

            print("Game: %s reward %s average scores %s critic cost %s" %
                  (self._games, self._total_reward,
                   np.mean(self._scores), np.mean(self._critic_costs)))

            self._episode_observation.extend(self._current_game_observations)
            self._episode_actions.extend(self._current_game_actions)
            self._episode_rewards.extend(advantages)

            self._total_reward = 0
            self._current_game_observations = []
            self._current_game_rewards = []
            self._current_game_actions = []

            if self._games % self.GAMES_PER_TRAINING == 0 and not self._playback_mode:
                self._train(self._episode_observation, self._episode_actions, self._episode_rewards)

                self._episode_observation = []
                self._episode_actions = []
                self._episode_rewards = []

        self._time += 1

        # update the old values
        if terminal:
            self._last_state = self._empty_iamge()
        else:
            self._last_state = current_state

        self._last_action = next_action

        # save checkpoints for later
        if self._time % self.SAVE_EVERY_X_STEPS == 0:
            self._saver.save(self._session, self._checkpoint_path + '/network', global_step=self._time)

        return HalfPongPlayer.action_index_to_key(np.argmax(next_action))

    def _choose_next_action(self, state):
        probability_of_actions = self._session.run(self._actor_output_layer, feed_dict={self._input_layer: [state]})[0]
        try:
            move = np.random.multinomial(1, probability_of_actions)
        except ValueError:
            # sometimes because of rounding errors we end up with probability_of_actions summing to greater than 1.
            # so need to reduce slightly to be a valid value
            move = np.random.multinomial(1, probability_of_actions / (sum(probability_of_actions) + 1e-6))
        return move

    def _train(self, states, actions_taken, advantages):
        self._session.run(self._actor_train_operation, feed_dict={
            self._input_layer: states,
            self._actor_action_placeholder: actions_taken,
            self._actor_advantage_placeholder: advantages})

    def _create_network(self, input_layer):
        # network weights
        convolution_weights_1 = tf.Variable(tf.truncated_normal([4, 4, self.STATE_FRAMES, 32], stddev=0.01))
        convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

        convolution_weights_2 = tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev=0.01))
        convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

        feed_forward_weights_1 = tf.Variable(tf.truncated_normal([1600, 256], stddev=0.01))
        feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

        feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[256]))

        hidden_convolutional_layer_1 = tf.nn.conv2d(input_layer, convolution_weights_1, strides=[1, 2, 2, 1],
                                                    padding="SAME") + convolution_bias_1

        hidden_max_pooling_layer_1 = tf.nn.relu(tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                                               strides=[1, 2, 2, 1], padding="SAME"))

        hidden_convolutional_layer_2 = tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2,
                                                    strides=[1, 1, 1, 1],
                                                    padding="SAME") + convolution_bias_2

        hidden_max_pooling_layer_2 = tf.nn.relu(
            tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding="SAME"))

        hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_2, [-1, 1600])

        hidden_flat_layer = tf.nn.tanh(
            tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

        final_hidden_activations = tf.nn.tanh(
            tf.matmul(hidden_flat_layer, feed_forward_weights_2) + feed_forward_bias_2)

        regularizer = tf.add_n([tf.nn.l2_loss(x) for x in (convolution_weights_1, convolution_weights_2,
                                                           feed_forward_weights_1,
                                                           feed_forward_weights_2)])

        return final_hidden_activations, regularizer


if __name__ == '__main__':
    player = ActorCriticHalfPongPlayer()
    player.start()
