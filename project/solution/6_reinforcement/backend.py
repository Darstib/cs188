import collections
import math
import os
import os.path
import random
import time
import copy
import weakref
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np

use_graphics = True

def maybe_sleep_and_close(seconds):
    if use_graphics and plt.get_fignums():
        time.sleep(seconds)
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            plt.close(fig)
            try:
                # This raises a TclError on some Windows machines
                fig.canvas.start_event_loop(1e-3)
            except:
                pass

# Stats should include all of the key quantities used for grading.
# This backend file deals with all data loading / environment construction, so
# once a function get_data_and_monitor_* returns the dataset might have been
# thrown away even though the model still exists
all_stats = weakref.WeakKeyDictionary()

def get_stats(model):
    return all_stats.get(model, None)

def set_stats(model, stats_dict):
    all_stats[model] = stats_dict

def get_data_path(filename):
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise Exception("Could not find data file: {}".format(filename))

    return path

def make_get_data_and_monitor_perceptron():
    points = 500

    x = np.hstack([np.random.randn(points, 2), np.ones((points, 1))])
    y = np.where(x[:, 0] + 2 * x[:, 1] - 1 >= 0, 1, -1)

    if use_graphics:
        fig, ax = plt.subplots(1, 1)
        limits = np.array([-3.0, 3.0])
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        positive = ax.scatter(*x[y == 1, :-1].T, color="red", marker="+")
        negative = ax.scatter(*x[y == -1, :-1].T, color="blue", marker="_")
        line, = ax.plot([], [], color="black")
        text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
        ax.legend([positive, negative], [1, -1])
        plt.show(block=False)

    def monitor(perceptron, epoch, point, log):
        w = perceptron.get_weights()

        if log:
            print("epoch {:,} point {:,}/{:,} weights {}".format(
                epoch, point, points, w))

        if use_graphics:
            if w[1] != 0:
                line.set_data(limits, (-w[0] * limits - w[2]) / w[1])
            elif w[0] != 0:
                line.set_data(np.full(2, -w[2] / w[0]), limits)
            else:
                line.set_data([], [])
            text.set_text("epoch: {:,}\npoint: {:,}/{:,}\nweights: {}\n"
                          "showing every {:,} updates".format(
                epoch, point, points, w, min(2 ** (epoch + 1), points)))
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(1e-3)

    # Use a dictionary since the `nonlocal` keyword doesn't exist in Python 2
    nonlocals = {"epoch": 0}
    stats = {}

    def get_data_and_monitor_perceptron(perceptron):
        for i in range(points):
            yield x[i], y[i]
            if i % (2 ** (nonlocals["epoch"] + 1)) == 0:
                monitor(perceptron, nonlocals["epoch"], i, False)

        monitor(perceptron, nonlocals["epoch"], points, True)
        nonlocals["epoch"] += 1

        set_stats(perceptron, stats)
        w = perceptron.get_weights()
        stats['accuracy'] = np.mean(np.where(np.dot(x, w) >= 0, 1, -1) == y)

    return get_data_and_monitor_perceptron

def get_data_and_monitor_regression(model):
    stats = {}
    set_stats(model, stats)

    points = 200
    iterations = 20000

    x = np.linspace(-2 * np.pi, 2 * np.pi, num=points)[:, np.newaxis]
    y = np.sin(x)

    if use_graphics:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-1.4, 1.4)
        real, = ax.plot(x, y, color="blue")
        learned, = ax.plot([], [], color="red")
        text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
        ax.legend([real, learned], ["real", "learned"])
        plt.show(block=False)

    def monitor(iteration, log):
        predicted = model.run(x)
        loss = np.mean(np.square(predicted - y) / 2)
        stats['loss'] = loss

        assert np.allclose(x, -x[::-1,:])
        asymmetry = np.abs(predicted + predicted[::-1])
        stats['max_asymmetry'] = np.max(asymmetry)
        stats['max_asymmetry_x'] = float(x[np.argmax(asymmetry)])

        if log:
            print("iteration {:,}/{:,} loss {:.6f}".format(
                iteration, iterations, loss))

        if use_graphics:
            learned.set_data(x, predicted)
            text.set_text("iteration: {:,}/{:,}\nloss: {:.6f}".format(
                iteration, iterations, loss))
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(1e-3)

    for iteration in range(iterations):
        yield x, y
        if iteration % 20 == 0:
            monitor(iteration, iteration % 1000 == 0)

    monitor(iterations, True)

    if use_graphics:
        plt.close(fig)
        try:
            # This raises a TclError on some Windows machines
            fig.canvas.start_event_loop(1e-3)
        except:
            pass

def get_data_and_monitor_digit_classification(model):
    stats = {}
    set_stats(model, stats)

    epochs = 5
    batch_size = 100

    mnist_path = get_data_path("mnist.npz")

    with np.load(mnist_path) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        dev_images = data["test_images"]
        dev_labels = data["test_labels"]

    num_train = len(train_images)

    train_labels_one_hot = np.zeros((num_train, 10))
    train_labels_one_hot[range(num_train), train_labels] = 1

    if use_graphics:
        width = 20  # Width of each row expressed as a multiple of image width
        samples = 100  # Number of images to display per label
        fig = plt.figure()
        ax = {}
        images = collections.defaultdict(list)
        texts = collections.defaultdict(list)
        for i in reversed(range(10)):
            ax[i] = plt.subplot2grid((30, 1), (3 * i, 0), 2, 1, sharex=ax.get(9))
            plt.setp(ax[i].get_xticklabels(), visible=i == 9)
            ax[i].set_yticks([])
            ax[i].text(-0.03, 0.5, i, transform=ax[i].transAxes, va="center")
            ax[i].set_xlim(0, 28 * width)
            ax[i].set_ylim(0, 28)
            for j in range(samples):
                images[i].append(ax[i].imshow(
                    np.zeros((28, 28)), vmin=0, vmax=1, cmap="Greens", alpha=0.3))
                texts[i].append(ax[i].text(
                    0, 0, "", ha="center", va="top", fontsize="smaller"))
        ax[9].set_xticks(np.linspace(0, 28 * width, 11))
        ax[9].set_xticklabels(np.linspace(0, 1, 11))
        ax[9].tick_params(axis="x", pad=16)
        ax[9].set_xlabel("Probability of Correct Label")
        status = ax[0].text(
            0.5, 1.5, "", transform=ax[0].transAxes, ha="center", va="bottom")
        plt.show(block=False)

    def softmax(x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def monitor(epoch, log):
        dev_logits = model.run(dev_images)
        dev_predicted = np.argmax(dev_logits, axis=1)
        dev_accuracy = np.mean(dev_predicted == dev_labels)
        stats['dev_accuracy'] = dev_accuracy

        if log:
            print("epoch {:.2f}/{:.2f} validation-accuracy {:.2%}".format(
                epoch, epochs, dev_accuracy))

        if use_graphics:
            status.set_text("epoch: {:.2f}/{:.2f}, validation-accuracy: {:.2%}".format(
                epoch, epochs, dev_accuracy))
            dev_probs = softmax(dev_logits)
            for i in range(10):
                predicted = dev_predicted[dev_labels == i]
                probs = dev_probs[dev_labels == i][:, i]
                linspace = np.linspace(0, len(probs) - 1, samples).astype(int)
                indices = probs.argsort()[linspace]
                for j, (prob, image) in enumerate(zip(
                        probs[indices], dev_images[dev_labels == i][indices])):
                    images[i][j].set_data(image.reshape((28, 28)))
                    left = prob * (width - 1) * 28
                    if predicted[indices[j]] == i:
                        images[i][j].set_cmap("Greens")
                        texts[i][j].set_text("")
                    else:
                        images[i][j].set_cmap("Reds")
                        texts[i][j].set_text(predicted[indices[j]])
                        texts[i][j].set_x(left + 14)
                    images[i][j].set_extent([left, left + 28, 0, 28])
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(1e-3)

    for epoch in range(epochs):
        for index in range(0, num_train, batch_size):
            x = train_images[index:index + batch_size]
            y = train_labels_one_hot[index:index + batch_size]
            yield x, y
            if index % 5000 == 0:
                monitor(epoch + 1.0 * index / num_train, index % 15000 == 0)

    monitor(epochs, True)

    if use_graphics:
        plt.close(fig)
        try:
            # This raises a TclError on some Windows machines
            fig.canvas.start_event_loop(1e-3)
        except:
            pass

def get_data_and_monitor_lang_id(model):
    stats = {}
    set_stats(model, stats)

    iterations = 15000
    batch_size = 16

    data_path = get_data_path("lang_id.npz")

    with np.load(data_path) as data:
        chars = data['chars']
        language_codes = data['language_codes']
        language_names = data['language_names']

        train_x = data['train_x']
        train_y = data['train_y']
        train_buckets = data['train_buckets']
        dev_x = data['test_x']
        dev_y = data['test_y']
        dev_buckets = data['test_buckets']

    chars_print = chars
    try:
        print(u"Alphabet: {}".format(u"".join(chars)))
    except UnicodeEncodeError:
        chars_print = "abcdefghijklmnopqrstuvwxyzaaeeeeiinoouuacelnszz"
        print("Alphabet: " + chars_print)
        chars_print = list(chars_print)
        print("""
NOTE: Your terminal does not appear to support printing Unicode characters.
For the purposes of printing to the terminal, some of the letters in the
alphabet above have been substituted with ASCII symbols.""".strip())
    print("")

    num_chars = len(chars)
    num_langs = len(language_names)

    bucket_weights = train_buckets[:,1] - train_buckets[:,0]
    bucket_weights = bucket_weights / float(bucket_weights.sum())

    # Select some examples to spotlight in the monitoring phase (3 per language)
    spotlight_idxs = []
    for i in range(num_langs):
        idxs_lang_i = np.nonzero(dev_y == i)[0]
        idxs_lang_i = np.random.choice(idxs_lang_i, size=3, replace=False)
        spotlight_idxs.extend(list(idxs_lang_i))
    spotlight_idxs = np.array(spotlight_idxs, dtype=int)

    def encode(inp_x, inp_y):
        xs = []
        for i in range(inp_x.shape[1]):
            xs.append(np.eye(num_chars)[inp_x[:,i]])
        y = np.eye(num_langs)[inp_y]
        return xs, y

    def make_templates():
        max_word_len = dev_x.shape[1]
        max_lang_len = max([len(x) for x in language_names])

        predicted_template = u"Pred: {:<NUM}".replace('NUM',
            str(max_lang_len))

        word_template = u"  "
        word_template += u"{:<NUM} ".replace('NUM', str(max_word_len))
        word_template += u"{:<NUM} ({:6.1%})".replace('NUM', str(max_lang_len))
        word_template += u" {:<NUM} ".replace('NUM',
            str(max_lang_len + len('Pred: ')))
        for i in range(num_langs):
            word_template += u"|{}".format(language_codes[i])
            word_template += "{probs[" + str(i) + "]:4.0%}"

        return word_template, predicted_template

    word_template, predicted_template = make_templates()

    def softmax(x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def monitor(iteration):
        all_predicted = []
        all_correct = []
        for bucket_id in range(dev_buckets.shape[0]):
            start, end = dev_buckets[bucket_id]
            xs, y = encode(dev_x[start:end], dev_y[start:end])
            predicted = model.run(xs)

            all_predicted.extend(list(predicted))
            all_correct.extend(list(dev_y[start:end]))

        all_predicted_probs = softmax(np.asarray(all_predicted))
        all_predicted = np.asarray(all_predicted).argmax(axis=-1)
        all_correct = np.asarray(all_correct)

        dev_accuracy = np.mean(all_predicted == all_correct)
        stats['dev_accuracy'] = dev_accuracy

        print("iteration {:,} accuracy {:.1%}".format(
            iteration, dev_accuracy))

        for idx in spotlight_idxs:
            correct = (all_predicted[idx] == all_correct[idx])
            word = u"".join([chars_print[ch] for ch in dev_x[idx] if ch != -1])

            print(word_template.format(
                word,
                language_names[all_correct[idx]],
                all_predicted_probs[idx, all_correct[idx]],
                "" if correct else predicted_template.format(
                    language_names[all_predicted[idx]]),
                probs=all_predicted_probs[idx,:],
            ))
        print("")

    for iteration in range(iterations + 1):
        # Sample a bucket
        bucket_id = np.random.choice(bucket_weights.shape[0], p=bucket_weights)
        example_ids = train_buckets[bucket_id, 0] + np.random.choice(
            train_buckets[bucket_id, 1] - train_buckets[bucket_id, 0],
            size=batch_size)

        yield encode(train_x[example_ids], train_y[example_ids])
        if iteration % 1000 == 0:
            monitor(iteration)

# class CartPoleEnv(object):
#     # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
#     # Licensed under MIT license: https://opensource.org/licenses/MIT
#
#     def __init__(self, theta_threshold_degrees=12, seed=1, max_steps=200):
#         self.gravity = 9.8
#         self.masscart = 1.0
#         self.masspole = 0.1
#         self.total_mass = (self.masspole + self.masscart)
#         self.length = 0.5 # actually half the pole's length
#         self.polemass_length = (self.masspole * self.length)
#         self.force_mag = 10.0
#         self.tau = 0.02 # seconds between state updates
#
#         self.max_steps = max_steps
#
#         # Angle at which to fail the episode
#         self.theta_threshold_degrees = theta_threshold_degrees
#         self.theta_threshold_radians = theta_threshold_degrees * 2 * math.pi / 360
#         self.x_threshold = 2.4
#
#         # Angle limit set to 2 * theta_threshold_radians so failing observation
#         # is still within bounds
#         high = np.array([
#             self.x_threshold * 2,
#             np.finfo(np.float32).max,
#             self.theta_threshold_radians * 2,
#             np.finfo(np.float32).max])
#
#         self.action_space = {0, 1}
#         self.num_actions = len(self.action_space)
#         self.observation_state_size = 2
#
#         self.np_random = np.random.RandomState(seed)
#         self.state = None
#
#         self.steps_taken = 0
#         self.steps_beyond_done = None
#
#     def reset(self):
#         self.steps_taken = 0
#         self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
#         self.steps_beyond_done = None
#         return np.array(self.state)
#
#     def step(self, action):
#         assert action in self.action_space, "%r (%s) invalid" % (action, type(action))
#         state = self.state
#         x, x_dot, theta, theta_dot = state
#         force = self.force_mag if action == 1 else -self.force_mag
#         costheta = math.cos(theta)
#         sintheta = math.sin(theta)
#         temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
#         thetaacc = (self.gravity * sintheta - costheta * temp) / (
#             self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
#         xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
#         x = x + self.tau * x_dot
#         x_dot = x_dot + self.tau * xacc
#         theta = theta + self.tau * theta_dot
#         theta_dot = theta_dot + self.tau * thetaacc
#         self.state = (x, x_dot, theta, theta_dot)
#         done = (
#             x < -self.x_threshold
#             or x > self.x_threshold
#             or theta < -self.theta_threshold_radians
#             or theta > self.theta_threshold_radians)
#         done = bool(done)
#
#         if not done:
#             reward = 1.0
#         elif self.steps_beyond_done is None: # Pole just fell!
#             self.steps_beyond_done = 0
#             reward = 1.0
#         else:
#             if self.steps_beyond_done == 0:
#                 print("You are calling 'step()' even though this environment "
#                       "has already returned done = True. You should always "
#                       "call 'reset()' once you receive 'done = True' -- any "
#                       "further steps are undefined behavior.")
#             self.steps_beyond_done += 1
#             reward = 0.0
#
#         self.steps_taken += 1
#
#         if self.steps_taken >= self.max_steps:
#             done = True
#
#         return np.array(self.state), reward, done, {}

Transition = namedtuple("Transition", field_names=[
    "state", "action", "reward", "next_state", "done"])

class ReplayMemory(object):
    def __init__(self, capacity):
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        """Creates `Transition` and insert
        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)
        state = np.array(state).astype("float64")
        next_state = np.array(next_state).astype("float64")

        self.memory[self.cursor] = Transition(state, action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size):
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the length """
        return len(self.memory)

def get_data_and_monitor_online_rl(model, target_model, agent, env):
    import gridworld
    # Adapted from https://gist.github.com/kkweon/52ea1e118101eb574b2a83b933851379
    stats = {}
    # set_stats(model, stats)
    stats['mean_reward'] = 0

    # Max size of the replay buffer
    capacity = 50000

    # After max episode, eps will be `min_eps`
    max_eps_episode = 50

    # eps will never go below this value
    min_eps = 0.15

    # Number of transition samples in each minibatch update
    batch_size = 64

    # Discount parameter
    gamma = 0.95
    # gamma = 1

    # Max number of episodes to run
    n_episode = 100

    # Random seed
    seed = 1

    # Win if you average at least this much reward (max reward is 200) for
    # num_episodes_to_average consecutive episodes
    reward_threshold = -20 # Cliff World
    # reward_threshold = 0.8
    num_episodes_to_average = 10

    # If set (an integer), clip the absolute difference between Q_pred and
    # Q_target to be no more than this
    td_error_clipping = None

    episode_print_interval = 10

    steps = 0

    stats['reward_threshold'] = reward_threshold

    # env = gridworld.GridworldEnvironment(gridworld.getCliffGrid())
    rewards = deque(maxlen=num_episodes_to_average)
    input_dim, output_dim = 2, 4
    replay_memory = ReplayMemory(capacity)

    def train_helper(minibatch):
        """Prepare minibatches
        Args:
            minibatch (List[Transition]): Minibatch of `Transition`
        Returns:
            float: Loss value
        """
        states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        done = np.array([x.done for x in minibatch])

        Q_predict = model.run(states)
        Q_target = np.copy(Q_predict)
        for s, state in enumerate(states):
            target = rewards[s] + (1 - done[s]) * gamma * np.max(target_model.run(np.array([next_states[s]])), axis=1)
            # if target > 10 and -1 not in next_states[s] :
                # print("target model", np.max(target_model.run(np.array([next_states[s]])), axis=1))
                # print("STEPS", steps)
                # print("target", target)
                # print(state, actions[s], rewards[s], next_states[s])
            if -1 in next_states[s]:
                target = [rewards[s] for _ in range(4)]
                Q_target[s] = target
            else:
                Q_target[s, actions[s]] = target

            # if td_error_clipping is not None:
            #     Q_target = Q_predict + np.clip(
            #         Q_target - Q_predict, -td_error_clipping, td_error_clipping)

        # print("max target", Q_target.max())
        # print("max error", np.abs(error).max())

        return Q_target

    annealing_slope = (min_eps - 1.0) / max_eps_episode

    for episode in range(n_episode):
        eps = max(annealing_slope * episode + 1.0, min_eps)
        # render = play_every != 0 and (episode + 1) % play_every == 0

        env.reset()
        s = np.array(env.state)
        done = False
        total_reward = 0

        possible_action_list = env.gridWorld.get4Actions(s)#['north','west','south','east']

        while not done:
            a = agent.getAction(s)

            s2, r = env.doAction(a)
            steps += 1

            done = env.gridWorld.isTerminal(s2) # deleted info

            total_reward += r

            next_state = s2 if not done else (-1, -1) # define terminal state to be -1, -1
            action_num = possible_action_list.index(a)
            reward = r if r is not None else 0
            print("(s, action_num, reward, next_state, done)", (s, action_num, reward, next_state, done))
            replay_memory.push(s, action_num, reward, next_state, done)

            if len(replay_memory) > batch_size and steps % 5 == 0:
                minibatch = replay_memory.pop(batch_size)
                Q_target = train_helper(minibatch)
                states = np.vstack([x.state for x in minibatch])
                yield states, Q_target

            # if steps % 100 == 0:
            if steps % 2000 == 0:
                print("UPDATE TARGET")
                target_model.set_weights(copy.deepcopy(model.layers))

            s = np.array(s2)
            possible_action_list = env.gridWorld.get4Actions(s)

        rewards.append(total_reward)
        if (episode + 1) % episode_print_interval == 0:
            print("[Episode: {:3}] Reward: {:5} Mean Reward of last {} episodes: {:5.1f} epsilon: {:5.2f}".format(
                episode + 1, total_reward, num_episodes_to_average, np.mean(rewards), eps))

        if len(rewards) == rewards.maxlen:
            stats['mean_reward'] = np.mean(rewards)
            if np.mean(rewards) >= reward_threshold:
                print("Completed in {} episodes with mean reward {}".format(
                    episode + 1, np.mean(rewards)))
                stats['reward_threshold_met'] = True
                break
    else:
        # reward threshold not met
        print("Aborted after {} episodes with mean reward {}".format(
            episode + 1, np.mean(rewards)))

def get_data_and_monitor_offline_rl(model, target_model, agent, env):
    import gridworld
    # Adapted from https://gist.github.com/kkweon/52ea1e118101eb574b2a83b933851379
    stats = {}
    # set_stats(model, stats)
    stats['mean_reward'] = 0

    # Max size of the replay buffer
    capacity = 50000

    # After max episode, eps will be `min_eps`
    max_eps_episode = 50

    # eps will never go below this value
    min_eps = 0.15

    # Number of transition samples in each minibatch update
    batch_size = 64

    # Discount parameter
    gamma = 0.9
    # gamma = 1

    # Max number of episodes to run
    n_episode = 2000

    # Random seed
    seed = 1

    # Win if you average at least this much reward (max reward is 200) for
    # num_episodes_to_average consecutive episodes
    reward_threshold = -20 # Cliff World
    # reward_threshold = 0.8
    num_episodes_to_average = 10

    # If set (an integer), clip the absolute difference between Q_pred and
    # Q_target to be no more than this
    td_error_clipping = None

    episode_print_interval = 10

    steps = 0

    stats['reward_threshold'] = reward_threshold

    # env = gridworld.GridworldEnvironment(gridworld.getCliffGrid())
    rewards = deque(maxlen=num_episodes_to_average)
    input_dim, output_dim = 2, 4
    replay_memory = ReplayMemory(capacity)

    def train_helper(minibatch):
        """Prepare minibatches
        Args:
            minibatch (List[Transition]): Minibatch of `Transition`
        Returns:
            float: Loss value
        """
        states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        done = np.array([x.done for x in minibatch])

        Q_predict = model.run(states)
        Q_target = np.copy(Q_predict)
        for s, state in enumerate(states):
            target = rewards[s] + (1 - done[s]) * gamma * np.max(target_model.run(np.array([next_states[s]])), axis=1)
            # if target > 10 and -1 not in next_states[s] :
                # print("target model", np.max(target_model.run(np.array([next_states[s]])), axis=1))
                # print("STEPS", steps)
                # print("target", target)
                # print(state, actions[s], rewards[s], next_states[s])
            if -1 in next_states[s]:
                target = [rewards[s] for _ in range(4)]
                Q_target[s] = target
            else:
                Q_target[s, actions[s]] = target

            # if td_error_clipping is not None:
            #     Q_target = Q_predict + np.clip(
            #         Q_target - Q_predict, -td_error_clipping, td_error_clipping)

        # print("max target", Q_target.max())
        # print("max error", np.abs(error).max())

        return Q_target

    annealing_slope = (min_eps - 1.0) / max_eps_episode

    # Fill up the replay buffer
    for transition in list_of_transitions:
        replay_memory.push(*transition)

    print("replay_memory len", len(replay_memory))

    for episode in range(n_episode):
        eps = max(annealing_slope * episode + 1.0, min_eps)
        # render = play_every != 0 and (episode + 1) % play_every == 0

        env.reset()
        s = np.array(env.state)
        done = False
        total_reward = 0

        possible_action_list = env.gridWorld.get4Actions(s)#['north','west','south','east']

        # while not done:
            # random_num = np.random.rand()
            # if random_num <= eps:
            #     a = np.random.choice(possible_action_list)
            # else:
            #     a = model.get_action(s[np.newaxis,:], eps)[0]

            # s2, r = env.doAction(a)
            # steps += 1

            # done = env.gridWorld.isTerminal(s2) # deleted info

            # total_reward += r

            # next_state = s2 if not done else (-1, -1) # define terminal state to be -1, -1
            # action_num = possible_action_list.index(a)
            # reward = r if r is not None else 0
            # print("(s, action_num, reward, next_state, done)", (s, action_num, reward, next_state, done))
            # replay_memory.push(s, action_num, reward, next_state, done)

        steps += 1
        # print("steps", steps)

        if len(replay_memory) > batch_size and steps % 5 == 0:
            minibatch = replay_memory.pop(batch_size)
            # print(minibatch)
            # import ipdb; ipdb.set_trace()
            Q_target = train_helper(minibatch)
            states = np.vstack([x.state for x in minibatch])
            yield states, Q_target

        # if steps % 100 == 0:
        if steps % 1000 == 0:
            print("UPDATE TARGET")
            target_model.set_weights(copy.deepcopy(model.layers))

            # s = np.array(s2)
            # possible_action_list = env.gridWorld.get4Actions(s)

        # rewards.append(total_reward)
        # if (episode + 1) % episode_print_interval == 0:
        #     print("[Episode: {:3}] Reward: {:5} Mean Reward of last {} episodes: {:5.1f} epsilon: {:5.2f}".format(
        #         episode + 1, total_reward, num_episodes_to_average, np.mean(rewards), eps))

        # if len(rewards) == rewards.maxlen:
        #     stats['mean_reward'] = np.mean(rewards)
        #     if np.mean(rewards) >= reward_threshold:
        #         print("Completed in {} episodes with mean reward {}".format(
        #             episode + 1, np.mean(rewards)))
        #         stats['reward_threshold_met'] = True
        #         break
    else:
        # reward threshold not met
        print("Aborted after {} episodes with mean reward {}".format(
            episode + 1, np.mean(rewards)))
