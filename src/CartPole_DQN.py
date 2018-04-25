import tensorflow as tf
import numpy as np
import gym
from collections import deque

"""
TO DO:
Implement Experience Replay to Train network off of


Implement e decay

"""

"""
GOLD BABY GOLD
"""
# learning parameters
GAMMA = .99  # discount factor
e = 1.0   # exploration rate
num_episodes = 10000

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

n_nodes_hl1 = 12
n_nodes_hl2 = 60
n_nodes_hl3 = 12
n_nodes_hl4 = 300
n_nodes_hl5 = 10


def update_explore_rate(episode):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - np.log10((episode + 1) / 25)))


def update_learning_rate(episode):
    # return 1;
    return max(MIN_LEARNING_RATE, (min(0.5, 1.0 - np.log10((episode + 1) / 50))))

def neural_network_model(data):
    '''
    Define Model
    '''
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([obs_dim, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    # hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
    #                   'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}

    # hidden_5_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
    #                   'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}

    # notice biases size 3 x number of classes
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    '''
    FORWARD PROP!!!!
    '''
    # (input data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # activation
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.sigmoid(l3)

    # l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    # l4 = tf.nn.relu(l4)

    # l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    # l5 = tf.nn.relu(l5)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output



D = deque()

env = gym.make('CartPole-v0')
# env = gym.make('FrozenLake-v0')
# env = gym.make('MountainCar-v0')

# saver = tf.train.Saver()
tf.reset_default_graph()

obs_dim = env.observation_space.shape[0]
n_classes = env.action_space.n
input_Layer = tf.placeholder(shape=[1, obs_dim], dtype=tf.float32)
# input_Layer = tf.nn.l2_normalize(tf.placeholder(shape=[1, 4], dtype=tf.float32),1)
Qout = neural_network_model(input_Layer)
predict = tf.argmax(Qout, 1)
nextQ = tf.placeholder(shape=[1, n_classes], dtype=tf.float32)
cost = tf.reduce_sum(tf.square(nextQ - Qout))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)
# # LR: .001 default
optimizer = tf.train.AdamOptimizer(learning_rate=.005)
updateModel = optimizer.minimize(cost)

"""
END GOLD
"""

# init = tf.global_variables_initializer()


successful_episodes = 0
ep_step = 0.0
total_reward = 0.0
with tf.Session() as sess:
    # saver = tf.train.import_meta_graph('Models/model.ckpt-1000.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('Models/./'))
    # saver = tf.train.import_meta_graph('Models/model.ckpt.meta')
    # saver.restore(sess, "Models/model.ckpt")
    # saver = tf.train.import_meta_graph('Models/model.ckpt.meta')

    # saver.restore(sess, "Models/model.ckpt")

    sess.run(tf.global_variables_initializer())

    for episode in range(num_episodes):
        s = env.reset()
        step = 0
        ep_reward = 0.0
        while step < 250:
            step += 1
            a, allQ = sess.run([predict, Qout], feed_dict={input_Layer: np.array([s])})  # a is action with max q value
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # Get new state and reward from environment
            s1, reward, done, _ = env.step(a[0])
            ep_reward += reward
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={input_Layer: np.array([s1])})  # one hot
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = reward + GAMMA * maxQ1  # TargetQ is 1x4 a[0] was max q valued action at s_0
            # thus targetQ at the max action is set to r + y*maxQ1 all others are left alone
            sess.run([updateModel, cost], feed_dict={input_Layer: np.array([s]), nextQ: targetQ})
            s = s1
            if done is True:
                total_reward += ep_reward
                # Reduce chance of random action as we train the model.
                e = update_explore_rate(episode=episode)
                # e = 1. / ((episode / 50) + 10)
                if ep_reward > 195:
                    e = -10

                if step > 150:
                    successful_episodes += 1
                break

        if episode % 100 == 0:
            # if total_reward//100.0 == 200.0:
            # 	break

            print("episode: ", episode, "avg Reward: ", total_reward // 100.0)
            total_reward = 0.0

    print("Number of succescful episodes", successful_episodes)
    print("Percent of successful episodes: ", 100 * successful_episodes / num_episodes)
    # Save the variables to disk.
    # save_path = saver.save(sess, "Models/model.ckpt")
    # print("Model saved in file: %s" % save_path)

    # total_reward = 0.0
    for episode in range(100):
        s = env.reset()
        step = 0
        ep_reward = 0.0
        while step < 200:
            step += 1
            env.render()
            action = sess.run([predict], feed_dict={input_Layer: np.array([s])})  # a is action with max q value
            action = action[0]
            s1, reward, done, _ = env.step(action[0])
            if done is True:
                print(step)
                total_reward += ep_reward
                if step > 150:
                    successful_episodes += 1
                break

