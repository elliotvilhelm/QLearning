import numpy as np
import random
import gym
import queue

import matplotlib.pyplot as plt



env = gym.make('CartPole-v0')

GAMMA = 0.99 # determines how much future rewards are worth range 0-1 (discount) no randomness here future is definite
#LEARNING_RATE = .99 # the extent to which new information overrides old
NUM_EPISODES = 3000
MIN_EXPLORE_RATE = 0.01

COMPLETION_IN_ROW = 0
"""
the learning rate, set between 0 and 1.
Setting it to 0 means that the Q-values are never updated,
hence nothing is learned. Setting a high value such as 0.9
means that learning can occur quickly.
"""
Results = queue.deque()

MIN_LEARNING_RATE = 0.1
#Q = np.random.rand(540, env.action_space.n)
Q = np.zeros([162, 2])
# 162 boxes = 3 * 3 * 6 * 3
#BIN_NUMBERS = (3, 3, 6, 5)
def get_Box(observation):
    x, x_dot, theta, theta_dot = observation

    # {-2.4, 2.4}
    if x < -.8:
        box_number = 0
    elif x < .8:
        box_number = 1
    else:
        box_number = 2

    if x_dot < -.5:
        pass
    elif x_dot < .5:
        box_number += 3
    else:
        box_number += 6
    if theta < np.radians(-12):
        pass
    elif theta < np.radians(-1.0):
        box_number += 9
    elif theta < np.radians(0):  # golden spot
        box_number += 18
    elif theta < np.radians(1.0):
        box_number += 27
    elif theta < np.radians(12):
        box_number += 36
    else:
        box_number += 45
	#
    # if theta_dot < np.radians(-50):
    #     pass
    # elif theta_dot < np.radians(-20):
    #     box_number += 54
    # elif theta_dot < np.radians(20):
    #     box_number += 108
    # elif theta_dot < np.radians(50):
    #     box_number += 162
    # else:
    #     box_number += 216


    if theta_dot < np.radians(-50):
        pass
    elif theta_dot < np.radians(50):
        box_number += 54
    else:
        box_number += 108
    return box_number

def update_explore_rate(episode):

    # this number slowly decreases from roughly 2.4
    # start big, LEARN, stop exploring as much
    if COMPLETION_IN_ROW >= 300:
        return 0

    return max(MIN_EXPLORE_RATE, min(1, 1.0 - np.log10((episode+1)/25)))

def update_learning_rate(episode):
    """
    The learning rate or step size determines to what
    extent the newly acquired information will override
    the old information. A factor of 0 will make the agent
    not learn anything, while a factor of 1 would make the agent
    consider only the most recent information.
    """
    # we stop learning as fast as time progresses
    #return min()
    if COMPLETION_IN_ROW >= 300:
        return 0
    return max(MIN_LEARNING_RATE, (min(0.5, 1.0 - np.log10((episode + 1) / 25))))

def update_action(state, explore_rate):
    if random.random() < explore_rate:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


def q_learn():
    total_reward = 0
    total_completions = 0
    explore_rate = update_explore_rate(0)
    learning_rate = update_learning_rate(0)
    streak = 0

    for i in range(NUM_EPISODES):
        observation = env.reset()
        state_0 = get_Box(observation)
        for _ in range(250):
            env.render()
            action = update_action(state_0, explore_rate)
            obv, reward, done, info = env.step(action)

            state_1 = get_Box(obv)
            q_max = np.max(Q[state_0])
            Q[state_0, action] += learning_rate*(reward + GAMMA*np.amax(Q[state_1]) - Q[state_0, action])

            state_0 = state_1
            total_reward += reward

            if done:
                Results.append([i, _, learning_rate, explore_rate])
                #print("Episode finished after %f time steps" % (_))
                #print("learning rate: ", learning_rate)
                #print("explore rate: ", explore_rate)
                #print("best q: ", q_max)
               # print("total completions: ", total_completions)


                if _ > 198:
                    COMPLETION_IN_ROW += 1
                    total_completions+=1
                else:
                    COMPLETION_IN_ROW = 0
                    #pass
                    streak = total_completions
                break
                print(Q)

        learning_rate = update_learning_rate(i)
        explore_rate = update_explore_rate(i)
       # print("explore rate: ", explore_rate)
       # print("learning rate: ", learning_rate)

        print("total completions: ", total_completions)
        #print("streak: ", streak)
        streak=0
        print("Episode finished after %f time steps" % (_))
        print("Streak: ", COMPLETION_IN_ROW)
        #print("REWARD/TIME: ", total_reward/(i+1))
        print("Episode: ", i)


    #print("Final Q values: ", Q)

def main():
    q_learn()


    for i in range(NUM_EPISODES):
        plt.plot(Results[i][0], Results[i][1], 'rx')
        plt.plot(Results[i][0], Results[i][2]*100, 'bx')


    plt.show()


if __name__ == "__main__":
    main()














