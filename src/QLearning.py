import numpy as np
import random
import gym

env = gym.make("CartPole-v0")
"""
GAMMA determines how much future reward is worth no
randomness in the future
"""
GAMMA = 0.99

NUM_EPISODES = 2000
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
Q = np.random.rand(162, env.action_space.n)

# BIN_NUMBERS = (3,3,6,3)
# 162 boxes = 3*3*6*3

def get_Box(obv):
	x, x_dot, theta, theta_dot = obv

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
	elif theta < np.radians(-1.5):
		box_number += 9
	elif theta < np.radians(0):  # goldzone
		box_number += 18
	elif theta < np.radians(1.5):
		box_number += 27
	elif theta < np.radians(12):
		box_number += 36
	else:
		box_number += 45

	if theta_dot < np.radians(-50):
		pass
	elif theta_dot < np.radians(50):
		box_number += 54
	else:
		box_number += 108

	return box_number


def update_explore_rate(episode):
	return max(MIN_EXPLORE_RATE, min(1, 1.0 - np.log10((episode + 1) / 25)))

def update_learning_rate(episode):
	# return 1;
	return max(MIN_LEARNING_RATE, (min(0.5, 1.0 - np.log10((episode + 1) / 50))))


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

	for i in range(NUM_EPISODES):
		observation = env.reset()
		state_0 = get_Box(observation)
		for _ in range(250):
			env.render()
			action = update_action(state_0, explore_rate)
			obv, reward, done, info = env.step(action)
			state_1 = get_Box(obv)

			q_max = np.max(Q[state_0])
			#print(q_max)


			Q[state_0, action] += learning_rate*(reward + GAMMA*np.amax(Q[state_1])- Q[state_0, action])

			state_0 = state_1

			total_reward += reward

			if done:
				print(Q)
				print("Episode finished after ", _, " time steps")

				if _ > 192:
					total_completions += 1
				break

		learning_rate = update_learning_rate(i)
		explore_rate = update_explore_rate(i)
		#print("explore rate: ", explore_rate)
		print("learning rate: ", learning_rate)

		print("Completions : ", total_completions)
		print("REWARD/TIME: ", total_reward/(i+1))
		print("Trial: ", i)
		#print("Final Q values: ", Q)

def main():
	q_learn()

if __name__ == "__main__":
	main()












