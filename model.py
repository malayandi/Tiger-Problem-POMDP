import numpy as np

from parameters import *

"""
Function that returns the probability of reaching FINAL state from INITIAL state
taking ACTION.
"""
def transition(initial, action, final):
    if action == "listen":
        return (initial == final) * 1
    elif action == "left" or action == "right":
        return 0.5

"""
Function that returns the probability of getting OBSERVATION after taking ACTION
to land in STATE.
"""
def sensor(observation, action, state):
    if action == "left" or action == "right":
        return 0.5
    elif action == "listen":
        return p_correct_obs if observation == state else 1-p_correct_obs


"""
Function that returns the reward for taking ACTION in STATE.
"""
def getReward(state, action):
    if action == "listen":
        return -1
    elif action == "left" or action == "right":
        if action != state:
            return 10
        else:
            return -100

"""
Function that returns an observation and a new state to which the game is
reinitialised after taking ACTION in STATE.
"""
def getObservation(state, action):
    if action == "listen":
        if state == "left":
            observation = np.random.choice(observations, p=prob_obs)
        elif state == "right":
            observation = np.random.choice(observations, p=1-prob_obs)
    elif action == "left" or action == "right":
        observation = np.random.choice(observations)
        # reinitializing state randomly
        state = np.random.choice(states, p=prob_states)
    return observation, state
