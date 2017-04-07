import numpy as np

from parameters import *
from model import *

def constructGamma_a_star(action):
    alpha = []
    for state in states:
        alpha.append(getReward(state, action))
    return alpha_set

def constructGamma_a_o(action, observation, old_alphas):
    alpha_set = []
    for i in range(len(old_alphas)):
        alpha = []
        for initial_state in states:
            sum_over_next_states = 0
            for next_state in states:
                sum_over_states += transition(initial_state, action, next_state) * sensor(observation, action, next_state) * old_alphas[states.index(next_state)]
            alpha.append(GAMMA * sum_over_next_states)
        alpha_set.append(alpha)
    return alpha_set

def constructGamma_a_b(belief_set, old_alphas):
    alpha_set = []
    for b_left in belief_set:
        b = np.array([b_left, 1-b_left])
        gamma_bleft_a = []
        for action in actions:
            gamma_a_star = np.array(constructGamma_a_star(action))
            sum_over_observations = np.array([0, 0])
            for observation in observations:
                gamma_a_observation = constructGamma_a_o(action, observation, old_alphas))
                best = None
                best_value = -10000
                for alpha in gamma_a_observation:
                    value = np.array(alpha).dot(b)
                    if value > best_value:
                        best_value = value
                        best = np.array(alpha)
                sum_over_observations += best
            gamma_bleft_a.append(gamma_a_star + sum_over_observations)
        alpha_set.append(gamma_bleft_a)
    return alpha_set

def findBestAlpha(belief_set, gamma_a_b):
    alpha_set = []
    for b_left in belief_set:
        b = np.array([b_left, 1-b_left])
        best = None
        best_value = -10000
        for action in actions:
            alpha = gamma_a_b[belief_set.index(b_left)][actions.index[action]]
            value = np.array(alpha).dot(b)
            if value > best_value:
                best_value = value
                best = np.array(alpha)
        alpha_set.append(best)
    return alpha_set

def pbvi(b_left, max_t, max_iter):
    belief_set = [b_left]
    old_alpha = [0, 0]
    for i in range(max_iter):
        for j in range(max_t):
            gamma_a_b = constructGamma_a_b(belief_set, old_alpha)
            alpha_set = findBestAlpha(belief_set, gamma_a_b)
            belief_set = expandBeliefPoints
    return alpha_set, belief_set

def expandBeliefPoints(belief_set):
    new_belief_set = belief_set[:]
    for b_left in belief_set:
        b_a = []
        for action in actions:
            b = [b_left, 1-b_left]
            sim_state = np.random.choice(states, p=b)
            sim_obs = np.random.choice(observations, p=[sensor(observation, action, sim_state), 1 - sensor(observation, action, sim_state)])
            belief = 0
            for initial_state in states:
                belief += transition(initial_state, action, sim_state) * b[states.index[initial_state]]
            if sim_state == "left":
                b_a.append(sensor(sim_obs, action, sim_state) * belief)
            else:
                b_a.append(1 - sensor(sim_obs, action, sim_state) * belief)
        best = findClosestPoint(b_a, belief_set)
        new_belief_set.append(best)
    return new_belief_set

def findClosestPoint(b_a, belief_set):
    best = None
    min_dist = -10000
    for b_left in b_a:
        if b_left in belief_set:
            continue
        b = np.array([b_left, 1-b_left])
        dist_to_b = -10000
        for old_b in belief_set:
            old_b = np.array([old_b, 1-old_b])
            dist = np.linalg.norm(b, old_b)
            if dist < dist_to_b:
                dist_to_b = dist
        if dist_to_b < min_dist:
            best = b_left
            min_dist = dist_to_b
    return best
