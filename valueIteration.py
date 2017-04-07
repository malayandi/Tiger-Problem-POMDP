import numpy as np

from model import *
from parameters import *
from pruning import *

"""
Returns the optimal action after evaluating policies using value iteration.
"""
def valueIteration(initial_set, max_t = 1):
    old_set = []
    current_set = initial_set
    t = 0
    while t < max_t:
        for i in range(len(current_set)):
            plan = current_set[i]
            current_set[i] = evaluatePlan(plan)
        t += 1
        if t == max_t:
            break
        old_set, optimal_map = prune(current_set)
        current_set = []
        for action in actions:
            for left_plan in old_set:
                for right_plan in old_set:
                    mapping = {}
                    mapping["left"] = left_plan
                    mapping["right"] = right_plan
                    plan = (action, mapping, [])
                    current_set.append(plan)
    return current_set

"""
Given a set of plans (with value functions), returns the best action to take
given that the (left) belief state of the agent is b_left.
"""
def pickBestAction(b_left, current_set):
    values = []
    for i in range(len(current_set)):
        alpha = current_set[i][2]
        value = evaluateBeliefState(b_left, alpha)
        values.append(value)
    opt_index = np.argmax(values)
    action = current_set[opt_index][0]
    #count = 0
    #for i in range(len(current_set)):
    #    if values[i] == max(values):
    #        count += 1
    #print(count)
    return action

"""
Returns a new plan, with the same action and mapping as the old plan, but with
the value of the plan for each state computed.
"""
def evaluatePlan(plan):
    action = plan[0]
    map_to_old_plans = plan[1]

    values = []
    for initial_state in states:
        rwrd = getReward(initial_state, action)
        sum_over_states = 0
        for next_state in states:
            T = transition(initial_state, action, next_state)
            sum_over_observations = 0
            for observation in observations:
                Z = sensor(observation, action, next_state)
                next_plan = map_to_old_plans[observation]
                value = next_plan[2][states.index(next_state)]
                sum_over_observations += (Z * value)
            sum_over_states += (T * sum_over_observations)
        value = rwrd + GAMMA * sum_over_states
        values.append(value)

    new_plan = (action, map_to_old_plans, values)
    return new_plan


current_set = valueIteration(step_1_set, max_t = 20)
parsimonius_set, optimal_map = prune(current_set)
action_map = createOptimalActionMap(current_set, optimal_map)
print(action_map)
