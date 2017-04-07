import numpy as np

"""
Given a set of conditional plans, returns a set of alpha vectors corresponding
to the set of plans. Maintains the index of the conditional plans.
"""
def formAlphaSet(plan_set):
    alpha_set = []
    for plan in plan_set:
        alpha_set.append(plan[2])
    return alpha_set

"""
Evaluates a plan over belief space.
"""
def evaluateBeliefState(b_left, alpha):
    belief = np.array([b_left, 1-b_left])
    alpha = np.array(alpha)
    return belief.dot(alpha)

"""
Given a set of conditional plans, returns the set of conditional plans that are
optimal over some interval in belief space and returns a dictionary mapping
each conditional plan's index to the set of b_left over which it is optimal.
b_jump is the size of each jump in belief space in the linear program.
"""
def prune(plan_set, b_jump = 0.01):
    alpha_set = formAlphaSet(plan_set)
    parsimonius_set = []
    optimal_map = {}
    for b_left in np.arange(0, 1.001, b_jump):
        values = []
        for i in range(len(alpha_set)):
            values.append(evaluateBeliefState(b_left, alpha_set[i]))
        best = np.argmax(values)
        if best not in optimal_map:
            optimal_map[best] = [b_left]
        else:
            optimal_map[best].append(b_left)
    for i in list(optimal_map.keys()):
        parsimonius_set.append(plan_set[i])
    return parsimonius_set, optimal_map

"""
Given a set of conditional plans a map between the index of the plans and the
region over belief space where it is optimal, returns a map that maps the
initial action to regions over belief space where it is optimal.
"""
def createOptimalActionMap(plan_set, optimal_map):
    action_map = []
    for i in list(optimal_map.keys()):
        plan = plan_set[i]
        action = plan[0]
        smallest = min(optimal_map[i])
        biggest = max(optimal_map[i])
        action_map.append((action, [str(round(smallest, 2)), str(round(biggest, 2))]))
    action_map = sorted(action_map, key = lambda x : x[1])
    return action_map
