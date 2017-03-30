import numpy as np


###############################################################################
### Basic Game Parameters

# The states of the Game
states = ["left", "right"]

# Probabilities with which the states are initialized (p_left is probability of
# the state being initalized as left)
p_left = 0.5
prob_states = [p_left, 1-p_left]

# The actions available to the player
actions = ["left", "right", "listen"]

# The observations returned to the player after every actions
observations = ["left", "right"]

# The probability of getting the correct observation (and wrong observation)
# by listening
p_correct_obs = 0.85
prob_obs = np.array([p_correct_obs, 1-p_correct_obs])

###############################################################################
### Value Iteration Parameters

# The discount factor used in value iteration
GAMMA = 1

# each plan is a triple
# (action, map from observation to plans in old set, alpha vector)
# trivial plan
trivial_plan = (None, None, [0, 0])

# the trivial map from observations to plans in old set
trivial_map = {"left": trivial_plan, "right": trivial_plan}

# set of height 1 plans
step_1_set = [("left", trivial_map, []), ("listen", trivial_map, []), ("right", trivial_map, [])]
