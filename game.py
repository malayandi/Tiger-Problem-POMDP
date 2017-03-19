import numpy as np

# The states of the Game
states = ["left", "right"]
# The actions available to the player
actions = ["left", "right", "listen"]
# The observations returned to the player after every actions
# TL: Tiger heard on left
# TR: Tiger heard on right
observations = ["TL", "TR"]
# The probability of getting the correct observation (and wrong observation)
# by listening
prob = [0.85, 0.15]

class Game:
    """
    The game itself. Each instance of a game contains two parameters: state
    and time.
    """
    def __init__(self):
        rand = np.random.choice([0, 1])
        self._state = states[rand]
        self._time = 0

    def respond(self, action):
        """
        Responds to an agent's action with a reward and observation (i.e. TL
        or TR) and randomly reinitialises the state.
        """
        reward = 0
        observation = None
        if action == "listen":
            reward = -1

            if self._state == "left":
                index = np.random.choice([0, 1], p=prob)
            else:
                index = np.random.choice([1, 0], p=prob)
            observation = observations[index]
        else:
            index = np.random.choice([0,1])
            observation = observations[index]

            if self._state == action:
                reward = -100
            else:
                reward = 10
            # reinitializing state
            rand = np.random.choice([0, 1])
            self._state = states[rand]
        # updating time
        self._time += 1

        return reward, observation
