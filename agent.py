import numpy as np

from game import Game

# The actions available to the player
actions = ["left", "right", "listen"]

class Agent:
    """
    A player in the Tiger game.
    """
    def __init__(self):
        # The belief probability that the agent is in the left states
        # Note that b_right is simply 1 - b_left
        self._b_left = None
        self._reward = 0
        self._observation = None

    def act(self, game, action):
        reward, observation = game.respond(action)
        self._reward += reward
        self._observation = observation

    def update_reward(self, new_reward):
        self._reward += new_reward

    def update_observation(self, new_observation):
        self._observation = new_observation

    def get_reward(self):
        return self._reward

class AI_Agent(Agent):
    """
    An AI agent.
    """
    def __init__(self):
        super(AI_Agent, self).__init__()

    def pick_action(self):
        """
        Temporarily using a random strategy.
        """
        return actions[np.random.choice([0,1,2])]

class Human_Agent(Agent):
    """
    A human agent.
    """
    def __init__(self):
        super(Human_Agent, self).__init__()

    def pick_action(self, action):
        return action
