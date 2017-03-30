import numpy as np
import random

from model import *
from parameters import *

class Game:
    """
    The game itself. Each instance of a game contains two parameters: state
    and time.
    """
    # DO NOT INITIALIZE STATE - FOR TESTING ONLY
    def __init__(self, _state = None):
        self._state = _state if _state else np.random.choice(states, p=prob_states)
        self._time = 0

    def getState(self):
        return self._state

    def respond(self, action):
        """
        Responds to an agent's action with a reward and observation (i.e. TL
        or TR) and randomly reinitialises the state.
        """
        rwrd = getReward(self._state, action)
        obsrv, self._state = getObservation(self._state, action)
        self._time += 1

        return rwrd, obsrv
