"""
Runs an instance of the tiger game. Command line input has the following format:

python play.py ["AI"/"Human"] [MAX_TIME]

If playing with a human player, the user will be prompted with "What action
would you like to make? "
The options are "left", "right" or "listen"
"""

import sys
import numpy as np

from game import Game
from agent import Agent, AI_Agent, Human_Agent

def play(player, max_time):
    time = 0
    game = Game()
    if player == "AI":
        player = AI_Agent()
    elif player == "Human":
        player = Human_Agent()
    while time < max_time:
        if isinstance(player, Human_Agent):
            move = input("What action would you like to make? ")
            move = player.pick_action(move)
        else:
            move = player.pick_action()
        reward, observation = game.respond(move)
        player.update_observation(observation)
        if move == "listen":
            tiger_sound(observation)
        player.update_reward(reward)
        print("You received a reward of " + str(reward))
        time += 1
    print("Total Reward: " + str(player.get_reward()))

def tiger_sound(observation):
    door = "left" if observation == "TL" else "right"
    print("The tiger sound came from the " + door + " door")

if __name__ == "__main__":
    args = sys.argv
    # whether the player is human or AI
    player = args[1]
    # the number of time steps over which the game occurs
    max_time = int(args[2])

    play(player, max_time)
