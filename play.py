"""
Runs an instance of the tiger game. Command line input has the following format:

python play.py [player] [max_time]

player: The options are "Human" or "AI". If playing with a human player, the
user will be prompted with "What action would you like to make? " The options
are "left", "right" or "listen". If playing with an AI player, the player will
player according to the predetermined strategy.

max_time: The number of time steps over which the game is played.
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
        print("Step " + str(time) + ":")
        if isinstance(player, Human_Agent):
            move = input("What action would you like to make? ")
            move = player.pick_action(move)
        else:
            move = player.pick_action()
        reward, observation = game.respond(move)
        player.update_observation(observation)
        if move == "listen":
            print("You chose to listen!")
            tiger_sound(observation)
        player.update_reward(reward)
        print("You received a reward of " + str(reward) + "\n")
        time += 1
    print("Game over! Total Reward: " + str(player.get_reward()))

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
