# Tiger-Problem-POMDP
Implementation of POMDP algorithms (value iteration and point-based value iteration) on the tiger problem, as described in Littman, Cassandra and Kaelbling (1998).

## Usage

Clone the repo locally using the following command:

	git clone https://github.com/malayandi/Tiger-Problem-POMDP.git
	
### Human Play	
	
To play the game as a human player, run the following command:

	python play.py Human max_time
	
where `max_time` is the horizon of the game.

### AI Play (Using Value Iteration)

To have an AI play and solve the game exactly using value iteration, run the following command:

	python play.py AI max_time
	
## Running Your Own Experiments

It is fairly straightforward to run your own experiments with this code. The core files are organized as follows (files that are starred need not be changed):

* `play.py`: The main file that runs of the game.
* `game.py`: Keeps track of the agent's overall performane in the game.
* `model.py`: Specifies the uncertainty dynamics of the game, including the transition distribution, observation distribution and reward function.
* `parameters.py`: Specifies the parameters of the game, including the set of states, set of actions available to the agent, set of observations, discount factor etc.
* `agent.py`: Contains subclasses for Human/ AI agents. For AI agents, specifies the initial belief and solver to use (i.e. Value Iteration or Point-Based Value Iteration).
* `valueIteration.py`: The POMDP Value Iteration algorithm.
* `pruning.py`: The pruning method used in valueIteration.py
* `pbvi.py`: The Point-Based Value Iteration algorithm.

## Contact

For questions or inquiries, please contact <malayandi12@gmail.com>.

## References

1. Sondik, Edward J. *The Optimal Control of Partially Observable Markov Processes*. PhD thesis, Stanford Uni- versity, 1971.
2. Kaelbling, Leslie Pack, Littman, Michael L, and Cassandra, Anthony R. *Planning and acting in partially observable stochastic domains*. Artificial intelligence, 101(1): 99–134, 1998.
3. Pineau, J., Gordon, G. and Thrun, S. *Point-based value iteration: An anytime algorithm for POMDPs*.  IJCAI, vol. 3, pp. 1025-1032. 2003.