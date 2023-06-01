# Incomplete - Learning exercise

## Tic-Tac-Toe Q-Learning (Quality Learning)

This is an implementation of a Tic-Tac-Toe game where two AI agents, represented as X and O, are trained to play the game using Q-Learning, a type of reinforcement learning technique.

### What is Q-Learning?

Q-Learning is an off-policy reinforcement learning algorithm. It learns a policy that decides what action to take based on an action-value function. This function provides an estimation of the total reward an agent would receive, starting at a particular state and performing a certain action at that state.

In Q-Learning, an agent interacts with the environment to learn the best action to perform in each state. It does this by maintaining a table called the Q-table. Each cell in the Q-table represents the expected future reward that the agent can achieve by taking an action at a particular state. The agent explores the environment and updates the Q-table using the Q-Learning algorithm.

### Properties of Q-Learning

1. Off-Policy: In Q-Learning, learning is independent of the policy being followed, meaning the learning process does not follow the current policy for choosing its actions. It instead learns from actions that are greedy with respect to the current estimated action values.

2. Exploration and Exploitation: The agent needs to balance exploration (selecting random actions) and exploitation (always selecting the best known action). This is often done using an ε-greedy strategy. With a probability of ε, a random action is selected (exploration), and with a probability of 1-ε, the action with the highest estimated reward is selected (exploitation).

3. Temporal Difference (TD) Learning: Q-Learning is a type of Temporal Difference learning, meaning the agent learns from a difference, a temporal difference, between the estimated Q-values.

4. Convergence: Given sufficient training under certain conditions, the Q-Learning algorithm finds an optimal policy that achieves the maximum expected reward for each state.

Alpha (Learning Rate): A smaller alpha results in slower learning, but might lead to more stable results.

Gamma (Discount Factor): This parameter dictates how much future rewards should be taken into account. A higher value means that the agent considers future rewards more heavily, but it might make learning slower.

Epsilon (Exploration Rate): More exploration (higher epsilon) is beneficial at the start of training, but over time you want the model to exploit what it has learned (lower epsilon).

### The Implementation

The implementation involves creating a TicTacToe class which has the rules of the game and a QPlayer class which implements the Q-learning. The game can be played between two QPlayer agents, or between a QPlayer agent and a HumanPlayer. In the training phase, the game is played only between two QPlayer agents.

The QPlayer agent explores the game environment by initially taking random actions. As it starts to learn the game, it gradually shifts from exploration to exploitation. It updates the Q-values of its Q-table using the reward received and the maximum Q-value of new state, according to the Q-learning formula.

After training for a sufficient number of games, the Q-table should ideally contain the expected rewards for each possible state-action pair in the game. The QPlayer agent can then use this Q-table to play the game intelligently by choosing the actions that lead to the maximum expected reward.

To run the code, use the following command:

```
python tic_tac_toe.py --trainAI
```

Use the `--showGraph` argument to display a graph of the training process:

```
python tic_tac_toe.py --trainAI --showGraph
```

To play against the trained AI agent, use the following command:

```
python tic_tac_toe.py
```

Note: For best results, ensure the AI agent has been adequately trained (via the `--trainAI` argument) before attempting to play against it.

### Files

- `tic_tac_toe.py`: Main script file containing the implementation of the game and the Q-learning agent.
- `q_table_x.pkl` and `q_table_o.pkl`: These files contain the Q-tables for the X-player and O-player respectively. These files are created after training the AI and are used

## Other Techniques

- Policy Gradients
- Minimax Search
  Sarsa (State-Action-Reward-State-Action): Sarsa is an on-policy algorithm for TD (Temporal Difference) learning. It uses the current policy to choose the next action, thus the same policy is used for both the current state and the next state. This contrasts with Q-learning, which is an off-policy learner.

Deep Q Learning (Deep Q Networks - DQN): This approach extends traditional Q-learning by using a neural network as a function approximator for the Q-table. This allows DQN to handle problems with large state spaces, which would otherwise be infeasible with traditional Q-learning.

Monte Carlo (MC) Methods: Monte Carlo methods can be used in any problem with a well-defined notion of state, but are particularly useful in reinforcement learning problems with delayed reward. These methods typically sample sequences of states, actions, and rewards, and then use these samples to estimate the expected return for each state-action pair.

Double Q-Learning: Double Q-Learning is a Q-Learning variant that helps to mitigate overoptimistic value estimates by maintaining two separate Q-functions, each learned from different experiences. The maximum action selection is derived from one Q-function, and the corresponding value estimate is derived from the other Q-function.

Dueling Q Learning (Dueling DQN): This variant of DQN explicitly separates the representation of state values and (state-dependent) action advantages. This is advantageous for states where the value of taking any action does not vary much, yet the advantage of taking different actions can still be significant.

Advantage Actor-Critic (A2C/A3C) Algorithms: These methods are policy-based but also use a value function to reduce variance. A2C/A3C algorithms have separate policy and value function networks, and they update both networks simultaneously, which makes them more stable and less likely to converge to suboptimal policies.

Proximal Policy Optimization (PPO): PPO is a policy optimization method that's simple to implement, has little hyperparameter tuning, and offers good performance. It's an on-policy method and it's more sample-efficient than the A2C/A3C algorithms.
