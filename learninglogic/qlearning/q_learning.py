import numpy as np
from learning_algorithm import LearningAlgorithm
import pickle

class QLearning(LearningAlgorithm):
    def __init__(self, game, params):
        super().__init__(game,params)
        self.alpha = params.get('alpha', 0.5)
        self.gamma = params.get('gamma', 0.9)
        self.epsilon = params.get('epsilon', 1)
        self.initial_epsilon = params.get('epsilon', 1)
        self.min_epsilon = params.get('min_epsilon', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.999)
        self.q_table = {}

    def train(self, game_history, current_game_index):
        for i in range(len(game_history)):
            state, action, reward, player = game_history[i]

            if i != len(game_history) - 1:
                next_state, _, _, _ = game_history[i+1]
            else:
                next_state = None
                
            self.update_q_table(state, action, reward, next_state)
            self.decay_epsilon(current_game_index)

    def predict(self, game_state):
        legal_moves = self.game.available_moves()
        print(self.epsilon)
        if np.random.random() < self.epsilon:  # Explore
            return int(np.random.choice(legal_moves))
        else:  # Exploit
            q_values_of_state = self.q_table.get(game_state, {})
            q_values_of_legal_moves = {action: q_values_of_state.get(action, 0) for action in legal_moves}

            max_q_value = max(q_values_of_legal_moves.values())
            actions_with_max_q_value = [action for action, q_value in q_values_of_legal_moves.items() if q_value == max_q_value]
            
            return int(np.random.choice(actions_with_max_q_value))

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        next_state_q_value = 0.0
        if next_state is not None:
            next_state_q_values = self.q_table.get(next_state, {}).values()
            if next_state_q_values:
                next_state_q_value = max(next_state_q_values)

        current_q_value = self.q_table[state][action]
        td_error = reward + self.gamma * next_state_q_value - current_q_value

        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self, i):
        self.epsilon = max(self.min_epsilon, self.initial_epsilon * self.epsilon_decay**i)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        print('filepath: ' + filepath)
        try:
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
        except Exception as e:
            print("An error occurred while loading Q-table: ", e)
