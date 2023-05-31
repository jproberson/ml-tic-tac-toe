
# Obsolete code
class QPlayer(Player):
    def __init__(self, letter, alpha=0.5, gamma=1.0, epsilon=0.1):
        super().__init__(letter)
        self.letter = letter
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = {}

    def get_state(self, game):
        #return tuple(game.board)
        return ''.join(game.board)


    def get_move(self, game):
        state = game.get_state(game)
        legal_moves = [str(i) for i in game.available_moves()]
        if np.random.random() < self.epsilon:  # Explore
            return int(np.random.choice(legal_moves))
        else:  # Exploit
            q_values_of_state = self.q_table.get(state, {})
            q_values_of_legal_moves = {action: q_values_of_state.get(action, 0) for action in legal_moves}

            max_q_value = max(q_values_of_legal_moves.values())
            actions_with_max_q_value = [action for action, q_value in q_values_of_legal_moves.items() if q_value == max_q_value]
            
            return int(np.random.choice(actions_with_max_q_value))

    def get_best_action(self, state, game):
        if state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)
        else:
            return random.choice(game.available_moves())

    def update_q_table(self, state, action, reward, next_state, next_action):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        next_state_q_value = self.q_table.get(next_state, {}).get(next_action, 0)

        current_q_value = self.q_table[state][action]
        td_error = reward + self.gamma * next_state_q_value - current_q_value

        self.q_table[state][action] += self.alpha * td_error    
