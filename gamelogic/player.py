from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, letter, learning_algorithm=None):
        self.letter = letter
        self.learning_algorithm = learning_algorithm

    def get_move(self, game):
        return self.learning_algorithm.predict(game.get_state())

    def learn_from_game_history(self, game_history, current_game_index):
        self.learning_algorithm.train(game_history, current_game_index)

class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        self.letter = letter

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input('Enter your move (0-8): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val