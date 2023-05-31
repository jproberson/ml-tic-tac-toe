import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from utils.utils import *
from gamelogic.player import *
from learninglogic.qlearning.q_learning import QLearning

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X' 
        self.current_winner = None
        self.game_history = []
        
    def print_board(self):
        print('--------------------')
        for i in range(3):
            row = []
            for j in range(3):
                if self.board[i*3+j] == ' ':
                    row.append(' ({}) '.format(i*3+j))
                else:
                    row.append('  ' + self.board[i*3+j] + '  ')
            print('|' + '|'.join(row) + '|')
            print('--------------------')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def is_valid_move(self, square):
        return self.board[square] == ' '

    def get_state(self):
        return ''.join(self.board)
    
    def check_draw(self):
        return ' ' not in self.board and self.current_winner is None

    def make_move(self, square, letter):
        if self.is_valid_move(square):
            self.board[square] = letter
            if self.check_win(letter):
                self.current_winner = letter
            elif self.check_draw():
                self.current_winner = 'Draw'
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
        else:
            raise ValueError(f"Invalid move {square}!")

    def check_win(self, letter):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]

        for combination in win_combinations:
            if self.board[combination[0]] == self.board[combination[1]] == self.board[combination[2]] == letter:
                return True
        return False

def play(game, x_player, o_player, print_game=True):
    game_history = []
    player = x_player

    while game.empty_squares():
        if print_game:
            clear_screen()
            game.print_board()

        action = player.get_move(game)

        game.make_move(action, player.letter)
        game_history.append((game.get_state(), action, None))

        if game.current_winner:
            clear_screen()
            game.print_board()
            break

        player = o_player if player == x_player else x_player

    if print_game:
        if game.current_winner == 'Draw':
            print("It's a draw.")
        else:
            print(f"{game.current_winner} has won!")

    return game_history

def play_for_training(game, x_player, o_player):
    reward_mapping = { 'Win': 50, 'Loss': -100, 'Draw': 25, 'NonLosingMove': -5 }
    game_history = []
    while game.empty_squares():
        if game.current_player == 'X':
            action = x_player.get_move(game)
        else:
            action = o_player.get_move(game)

        game.make_move(action, game.current_player)
        state = game.get_state()

        game_history.append((state, action, None, game.current_player))

    reward = None
    if game.current_winner == 'X':
        reward = reward_mapping['Win']
    elif game.current_winner == 'O':
        reward = reward_mapping['Loss']
    elif game.current_winner == 'Draw':
        reward = reward_mapping['Draw']

    for i in range(len(game_history)):
        game_history[i] = game_history[i][:2] + (reward,) + game_history[i][3:]

    return game_history

def trainAI(games_to_train, showGraph = False):
    params = {
        "alpha": 0.5,
        "gamma": 0.9,
        "epsilon": 1.0,
        "min_epsilon": 0.0,
        "epsilon_decay": 0.999
    }

    t = TicTacToe()
    q_learning = QLearning(t,params)

    x_player = Player('X', q_learning)
    o_player = Player('O', q_learning)

    x_wins = [0] * games_to_train
    o_wins = [0] * games_to_train
    ties = [0] * games_to_train

    for i in range(games_to_train):        
        game_history = play_for_training(t, x_player, o_player)

        if t.current_winner == 'X':
            x_wins[i] = 1
        elif t.current_winner == 'O':
            o_wins[i] = 1
        else:
            ties[i] = 1

        for j in reversed(range(len(game_history))):  
            state, action, reward, player = game_history[j]
            next_state, next_action, _, _ = game_history[j+1] if j + 1 < len(game_history) else (None, None, None, None)

            if player == 'X':
                x_player.learning_algorithm.train([(state, action, reward, player)], i)
            else:
                o_player.learning_algorithm.train([(state, action, reward, player)], i)

    try:
        with open('q_table_x.pkl', 'wb') as f:
            pickle.dump(x_player.learning_algorithm.q_table, f)
        with open('q_table_o.pkl', 'wb') as f:
            pickle.dump(o_player.learning_algorithm.q_table, f)
    except Exception as e:
        print("An error occurred while writing the Q-tables: ", e)

    print('x wins:', sum(x_wins))
    print('o wins:', sum(o_wins))
    print('ties:', sum(ties))

    if(showGraph):
        window_size = 10
        x_win_rate = [sum(x_wins[i-window_size:i]) / window_size for i in range(window_size, len(x_wins))]
        o_win_rate = [sum(o_wins[i-window_size:i]) / window_size for i in range(window_size, len(o_wins))]
        ties_rate = [sum(ties[i-window_size:i]) / window_size for i in range(window_size, len(ties))]

        # Generate an array of timestamps (same length as each win_rate array)
        timestamps = list(range(len(x_win_rate)))

        # Fit a linear regression model to each time series and generate fitted lines
        x_fit = np.polyfit(timestamps, x_win_rate, 1)
        o_fit = np.polyfit(timestamps, o_win_rate, 1)
        ties_fit = np.polyfit(timestamps, ties_rate, 1)

        # Generate y-values for fitted lines
        x_trend_line = np.poly1d(x_fit)(timestamps)
        o_trend_line = np.poly1d(o_fit)(timestamps)
        ties_trend_line = np.poly1d(ties_fit)(timestamps)

        plt.figure(figsize=(10, 5))

        # Plot the original data and fitted lines
        plt.plot(x_win_rate, label='X-player win rate')
        plt.plot(o_win_rate, label='O-player win rate')
        plt.plot(ties_rate, label='Tie rate')

        plt.plot(x_trend_line, '--', label='Trend of X-player win rate')
        plt.plot(o_trend_line, '--', label='Trend of O-player win rate')
        plt.plot(ties_trend_line, '--', label='Trend of Tie rate')

        plt.xlabel('Number of games')
        plt.ylabel('Rate')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return x_player, o_player

def playHuman():
    human_player = HumanPlayer('X')
    params = {
        "alpha": 0,
        "gamma": 0,
        "epsilon": 0,
        "min_epsilon": 0,
        "epsilon_decay": 0
    }
    t = TicTacToe()
    q_learning = QLearning(t,params)
    computer_player = Player('O', q_learning)
    

    if human_player.letter == 'X':
        computer_player.q_table = q_learning.load("q_table_o.pkl")
    else:
        computer_player.q_table = q_learning.load("q_table_x.pkl")

    print(computer_player.q_table)

    while True:
        play(t, human_player, computer_player, print_game=True)
        answer = input("Do you want to play again? (Y/N): ")
        if answer.lower() != 'y':
            break

def main():
    parser = argparse.ArgumentParser(description='Tic Tac Toe game.')
    parser.add_argument('--trainAI', help='Train the AI.', action='store_true')
    parser.add_argument('--showGraph', help='Show the training graph.', action='store_true')

    games_to_train = 10000

    args = parser.parse_args()
    if args.trainAI:
        trainAI(games_to_train, args.showGraph)
    else:
        playHuman()

if __name__ == '__main__':
    main()

