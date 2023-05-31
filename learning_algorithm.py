from abc import ABC, abstractmethod

class LearningAlgorithm(ABC):
    def __init__(self, game, params):
        self.game = game
        self.params = params
        
    @abstractmethod
    def train(self, game_history):
        pass

    @abstractmethod
    def predict(self, game_state):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass