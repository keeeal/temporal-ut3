
import re, random, torch
from copy import deepcopy

class ModelPlayer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_action(self, game, epsilon=0):
        self.model.eval()
        return self.get_action_and_value(game, epsilon)[0]

    def get_action_and_value(self, game, epsilon=0):
        actions = game.get_valid_actions()
        random.shuffle(actions)
        next_states = []

        # sometimes make a random move
        if random.random() < epsilon:
            actions = [random.choice(actions)]

        # get all next states
        for action in actions:
            g = deepcopy(game).execute_move(action)
            end_value = g.is_over()
            if end_value == 1: return action, end_value
            next_states.append(g.flip().get_state())

        # get values for all next states, find max
        next_states = torch.tensor(next_states).to(self.device)
        next_values = 1 - self.model(next_states)
        max_value, i = next_values.max(0)
        action = actions[i]

        return action, max_value

class RandomPlayer:
    def get_action(self, game):
        return random.choice(game.get_valid_actions())

class GreedyPlayer:
    def get_action(self, game):
        actions = game.get_valid_actions()
        random.shuffle(actions)

        for action in actions:
            g = deepcopy(game).execute_move(action)
            u = tuple(int(i/game.n) for i in action)
            if g.is_win(board=g.get_microboard(u)): return action

        return random.choice(actions)

class HumanPlayer:
    def get_action(self, game):
        actions = game.get_valid_actions()
        print('\nValid moves:')
        print(*actions)

        while True:
            action = re.split('[, ]{1,}', input())
            action = tuple(int(i) for i in action)
            if action in actions: break
            print('Invalid input.')

        return action
