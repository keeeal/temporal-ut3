
import re, random, torch
from copy import deepcopy

class ModelPlayer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_action(self, game, epsilon=0):
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

def ask(question):
    hint = ''
    while True:
        answer = input(question + hint + '\n').lower()
        if answer == 'y' or answer == 'yes': return True
        if answer == 'n' or answer == 'no': return False
        hint = ' (y/n)'

def play(params=None, display=True):
    from game import Game
    game = Game()
    human = HumanPlayer()

    if params:
        from model import Model
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = Model().to(device)
        if torch.cuda.device_count() >= 1: model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(params))
        opponent = ModelPlayer(model, device)
    else:
        opponent = RandomPlayer()

    # choose starting player
    player = human if ask('Would you like to go first?') else opponent

    while True:
        if display: game.display()
        action = player.get_action(game)
        game.execute_move(action)
        end_value = game.is_over()

        if end_value:
            if display: game.display()
            if player is human:
                print('You won!')
            else:
                print('You lost.')
            break

        game.flip()
        player = opponent if player is human else human

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default=None)
    play(**vars(parser.parse_args()))
