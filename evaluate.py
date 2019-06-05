
from copy import deepcopy
from progress.bar import Bar

from model import Model
from game import Game
from player import *

def load_player(params=None):
    model = Model()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        if torch.cuda.device_count() >= 1:
            model = torch.nn.DataParallel(model)
        if params:
            model.load_state_dict(torch.load(params))
    else:
        device = torch.device('cpu')
        if params:
            model.load_state_dict(torch.load(params, map_location='cpu'))

    return ModelPlayer(model, device)

def ask(question):
    hint = ''
    while True:
        answer = input('\n' + question + hint + '\n').lower()
        if answer == 'y' or answer == 'yes': return True
        if answer == 'n' or answer == 'no': return False
        hint = ' (y/n)'

def evaluate(player_i, player_j, games=2, display=False):
    if player_i is player_j: player_j = deepcopy(player_j)
    score = {player_i:0, player_j:0, None:0}

    for n in Bar('Evaluating').iter(range(games)):
        game = Game()

        # choose starting player
        if isinstance(player_i, HumanPlayer):
            player = player_i if ask('Would you like to go first?') else player_j
            display = True
        elif isinstance(player_j, HumanPlayer):
            player = player_j if ask('Would you like to go first?') else player_i
            display = True
        else:
            player = player_i if n < games/2 else player_j

        while True:
            if display: game.display()
            action = player.get_action(game)
            game.execute_move(action)
            end_value = game.is_over()

            if end_value:
                if display: game.flip().display()
                winner = player if end_value == 1 else None
                score[winner] += 1
                break

            game.flip()
            player = player_j if player is player_i else player_i

    return score[player_i], score[None], score[player_j]

def make_player(identifier):
    if identifier.endswith('.params'):
        return load_player(identifier)
    if identifier == 'model':
        return load_player()
    if identifier == 'random':
        return RandomPlayer()
    if identifier == 'greedy':
        return GreedyPlayer()
    if identifier == 'human':
        return HumanPlayer()
    raise ValueError('Unknown player identifier:' + identifier)

def main(player_1, player_2, games=1, display=False):
    player_1 = make_player(player_1)
    player_2 = make_player(player_2)
    print(evaluate(player_1, player_2, games, display))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--player-1', '-p1', required=True)
    parser.add_argument('--player-2', '-p2', required=True)
    parser.add_argument('--games', '-n', type=int, default=1)
    parser.add_argument('--display', action='store_true')
    main(**vars(parser.parse_args()))
