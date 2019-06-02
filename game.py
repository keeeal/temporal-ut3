
from itertools import product
import numpy as np

class Game:
    '''
    Macro board data:
      1.=X, -1.=O, 0.=empty, -0.=blocked
    Pieces board data:
      1.=X, -1.=O, 0.=empty
    '''

    def __init__(self, n=3, move=0, pieces=None, macro=None):
        self.n, self.move = n, move
        self.pieces = np.array(pieces) if pieces else np.zeros((n**2, n**2))
        self.macro = np.array(macro) if macro else np.zeros((n, n))
        self.eps = np.finfo(self.pieces.dtype).eps

    def __getitem__(self, index):
        return self.pieces[index]

    def flip(self):
        self.macro = np.where(self.macro, -self.macro, self.macro)
        self.pieces = np.where(self.pieces, -self.pieces, self.pieces)
        return self

    def get_state(self, player=1):
        state = self.pieces[np.newaxis,:,:]
        return player*state.astype(np.float32)

    def get_symmetries(self, player=1):
        syms, state = [], self.get_state(player)
        for rot in range(4):
            for flip in True, False:
                s = np.rot90(state, rot, (-2,-1))
                if flip: s = np.flip(s, -1)
                syms.append(s)
        return syms

    def get_microboard(self, index):
        return self[tuple(slice(self.n*i, self.n*(i+1)) for i in index)]

    def get_valid_actions(self):
        moves = []
        for u in product(range(self.n), range(self.n)):
            if not self.macro[u] and 0 < np.copysign(1, self.macro[u]):
                for move in product(*(range(self.n*i, self.n*(i+1)) for i in u)):
                    if self.pieces[move] == 0:
                        moves.append(move)
        return moves

    def is_win(self, player=1, board=None):
        if board is None: board = self.macro
        for i in range(len(board)):
            if (board[i,:] == player).all(): return True
            if (board[:,i] == player).all(): return True
        if (board.diagonal() == player).all(): return True
        if (board[::-1].diagonal() == player).all(): return True
        return False

    def is_full(self, board=None):
        if board is None: board = self.macro
        return board.all()

    def is_over(self, player=1, board=None):
        if board is None: board = self.macro
        if self.is_win(player, board): return 1.
        if self.is_win(-player, board): return self.eps
        if self.is_full(board): return .5
        return 0.

    def execute_move(self, move, player=1):
        _u = tuple(int(i/self.n) for i in move)
        _v = tuple(int(i%self.n) for i in move)
        assert self.pieces[move] == 0
        self.pieces[move] = player
        uboard = self.get_microboard(_u)

        if self.is_full(uboard):
            self.macro[_u] = self.eps

        for player in 1, -1:
            if self.is_win(player, uboard):
                self.macro[_u] = player

        for u in product(range(self.n), range(self.n)):
            if not self.macro[u]:
                self.macro[u] = 0. if self.macro[_v] or u == _v else -0.

        self.move += 1
        return self

    def display(self, indent='  '):
        print('')
        print('Move number', self.move)
        print('')
        print(indent + '   0 | 1 | 2 ‖ 3 | 4 | 5 ‖ 6 | 7 | 8')
        print('')
        for n, row in enumerate(self.pieces if self.move%2 else -self.pieces):
            if n:
                if n % 3:
                    sep = '---+---+---'
                    print(indent + '- ' + sep + '‖' + sep + '‖' + sep)
                else:
                    sep = '==========='
                    print(indent + '= ' + sep + '#' + sep + '#' + sep)
            row = ' ‖ '.join(' | '.join(map(str, map(int, row[i:i+3]))) for i in range(0, len(row), 3))
            print(indent + str(n) + '  ' + row.replace('-1','O').replace('1','X').replace('0','.'))
        print('')
