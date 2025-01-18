from itertools import product
import torch
import numpy as np

from .Game import Game

class Reversi(Game):

    def __init__(self, board_len=8):

        self.DIRECTION = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
        ]

        self.piece_map = {1: "x", -1: "o", 0: "-"}
        self.player_map = {1: "Black", -1: "White"}

        self.player = 1
        self.len = board_len
        self.n_action = self.len * self.len
        self.result = None
        self.last_move = None
        
        # set up of board
        mid = (self.len - 1) // 2
        self.board = [[0 for _ in range(self.len)] for _ in range(self.len)]
        self.board[mid][mid] = self.player
        self.board[mid + 1][mid + 1] = self.player
        self.board[mid][mid + 1] = -self.player
        self.board[mid + 1][mid] = -self.player

        self.frontier = set([(mid - 1, mid - 1), (mid - 1, mid), (mid - 1, mid + 1), (mid - 1, mid + 2),
                             (mid + 2, mid - 1), (mid + 2, mid), (mid + 2, mid + 1), (mid + 2, mid + 2),
                             (mid, mid - 1), (mid + 1, mid - 1), (mid, mid + 2), (mid + 1, mid + 2)])
        
        # Caching the current valid moves
        self.valid_moves_dict = self._get_valid_moves_dict()

    def reset(self):
        self.__init__(board_len=self.len)
    
    # Getters
    def get_player(self):
        return self.player
    
    def get_result(self):
        return self.result
    
    def get_game_state(self):
        return self.board
    
    def get_valid_moves(self):
        return list(self.valid_moves_dict.keys())
    
    def _inside_board(self, x, y):
        return 0 <= x < self.len and 0 <= y < self.len
        
    def _get_update(self, x, y):

        """
        Given a move, get a list of updates. Return an empty list 
        if the move is invalid

        returns:
        updates: the list of positions to be updated (flipped)
        """
        
        updates = []
        for (dx, dy) in self.DIRECTION:

            path = []
            cx, cy = x + dx, y + dy
            while self._inside_board(cx, cy) and self.board[cx][cy] == -self.player:
                path.append((cx, cy))
                cx, cy = cx + dx, cy + dy

            if self._inside_board(cx, cy) and self.board[cx][cy] == self.player:
                updates.extend(path)
        return updates
    
    def _get_valid_moves_dict(self):

        """
        Get a dict of valid moves to corresponding update, cache the result.
        """

        dict = {}
        for (x, y) in self.frontier:
            updates = self._get_update(x, y)
            if len(updates) > 0:
                dict[(x, y)] = updates
        self.valid_moves_dict = dict
        return dict

    def _update_frontier(self, move):

        if move in self.frontier:
            self.frontier.remove(move)
        (x, y) = move
        for dx, dy in self.DIRECTION:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.len and 0 <= ny < self.len:
                if self.board[nx][ny] == 0:
                    self.frontier.add((nx, ny))

    def move(self, move) -> bool:

        if self.result is not None:
            return False
        
        if move not in self.valid_moves_dict.keys():
            return False
        
        # board update
        self.last_move = move
        updates = self.valid_moves_dict[move]
        for (x, y) in updates:
            self.board[x][y] = self.player
        self.board[move[0]][move[1]] = self.player
     
        self.player = -self.player
        self._update_frontier(move)

        self._get_valid_moves_dict()
        if len(self.valid_moves_dict) == 0:
            self.player = -self.player
            self._get_valid_moves_dict()
            if len(self.valid_moves_dict) == 0:
                self.player = -self.player
                self._end_game()

        return True
    
    def _end_game(self):

        white = 0
        black = 0
        for i in range(self.len):
            for j in range(self.len):
                if self.board[i][j] == -1:
                    white += 1
                elif self.board[i][j] == 1:
                    black += 1

        if black > white:
            self.result = 1
        elif black < white:
            self.result = -1
        else:
            self.result = 0
    
    def get_move(self):
        move = None
        while move not in self.valid_moves_dict.keys():
            s = input("Move: ")
            x = ord(s[0]) - ord('a')
            y = int(s[1]) - 1
            move = (y, x)
        return (y, x)
    
    def print_game(self):
        """
        A pretty print of the current board in the terminal
        """
        print(end="  ")
        for i in range(self.len):
            print(chr(i + ord('a')), end=" ")
        print()
        for i in range(self.len):
            print(i + 1, end=" ")
            for j in range(self.len):
                print(self.piece_map[self.board[i][j]], end=" ")
            print()
        print()
    
    def get_cononical_form(self):
        
        state = np.zeros((4, self.len, self.len), dtype=float)
        board = np.array(self.board)
        state[0] = (board == self.player).astype(float)
        state[1] = (board == -self.player).astype(float)
        for x, y in self.get_valid_moves():
            state[2, x, y] = 1.0
        if self.last_move is not None:
            (x, y) = self.last_move
            state[3, x, y] = 1.0
        return torch.tensor(state, dtype=torch.float32)
    
    def get_symmetries(self, state, result, policy):
        
        pi = np.zeros((self.len, self.len), dtype=float)
        for (x, y) in product(range(self.len), repeat=2):
            pi[x, y] = policy[self.action_to_ind((x, y))]
            
        rot_s = torch.rot90(state, k=2)
        flip_s = state.transpose(1, 2)
        rot_flip_s = torch.rot90(flip_s, k=2)

        rot_pi = np.rot90(pi, k=2)
        flip_pi = pi.transpose(1, 0)
        rot_flip_pi = np.rot90(flip_pi, k=2)
        
        flat_rot_pi, flat_flip_pi, flat_rot_flip_pi = [[0] * self.n_action for _ in range(3)]
        for x, y in product(range(self.len), repeat=2):
            flat_flip_pi[self.action_to_ind((x, y))] = flip_pi[x, y]
            flat_rot_pi[self.action_to_ind((x, y))] = rot_pi[x, y]
            flat_rot_flip_pi[self.action_to_ind((x, y))] = rot_flip_pi[x, y]

        states = [state, rot_s, flip_s, rot_flip_s]
        policies = [policy, flat_rot_pi, flat_flip_pi, flat_rot_flip_pi]
        results = [result] * 4

        return zip(states, results, policies)
            
    def action_to_ind(self, act):
        (x, y) = act
        return self.len * x + y
    