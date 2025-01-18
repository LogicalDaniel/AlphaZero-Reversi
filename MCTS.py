import copy
from ValuePolicyNet import ValuePolicyNet
from game import Game

import numpy as np
import torch

class Node:
    
    def __init__(self, prior=1.0, parent=None, player=None):
        """
        Create a node from the current game state.
        
        self.n: the total number of visit to the node.
        self.v: the action value of the node.
        self.player: the player of the currend node of the game
        self.prior: the probability that the current node is selected given 
        that its parent is selected.
        
        self.children: a dict from action (move) to Node.
        self.parent: the parent of the current node
        """
    
        self.children = {}
        self.parent = parent
        self.n = 0
        self.v = 0
        self.prior = prior
        self.player = player

    def _get_eval(self, c_puct):
        """
        return the upper confidence bound of the node on the perspective of 
        parent.
        """

        u = np.sqrt(self.parent.n + 1) / (1 + self.n)
        rev = 1 if self.player == self.parent.player else -1
        if self.player == None:
            rev = 0
        return rev * self.v + self.prior * c_puct * u

    def _select_best_child(self, c):
        """
        return the (move, node) pair with largest upper confidence bound.
        """

        return max(self.children.items(), key=lambda pair: pair[1]._get_eval(c))
    

class MCTS:

    def __init__(self, net: ValuePolicyNet, c_puct=5, n_sim=100):
        """
        self.c_puct: the constant for Upper Confidence Bound for Trees (UCT)
        self.eps: epsilon used in eps-greedy approach
        self.n_sim: number of iteration
        """

        self.c_puct = c_puct
        self.n_sim = n_sim

        self.net = net
        self.root = Node(player=1)

    def _update_nodes(self, nodes, value):
        """
        Update all the nodes on the path of a simulation.
        """
        
        for (node, p) in nodes:
            node.n += 1
            node.v += (p * value - node.v) / node.n

    def _mask_invalid_moves(self, game, policy):
        """
        mask all invalid moves and generate a new probability distribution of
        actions.
        """
        
        # mask invalid moves entry to 0
        valid_moves = game.get_valid_moves()
        move_indices = [game.action_to_ind(move) for move in valid_moves]
        new_policy = torch.zeros_like(policy)
        new_policy[move_indices] = torch.exp(policy[move_indices])

        # re-normalize to a valid probability distribution
        sum = new_policy.sum()
        if sum != 0:
            return new_policy / sum
        else:
            return torch.zeros_like(policy)
        
    def _predict(self, game: Game):
        """
        predict the action value of the current node and generate a policy for the 
        next action.
        """

        state = game.get_cononical_form()
        value, policy = self.net.predict(state)
        policy = policy.view(-1)
        policy = self._mask_invalid_moves(game, policy)
        return value, policy

    def _simulate(self, game: Game):
        """
        simulate a game from the current game state, update the corresponding nodes.
        """

        game = copy.deepcopy(game)
        curr_node = self.root
        path = [(curr_node, game.get_player())]

        while True:
            # not expanded
            if len(curr_node.children) == 0:
                curr_node.player = game.get_player()
                break

            (move, curr_node) = curr_node._select_best_child(self.c_puct)
            game.move(move)
            path.append((curr_node, game.get_player()))
            if game.result != None:
                break

        if game.result != None:
            value = game.get_result()
        else:
            # predict value
            value, policy = self._predict(game)
            valid_moves = game.get_valid_moves()
            value *= game.get_player()
            move_indices = [game.action_to_ind(move) for move in valid_moves]
            # expand
            for move, p in zip(valid_moves, policy[move_indices]):
                curr_node.children[move] = Node(prior=p, parent=curr_node)

        self._update_nodes(path, value)

    def get_policy(self, game: Game, temp=1):
        """
        temp: temperature in range (0, 1] controlling the level of exploitation.

        simulate from the current game status for n_sim times, return the 
        corresponding policy.
        """

        for _ in range(self.n_sim):
            self._simulate(game)
        
        moves = [act for act, _ in self.root.children.items()]
        visits = np.array([node.n for _, node in self.root.children.items()])
        
        scaled_visits = visits ** (1.0 / temp)
        probs = scaled_visits / np.sum(scaled_visits)

        return moves, probs
    
    def move(self, move):
        """
        change the root by performing the given action.
        """

        self.root = self.root.children[move]

    def reset(self, complete=True):
        """
        complete: whether the data from the tree are retained.

        Reset the MCTS by re-initialize the tree or setting the current root to 
        the first root.
        """

        if complete:
            self.root = Node(player=1)
            return
        while self.root.parent != None:
            self.root = self.root.parent


class Player:

    def __init__(self, net: ValuePolicyNet, c_puct=5, n_sim=100):

        self.mcts = MCTS(net, c_puct, n_sim)

    def move(self, game, move):
        
        # simulate once to ensure expansion of current node.
        self.mcts._simulate(game)
        self.mcts.move(move)
        game.move(move)
        
    def change_net(self, net):
        """
        change the value policy net for the Monte Carlo Search Tree.
        """

        self.mcts.net = net

    def _find_next_move(self, game: Game, is_self_play=True, temp=1.0):
        """
        find the next move using the mcts. return the policy followed and the 
        policy used.
        """

        moves, probs = self.mcts.get_policy(game, temp) 
        
        if is_self_play:
            # encourage exploration by introducing dirichlet noise
            noises = np.random.dirichlet(0.3 * np.ones(probs.shape))
            mixed_probs = probs * 0.75 + noises * 0.25
            mixed_probs /= mixed_probs.sum()
            index = np.random.choice(len(moves), p=mixed_probs)
        else:
            index = np.random.choice(len(moves), p=probs)
        
        indices = [ind for ind in map(lambda x: game.action_to_ind(x), moves)]
        policy = np.zeros(game.n_action)
        policy[indices] = probs 

        return moves[index], policy
    
    def self_play(self, game: Game):
        """
        play a game with the player itself, return the 
        (states, results, policies) tuples.
        """
        
        game.reset()
        self.mcts.reset(complete=False)
        policies, states, players = [], [], []

        while True:
            move, policy = self._find_next_move(game)
            players.append(game.get_player())
            states.append(game.get_cononical_form())
            policies.append(policy)
            self.move(game, move)
            if game.get_result() is not None:
                break

        result = game.get_result()
        results = [result * p for p in players]
        return zip(states, results, policies)
    
    def play_against(self, opponent, game: Game):
        """
        play a game against a given opponent. return the result of the 
        game.
        """
        
        game.reset()
        self.mcts.reset()
        opponent.mcts.reset()
        
        while game.get_result() is None:
            move_a, _ = self._find_next_move(game, is_self_play=False, temp=0.1)
            move_b, _ = opponent._find_next_move(game, is_self_play=False, temp=0.1)
            move = move_a if game.get_player() == 1 else move_b
            
            self.mcts.move(move)
            opponent.mcts.move(move)
            game.move(move)

        return game.get_result()
    
