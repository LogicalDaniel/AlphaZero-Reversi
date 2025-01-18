from itertools import product

class Game:

    def reset(self):
        """
        reset the state of the game.
        """
        raise NotImplementedError
    
    def get_move(self):
        """
        get the move from the terminal until the provided 
        move is valid.
        """
        raise NotImplementedError

    def get_valid_moves(self):
        """
        return all valids move at the current game status.
        """
        raise NotImplementedError

    def move(self, move) -> bool:
        """
        play the provided move and update game status, 
        return whether the move is successful
        """
        raise NotImplementedError
    
    def get_player(self):
        """
        return the player to play the next move.
        """
        raise NotImplementedError
    
    def get_result(self):
        """
        return the result of the game, None if the game has not ended.
        """
        raise NotImplementedError
    
    def get_cononical_form(self):
        """
        return the cononical representation of the current game state as the 
        input of policy net.
        """
    
    def action_to_ind(self, ind):
        """
        bijective map from actions to indices.
        """

    def print_game(self):
        """
        print the current game status in the terminal.
        """
        raise NotImplementedError
    
    def get_symmetries(self, state, result, policy):
        """
        Get the equivalent games to augment training data.
        """
        raise NotImplementedError
