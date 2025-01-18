from AlphaZero import AlphaZero
from game import Reversi
from networks import ReversiNet
from ValuePolicyNet import ValuePolicyNet

import config

if __name__ == "__main__":
    
    game = Reversi(board_len=config.board_len)

    # train the network loaded from path. set path to None if training from scratch.
    path = None
    
    network = ReversiNet(board_len=config.board_len)
    value_policy_net = ValuePolicyNet(network,
                                    lr=config.lr, 
                                    device=config.device,
                                    path=path)
    alphaZero = AlphaZero(game=game, policy_net=value_policy_net,
                        c_puct=config.c_puct,
                        n_sim=config.n_sim,
                        n_game=config.n_game,
                        n_epoch=config.n_epoch,
                        batch_size=config.batch_size,
                        buffer_size=config.buffer_size,
                        lr=config.lr)

    alphaZero.train()