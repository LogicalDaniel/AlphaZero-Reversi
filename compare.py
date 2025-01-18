from networks import ReversiNet, DefaultReversiNet
from ValuePolicyNet import ValuePolicyNet
from MCTS import Player
from game import Reversi
import config

import torch
from tqdm import tqdm

def compare(path_a=None, path_b=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if path_a is not None:
        network_a = ReversiNet(config.board_len)
    else:
        network_a = DefaultReversiNet(config.board_len)

    if path_b is not None:
        network_b = ReversiNet(config.board_len)
    else:
        network_b = DefaultReversiNet(config.board_len)

    net_a = ValuePolicyNet(network_a, device=device, path=path_a)
    net_b = ValuePolicyNet(network_b, device=device, path=path_b)

    net_a.network.eval()
    net_b.network.eval()

    game = Reversi(board_len=6)
    player_a = Player(net=net_a, c_puct=5, n_sim=200)
    player_b = Player(net=net_b, c_puct=5, n_sim=200)

    n_test = 10

    a_score = 0

    for _ in tqdm(range(n_test // 2)):
        result = player_a.play_against(player_b, game)
        a_score += (1 + result) / 2

    for _ in tqdm(range(n_test // 2)):
        result = player_b.play_against(player_a, game)
        a_score += (1 - result) / 2

    print(f"player A scores {a_score}/{n_test}")
    print(f"player B scores {n_test - a_score}/{n_test}")


if __name__ == "__main__":

    # the path of models for comparison
    path_a = None
    path_b = "./model/Reversi_6_6_ver3.model"

    compare(path_a, path_b)
