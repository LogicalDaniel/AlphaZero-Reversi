import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

board_len = 6

n_sim = 200
c_puct = 5

buffer_size = 10000

batch_size = 256
n_epoch = 5
lr = 0.002

n_game = 1