import torch
import torch.nn as nn
import torch.nn.functional as F

class DefaultReversiNet(nn.Module):

    def __init__(self, board_len):
        super(DefaultReversiNet, self).__init__()
        self.board_len = board_len
        self.n_action = self.board_len * self.board_len

         # a dummy parameter so there's something to optimize
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        x = x.view(-1, 4, self.board_len, self.board_len)
        batch_size = x.size(0)
        val = torch.zeros((batch_size, 1), device=x.device)

        logits = torch.zeros((batch_size, self.n_action), device=x.device)
        pi = F.log_softmax(logits, dim=1)

        return val, pi

class ReversiNet(nn.Module):

    def __init__(self, board_len):
        super(ReversiNet, self).__init__()
        self.board_len = board_len
        self.n_action = self.board_len * self.board_len
        
        # preprocessing
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)

        self.lin1 = nn.Linear(16 * self.n_action, 256)
        self.lin2 = nn.Linear(256, 128)
        
        # policy net
        self.pi_lin = nn.Linear(128, self.n_action)

        # value net
        self.val_lin1 = nn.Linear(128, 64)
        self.val_lin2 = nn.Linear(64, 1)

    def forward(self, x):
        
        #preprocessing
        x = x.view(-1, 4, self.board_len, self.board_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.view(-1, 16 * self.n_action)
        x = F.dropout(F.relu(self.lin1(x)), p=0.3, training=self.training)
        x = F.dropout(F.relu(self.lin2(x)), p=0.3, training=self.training)
        
        # policy
        logits = self.pi_lin(x)
        x_pi = F.log_softmax(logits, dim=1)
        
        # value
        x_val = F.relu(self.val_lin1(x))
        x_val = torch.tanh(self.val_lin2(x_val))

        return x_val, x_pi
