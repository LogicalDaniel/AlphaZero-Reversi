import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ValuePolicyNet:

    def __init__(self, network, lr=0.003, device='cpu', path=None):

        self.network = network
        self.lr = lr
        self.device = device

        if path is not None:
            self.network.load_state_dict(torch.load(path))

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=1e-4)

    def predict(self, state):
        """
        return the value and polcy given the cononical state of a game
        """

        self.network.to(self.device)
        state = state.to(self.device)
        value, policy = self.network(state)
        return value, policy
    
    def train(self, states, results, mcts_policies, lr=None):
        """
        train the network given a mini-batch of states, results and MCTS 
        policies
        """

        self.network.train()
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.optimizer.zero_grad()

        values_policies = list(map(lambda x: self.predict(x), states))
        values = torch.cat([vp[0] for vp in values_policies]).view(-1)
        net_policies = torch.cat([vp[1] for vp in values_policies])
        results = torch.tensor(results, dtype=torch.float32).to(self.device)
        mcts_policies = torch.tensor(np.array(mcts_policies, dtype=np.float32), dtype=torch.float32).to(self.device)
        
        value_loss_fn = nn.MSELoss()
        value_loss = value_loss_fn(values, results)
        prob_loss = -(mcts_policies * net_policies).sum(dim=1).mean()
        loss = value_loss + prob_loss

        loss.backward()
        self.optimizer.step()

        return value_loss.item(), prob_loss.item(), loss.item()
    
    def save_model(self, path, verbose=True):
        """
        Save the state dict of self.network to the given filepath.
        """

        torch.save(self.network.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")

    
    def load_model(self, path, verbose=True):
        """
        Load the state of self.network from the given filepath.
        """
        
        self.network.load_state_dict(torch.load(path))
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=1e-4)
        if verbose:
            print(f"Model loaded from {path}")