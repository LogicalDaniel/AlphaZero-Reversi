from ValuePolicyNet import ValuePolicyNet
from MCTS import Player
from game import Game
from copy import deepcopy

import random
import torch
from tqdm import tqdm

class AlphaZero:

    def __init__(self, game: Game, policy_net: ValuePolicyNet,
                 c_puct=5, n_sim=100,
                 n_game=1, n_epoch=10,
                 buffer_size=10000, batch_size=256,
                 lr=0.03):
        
        self.game = game
        self.policy_net = policy_net
        self.best_net = deepcopy(policy_net)
        self.policy_net.save_model("./model/tmp_curr.model")
        self.policy_net.save_model("./model/tmp_best.model")

        self.lr = lr
        self.lr_mult = 1.0

        self.c_puct = c_puct
        self.n_sim = n_sim
        self.n_game = n_game
        self.n_epoch = n_epoch
        self.batch_size = batch_size

        self.replay_buffer = []
        self.buffer_size = buffer_size

        self.player = Player(self.policy_net, self.c_puct, self.n_sim)

    def _extend_buffer(self, data):
        """
        extend the replay buffer, evict old data if the buffer is full.
        """
        
        self.replay_buffer.extend(data)
        pos = len(self.replay_buffer) - self.buffer_size
        if pos > 0:
            self.replay_buffer = self.replay_buffer[pos:]
    
    def _collect_data(self):
        """
        self-play for n_game of game. collect tuple including data from all games, 
        add the data to the replay buffer.
        """
        
        data, aug_data = [], []
        for _ in range(self.n_game):
            data.extend(self.player.self_play(self.game))
        for s, r, p in data:
            aug_data.extend(self.game.get_symmetries(s, r, p))

        self._extend_buffer(aug_data)

    def _sample(self, batch_size):
        """
        sample a mini-batch from the replay buffer.
        """
        
        return random.sample(self.replay_buffer, batch_size) 
    
    def _train_net(self):
        """
        train the value-policy network based on a mini-batch.
        """
        kl_bound = 0.02

        if self.batch_size > len(self.replay_buffer):
            print("need to collect more data")
            return
        
        data = self._sample(self.batch_size)
        states, results, policies = zip(*data)
        old_p = torch.cat([(self.policy_net.predict(s))[1] for s in states])
        v_loss_avg, p_loss_avg, loss_avg = 0, 0, 0

        for epoch in range(self.n_epoch):
          
            v_loss, p_loss, loss = self.policy_net.train(states, results, policies, self.lr * self.lr_mult)
            v_loss_avg += (v_loss - v_loss_avg) / (epoch + 1)
            p_loss_avg += (p_loss - p_loss_avg) / (epoch + 1)
            loss_avg += (loss - loss_avg) / (epoch + 1)

            self.policy_net.network.eval()
            new_p = torch.cat([(self.policy_net.predict(s))[1] for s in states])
            
            kl = (torch.exp(old_p) * (old_p - new_p)).sum(dim=1).mean()
            if kl > kl_bound * 4:
                print(f"early stopping due to bad KL divergence > {kl_bound}")
                break

        # adaptively adjust the learning rate
        if kl > kl_bound * 2 and self.lr_mult > 0.1:
            self.lr_mult /= 1.5
        elif kl < kl_bound / 2 and self.lr_mult < 10:
            self.lr_mult *= 1.5
            
        print(("epochs: {}, "
               "value_loss: {:.3f}, "
               "policy_loss: {:.3f}, "
               "loss: {:.3f}, "
               "KL divergence: {:.3f}"
               ).format(epoch, 
                        v_loss_avg, 
                        p_loss_avg, 
                        loss_avg,
                        kl))
        
    def _compare_net(self, n_iter):
        """
        play games between players using the old and new value-policy nets.
        reject the new network if its performance does not pass a pre-defined threshold.
        """

        n_test = 20
        ratio = 0.6
        best_player = Player(self.best_net, c_puct=5, n_sim=50)
        curr_player = Player(self.policy_net, c_puct=5, n_sim=50)
        curr_score = 0

        self.best_net.network.eval()
        self.policy_net.network.eval()
        
        print("Start comparing current net with best net")

        for _ in tqdm(range(n_test // 2), desc="Best Player against Current Player"):
            result = best_player.play_against(curr_player, self.game)
            curr_score += (1 - result) / 2

        for _ in tqdm(range(n_test // 2), desc="Current Player against Best Player"):
            result = curr_player.play_against(best_player, self.game)
            curr_score += (1 + result) / 2

        print(f"The current net score {curr_score}/{n_test} against the best net.")
        if curr_score / n_test >= ratio:
            print("new best policy net found")
            self.policy_net.save_model(f"./model/checkpoint_iter_{n_iter}.model")
            self.policy_net.save_model("./model/tmp_best.model")
            self.best_net.load_model("./model/tmp_best.model", verbose=False)
            self.player.change_net(self.best_net)
            self.player.mcts.reset()
        else:
            print("current net rejected")
            
    def train(self):
        """
        entry for training of value policy net.
        """

        n_iter = 1
        warm_up_iter = 5
        check_freq = 5
        
        # Collect more data to prevent steep change in network
        print("=== Warm-up ===")
        for _ in tqdm(range(warm_up_iter), desc="Warm-up iters"):
            self._collect_data()
            
        while True:
            print(f"=== Iteration {n_iter} ===")
            # training policy-value net
            self._collect_data()
            
            n_batch = len(self.replay_buffer) // (2 * self.batch_size) + 1
            n_batch = 8 if n_batch >= 8 else n_batch
            for _ in range(n_batch):
                self._train_net()
            self.policy_net.save_model("./model/tmp_curr.model", verbose=False)
                
            # comparing current net with best net
            if n_iter % check_freq == 0:
                self._compare_net(n_iter)

            n_iter += 1




         

        
        
        


    


