# AlphaZero Reversi
This project is a lightweight implementation of the AlphaZero algorithm, specifically tailored for the game of Reversi. The motivation is to help me better understand reinforcement learning (RL) concepts and gain hands-on experience by implementing these algorithms from scratch.

## Structure

### Directories
- **`game/`**  
  Contains the implementation of the board game logics and interface:
  - `Game.py`: An interface of board games.
  - `Reversi.py`: The logic of Reversi game implementing the interface.

- **`model/`**  
  Stores saved model checkpoints and trained models.

### Core Files
- **`AlphaZero.py`**  
  The core implementation of the AlphaZero algorithm, along with the training pipeline.

- **`MCTS.py`**  
  Implementation of Monte Carlo Tree Search (MCTS).

- **`ValuePolicyNet.py`**  
  A wrapper for managing the neural networks.

- **`networks.py`**  
  Defines the neural network architectures used for the policy and value functions.

## Usage
The required packages can be found in requirement.txt and installed.

Use train.py and run.py for training and playing against the models respectively.
pygame can be optionally installed if you need a GUI to play with the models.

## Acknowledgement
This project draws inspirations from these repositories:

- [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)

including the training pipelines and the structure of the project. I extend my gratitude to the authors of these repositories.
