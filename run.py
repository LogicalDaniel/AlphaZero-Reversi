from MCTS import Player
from game import Reversi
from ValuePolicyNet import ValuePolicyNet
from networks import ReversiNet

import config

def run(player: Player, game, first=True):
    """
    Main game loop: handle events, update game state, render the board.
    """

    import pygame
    import sys
    
    # constants
    CELL_SIZE = 80
    WINDOW_SIZE = BOARD_SIZE * CELL_SIZE
    FPS = 30

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHT_GREEN = (90, 200, 90)
    
    # helper functions
    def draw_board(surface, game):
        """
        Draws an 8x8 board using two-tone squares.
        Highlights squares if needed based on board_states.
        """
            
        board_states = game.get_game_state()
        surface.fill(LIGHT_GREEN)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                        
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, BLACK, rect, 1)

                if board_states[row][col] != 0:
                    piece_color = BLACK if board_states[row][col] == 1 else WHITE
                    x = col * CELL_SIZE + CELL_SIZE // 2
                    y = row * CELL_SIZE + CELL_SIZE // 2
                    pygame.draw.circle(surface, piece_color, (x, y), CELL_SIZE // 4)

    def get_move_from_mouse(pos):
            """
            Convert an (x, y) pixel coordinate to a (row, col) on the board.
            """
            x, y = pos
            row = y // CELL_SIZE
            col = x // CELL_SIZE
            return (row, col)
    
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Reversi")

    pygame.init()
    allow_click = first
    user = 1 if first else -1
    player.mcts.net.network.eval()
    clock = pygame.time.Clock()
    
    # main play loop
    while True:
        clock.tick(FPS)
        
        if allow_click is False:
            move, _ = player._find_next_move(game, is_self_play=False, temp=0.05)
            player.move(game, move)
            if (game.get_player() == user):
                allow_click = True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                move = get_move_from_mouse(mouse_pos)
                if move not in game.get_valid_moves():
                    continue
                player.move(game, move)
                if (game.get_player() == -user):
                    allow_click = False
            
        # update the board
        draw_board(screen, game)
        pygame.display.flip()

        if game.get_result() is not None:
            print("Game ended with board status:")
            game.print_game()
            print(f"player {game.get_result()} wins")
            pygame.quit()
            sys.exit()

def run_without_render(player, game, first=True):
    
    user = 1 if first else -1
    while game.get_result() is None:

        game.print_game()
        if game.get_player() == user:
            print("Your turn")
            move = game.get_move()
        else:
            print("opponent's turn")
            move, _ = player._find_next_move(game, is_self_play=False, temp=0.05)
        player.move(game, move)

    print(f"player {game.get_result()} wins")
            

if __name__ == "__main__":

    path = "./model/Reversi_6_6_ver3.model"
    BOARD_SIZE = config.board_len
    
    network = ReversiNet(board_len=BOARD_SIZE)
    net = ValuePolicyNet(network, device=config.device, path=path)
    game = Reversi(board_len=BOARD_SIZE)
    player = Player(net, c_puct=5, n_sim=800)

    # run(player, game, first=False)
    run_without_render(player, game, first=False)


