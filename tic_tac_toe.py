import random
import itertools


def new_board():
    return ((0, 0, 0),
            (0, 0, 0),
            (0, 0, 0))


def apply_move(board_state, move, side):
    move_x, move_y = move

    def get_tuples():
        for x in range(3):
            if move_x == x:
                temp = list(board_state[x])
                temp[move_y] = side
                yield tuple(temp)
            else:
                yield board_state[x]

    return tuple(get_tuples())


def available_moves(board_state):
    for x, y in itertools.product(range(3), range(3)):
        if board_state[x][y] == 0:
            yield (x, y)


def has_3_in_a_line(line):
    return all(x == -1 for x in line) | all(x == 1 for x in line)


def has_winner(board_state):
    # check rows
    for x in range(3):
        if has_3_in_a_line(board_state[x]):
            return board_state[x][0]
    # check columns
    for y in range(3):
        if has_3_in_a_line([i[y] for i in board_state]):
            return board_state[0][y]

    # check diagonals
    if has_3_in_a_line([board_state[i][i] for i in range(3)]):
        return board_state[0][0]
    if has_3_in_a_line([board_state[2 - i][i] for i in range(3)]):
        return board_state[0][2]

    return 0  # no one has won, return 0 for a draw


def play_game(plus_player_func, minus_player_func, log=False):
    board_state = new_board()
    player_turn = 1

    while True:
        available_moves = list(available_moves(board_state))
        if len(available_moves) == 0:
            # draw
            if log:
                print("no moves left, game ended a draw")
            return 0.
        if player_turn > 0:
            move = plus_player_func(board_state, 1)
        else:
            move = minus_player_func(board_state, -1)

        if move not in available_moves:
            # if a player makes an invalid move the other player wins
            if log:
                print("illegal move ", move)
            return -player_turn

        board_state = apply_move(board_state, move, player_turn)
        if log:
            print(board_state)

        winner = has_winner(board_state)
        if winner != 0:
            if log:
                print("we have a winner, side: %s" % player_turn)
            return winner
        player_turn = -player_turn


def random_player(board_state, _):
    moves = list(available_moves(board_state))
    return random.choice(moves)
