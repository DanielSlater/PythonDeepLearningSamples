import random


def new_board(board_width, board_height):
    return tuple(tuple(0 for _ in range(board_height)) for _ in range(board_width))


def apply_move(board_state, move_x, side):
    # find position in which move will settle
    move_y = 0
    for x in board_state[move_x]:
        if x == 0:
            break
        else:
            move_y += 1

    def get_tuples():
        for x in range(len(board_state)):
            if move_x == x:
                temp = list(board_state[x])
                temp[move_y] = side
                yield temp
            else:
                yield board_state[x]

    return tuple(get_tuples())


def available_moves(board_state):
    for x in range(len(board_state)):
        if any(y == 0 for y in board_state[x]):
            yield x


def has_winning_line(line, winning_length):
    count = 0
    last_side = 0
    for x in line:
        if x == last_side:
            count += 1
            if count == winning_length:
                return last_side
        else:
            count = 1
            last_side = x
    return 0


def has_winner(board_state, winning_length=4):
    board_width = len(board_state)
    board_height = len(board_state[0])

    # check rows
    for x in range(board_width):
        winner = has_winning_line(board_state[x], winning_length)
        if winner != 0:
            return winner
    # check columns
    for y in range(board_height):
        winner = has_winning_line((i[y] for i in board_state), winning_length)
        if winner != 0:
            return winner

    # check diagonals
    diagonals_start = -(board_width - winning_length)
    diagonals_end = (board_width - winning_length)
    for d in range(diagonals_start, diagonals_end):
        winner = has_winning_line(
            (board_state[i][i + d] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)
        if winner != 0:
            return winner
    for d in range(diagonals_start, diagonals_end):
        winner = has_winning_line(
            (board_state[i][board_height - i - d - 1] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)
        if winner != 0:
            return winner

    return 0  # no one has won, return 0 for a draw


def play_game(plus_player_func, minus_player_func, board_width=7, board_height=6, winning_length=4, log=False):
    board_state = new_board(board_width, board_height)
    player_turn = 1

    while True:
        _avialable_moves = list(available_moves(board_state))
        if len(_avialable_moves) == 0:
            # draw
            if log:
                print("no moves left, game ended a draw")
            return 0.
        if player_turn > 0:
            move = plus_player_func(board_state, 1)
        else:
            move = minus_player_func(board_state, -1)

        if move not in _avialable_moves:
            # if a player makes an invalid move the other player wins
            if log:
                print("illegal move ", move)
            return -player_turn

        board_state = apply_move(board_state, move, player_turn)
        if log:
            print(board_state)

        winner = has_winner(board_state, winning_length)
        if winner != 0:
            if log:
                print("we have a winner, side: %s" % player_turn)
            return winner
        player_turn = -player_turn


def random_player(board_state, _):
    moves = list(available_moves(board_state))
    return random.choice(moves)
