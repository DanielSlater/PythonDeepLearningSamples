from tic_tac_toe import available_moves, apply_move, has_winner
import sys


def score_line(line):
    minus_count = line.count(-1)
    plus_count = line.count(1)
    if minus_count + plus_count < 3:
        if minus_count == 2:
            return -1
        elif plus_count == 2:
            return 1
    return 0


def evaluate(board_state):
    score = 0
    for x in range(3):
        score += score_line(board_state[x])
    for y in range(3):
        score += score_line([i[y] for i in board_state])
    # diagonals
    score += score_line([board_state[i][i] for i in range(3)])
    score += score_line([board_state[2 - i][i] for i in range(3)])

    return score


def min_max(board_state, side, max_depth):
    best_score = None
    best_score_move = None

    moves = list(available_moves(board_state))
    if not moves:
        # this is a draw
        return 0, None

    for move in moves:
        new_board_state = apply_move(board_state, move, side)
        winner = has_winner(new_board_state)
        if winner != 0:
            return winner * 10000, move
        else:
            if max_depth <= 1:
                score = evaluate(new_board_state)
            else:
                score, _ = min_max(new_board_state, -side, max_depth - 1)
            if side > 0:
                if best_score is None or score > best_score:
                    best_score = score
                    best_score_move = move
            else:
                if best_score is None or score < best_score:
                    best_score = score
                    best_score_move = move
    return best_score, best_score_move


def min_max_alpha_beta(board_state, side, max_depth, alpha=-sys.float_info.max, beta=sys.float_info.max):
    best_score_move = None
    moves = list(available_moves(board_state))
    if not moves:
        return 0, None

    for move in moves:
        new_board_state = apply_move(board_state, move, side)
        winner = has_winner(new_board_state)
        if winner != 0:
            return winner * 10000, move
        else:
            if max_depth <= 1:
                score = evaluate(new_board_state)
            else:
                score, _ = min_max_alpha_beta(new_board_state, -side, max_depth - 1, alpha, beta)

        if side > 0:
            if score > alpha:
                alpha = score
                best_score_move = move
        else:
            if score < beta:
                beta = score
                best_score_move = move
        if alpha >= beta:
            break

    return alpha if side > 0 else beta, best_score_move


def min_max_player(board_state, side):
    return min_max(board_state, side, 5)[1]
