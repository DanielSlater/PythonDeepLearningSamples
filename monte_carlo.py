import collections
import random
import math
from tic_tac_toe import has_winner, available_moves, apply_move


def monte_carlo_sample(board_state, side):
    result = has_winner(board_state)
    if result != 0:
        return result, None
    moves = list(available_moves(board_state))
    if not moves:
        return 0, None

    # select a random move
    move = random.choice(moves)
    result, next_move = monte_carlo_sample(apply_move(board_state, move, side), -side)
    return result, move


def monte_carlo_tree_search(board_state, side, number_of_samples):
    move_wins = collections.defaultdict(int)
    move_samples = collections.defaultdict(int)
    for _ in range(number_of_samples):
        result, move = monte_carlo_sample(board_state, side)
        # store the result and a count of the number of times we have tried this move
        if result == side:
            move_wins[move] += 1
        move_samples[move] += 1

    # get the move with the best average result
    move = max(move_wins, key=lambda x: move_wins.get(x) / move_samples[move])

    return move_wins[move] / move_samples[move], move


def upper_confidence_bounds(wins, samples, total_samples):
    return wins / samples + math.sqrt(2 * math.log(total_samples) / samples)


def monte_carlo_tree_search_uct(board_state, side, number_of_samples):
    state_results = collections.defaultdict(float)
    state_samples = collections.defaultdict(float)

    for total_samples in range(number_of_samples):
        current_side = side
        current_board_state = board_state
        first_unvisited_child = True
        rollout_path = []
        winner = 0

        while True:
            move_states = {move: apply_move(current_board_state, move, current_side)
                           for move in available_moves(current_board_state)}

            if not move_states:
                winner = 0
                break

            if all((state in state_samples) for _, state in move_states):
                move, state = max(move_states, key=lambda _, s: upper_confidence_bounds(state_results[s],
                                                                                        state_samples[s],
                                                                                        total_samples))
            else:
                move = random.choice(move_states.keys())

            current_side = -current_side
            current_board_state = move_states[move]

            if first_unvisited_child:
                rollout_path.append(current_board_state)
                if current_board_state not in state_samples:
                    first_unvisited_child = False

            winner = has_winner(current_board_state)
            if winner != 0:
                break

        for state in rollout_path:
            state_samples[state] += 1.
            state_results[state] += winner * side

    move_states = {move: apply_move(board_state, move, side) for move in available_moves(board_state)}

    move = max(move_states, key=lambda x: state_results[move_states[x]] / state_samples[move_states[x]])

    return state_results[move_states[move]] / state_samples[move_states[move]], move
