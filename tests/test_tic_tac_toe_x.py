from unittest import TestCase

from tic_tac_toe_x import has_winner, has_winning_line, play_game, random_player


class TestTicTacToeX(TestCase):

    def test_has_winning_line(self):
        self.assertEqual(1, has_winning_line((0, 1, 1, 1, 1), 4))
        self.assertEqual(0, has_winning_line((0, 1, -1, 1, 1), 4))
        self.assertEqual(1, has_winning_line((1, 1, 1, 1, 1, 0), 4))
        self.assertEqual(0, has_winning_line((1, 1, 1, 1, 1, 0), 5))
        self.assertEqual(-1, has_winning_line((-1, -1, -1, -1, 1), 4))

    def test_has_winner(self):
        board_state = ((0, 0, 0, 0, 0),
                       (0, -1, 0, 0, 0),
                       (0, -1, 0, 0, 0),
                       (0, -1, 0, 0, 0),
                       (0, -1, 0, 0, 0))
        self.assertEqual(-1, has_winner(board_state))

        board_state = ((0, 1, 0, 0, 0),
                       (0, 0, 1, 0, 0),
                       (0, 0, 0, 1, 0),
                       (0, 0, 0, 0, 1),
                       (0, 0, 0, 0, 0))
        self.assertEqual(1, has_winner(board_state))

        board_state = ((0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 1),
                       (0, 0, 0, 1, 0),
                       (0, 0, 1, 0, 0),
                       (0, 1, 0, 0, 0))
        self.assertEqual(1, has_winner(board_state))

        board_state = ((0, 0, 0, -1, 0),
                       (0, 0, -1, 0, 0),
                       (0, -1, 0, 0, 0),
                       (-1, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0))
        self.assertEqual(-1, has_winner(board_state))

    def test_play_game(self):
        play_game(random_player, random_player)