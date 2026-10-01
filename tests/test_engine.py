"""Regression tests for Python 3 support and isolated game states."""
import unittest

from busters import GameState
from compatibility import old_div
from game import Directions, Grid
from layout import Layout


def initial_state():
    board = Layout(["%%%%%%%%%", "%      G%", "%       %", "%       %", "%PG     %",
                    "%%%%%%%%%", "% % %%%%%", "%%%%%%%%%"])
    state = GameState()
    state.initialize(board, 2)
    return state


class EngineTests(unittest.TestCase):
    def test_successor_does_not_mutate_predecessor(self):
        state = initial_state()
        state.ghostDirections[0] = "sentinel"
        old_positions = state.ghostPositions[:]
        nxt = state.generateSuccessor(0, Directions.NORTH)
        self.assertEqual(state.ghostPositions, old_positions)
        self.assertEqual(state.ghostDirections[0], "sentinel")
        self.assertNotEqual(nxt.ghostDirections[0], "sentinel")
        self.assertEqual(nxt.getGhostPositions(), [nxt.getGhostPosition(i) for i in (1, 2)])
        duplicate = state.deepCopy()
        duplicate.data.ghostDistances[0] = -9
        self.assertNotEqual(state.data.ghostDistances[0], -9)

    def test_capture_updates_cached_position(self):
        state = initial_state()
        nxt = state.generateSuccessor(0, Directions.EAST)
        self.assertFalse(nxt.getLivingGhosts()[1])
        self.assertEqual(nxt.getGhostPositions()[0], (1, 1))
        self.assertTrue(state.getLivingGhosts()[1])
        self.assertEqual(state.getGhostPositions()[0], (2, 3))

    def test_win_precedes_timeout(self):
        state = initial_state()
        state.livingGhosts = [False, False, False]
        state.maxMoves = state.numMoves = 1
        self.assertTrue(state.isWin())
        self.assertFalse(state.isLose())

    def test_integer_division_preserves_grid_serialization(self):
        self.assertEqual(old_div(5, 2), 2)
        self.assertEqual(old_div(5, 2.0), 2.5)
        grid = Grid(5, 8)
        grid[3][6] = True
        unpacked = Grid(5, 8, bitRepresentation=grid.packBits()[2:])
        self.assertEqual(grid, unpacked)


if __name__ == "__main__":
    unittest.main()
