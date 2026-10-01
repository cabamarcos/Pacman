"""Behavioral tests for the extension and corrected legacy state copies."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiment import MAZES, TRAIN_SEEDS, TEST_BASE, scenario, play, training_seed, train
from learningAgents import QLearningAgent, RandomMazeAgent, GreedyMazeAgent


class LearningTests(unittest.TestCase):
    def test_bellman_update_and_terminal(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        key, nxt = ("maze", 1, 1, 3, 3), ("maze", 2, 1, 3, 3)
        agent.q[nxt] = {"East": 10.0, "West": 100.0}
        # An illegal action's Q value must not enter the continuation.
        agent.update(key, "East", 2, nxt, ["East"], False)
        self.assertEqual(agent.q[key]["East"], 5.5)
        agent.update(key, "East", 2, nxt, ["East"], True)
        self.assertEqual(agent.q[key]["East"], 3.75)

    def test_evaluation_does_not_learn(self):
        agent = QLearningAgent(seed=3, training=False)
        agent.q[("irrelevant", 0, 0, 0, 0)] = {"East": 12}
        before = copy.deepcopy(agent.q)
        play(agent, "corridors", TEST_BASE, max_moves=10)
        self.assertEqual(agent.q, before)

    def test_model_round_trip_and_rejects_bad_values(self):
        agent = QLearningAgent()
        agent.q[("maze", 1, 2, 3, 4)] = {"East": 4.5}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.json"
            agent.save(model, {"example": True})
            loaded = QLearningAgent.load(model)
            self.assertEqual(agent.q, loaded.q)
            self.assertFalse(loaded.training)
            payload = json.loads(model.read_text())
            payload["rows"][0][1]["Teleport"] = 1
            model.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                QLearningAgent.load(model)

    def test_scenarios_connected_and_deterministic(self):
        for name, rows in MAZES.items():
            self.assertEqual(len({len(row) for row in rows}), 1)
            for seed in range(20):
                first, second = scenario(name, seed), scenario(name, seed)
                self.assertEqual(first.layoutText, second.layoutText)
                floor = {(x, y) for x in range(first.width) for y in range(3, first.height)
                         if not first.walls[x][y]}
                seen = {next(iter(floor))}
                pending = list(seen)
                while pending:
                    x, y = pending.pop()
                    for nxt in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                        if nxt in floor and nxt not in seen:
                            seen.add(nxt)
                            pending.append(nxt)
                self.assertEqual(seen, floor)
                self.assertEqual(len(set(pos for _, pos in first.agentPositions)), 3)

    def test_training_seed_ids_are_disjoint(self):
        ranges = [set(training_seed(seed, n) for n in range(5000)) for seed in TRAIN_SEEDS]
        for i, first in enumerate(ranges):
            self.assertTrue(all(number < TEST_BASE for number in first))
            for second in ranges[i+1:]:
                self.assertFalse(first & second)

    def test_legal_actions_and_timeout(self):
        result = play(RandomMazeAgent(5), "detour", 17, max_moves=1)
        self.assertLessEqual(result["moves"], 1)
        self.assertFalse(result["win"])
        self.assertEqual(result["score"], 200 * result["captures"] - result["moves"])

    def test_greedy_goes_around_walls(self):
        result = play(GreedyMazeAgent(), "detour", 13, record=True)
        self.assertTrue(result["win"])
        self.assertEqual(len(result["frames"]), result["moves"] + 1)
        rows = result["board"]
        for frame in result["frames"]:
            x, y = frame["pacman"]
            self.assertNotEqual(rows[len(rows)-1-y][x], "%")

    def test_training_and_loaded_policy_are_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first, history, model = train(12, 11, output)
            second, history2, _ = train(12, 11, output / "second")
            self.assertEqual(first.q, second.q)
            self.assertEqual(history, history2)
            self.assertEqual(play(QLearningAgent.load(model, seed=5), "room", TEST_BASE),
                             play(QLearningAgent.load(model, seed=5), "room", TEST_BASE))


if __name__ == "__main__":
    unittest.main()
