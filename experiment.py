"""Reproducible train/evaluate/replay CLI; Python 3.10+ and standard library."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

from busters import BustersGameRules
from ghostAgents import StaticGhost
from layout import Layout
from learningAgents import QLearningAgent, RandomMazeAgent, GreedyMazeAgent, position

ROOT = Path(__file__).resolve().parent
# Original small mazes. The two disconnected bottom cells are capture jails.
MAZES = {
    "room": ["%%%%%%%%%", "%       %", "%       %", "%       %", "%       %", "%       %",
             "%%%%%%%%%", "% % %%%%%", "%%%%%%%%%"],
    "corridors": ["%%%%%%%%%", "%       %", "% %%% % %", "%   %   %", "% % %%% %", "%       %",
                  "%%%%%%%%%", "% % %%%%%", "%%%%%%%%%"],
    "detour": ["%%%%%%%%%", "%       %", "% %%%%% %", "% %   % %", "% % % % %", "%   %   %",
               "%%%%%%%%%", "% % %%%%%", "%%%%%%%%%"],
}
TRAIN_SEEDS = (11, 29, 47)
TEST_BASE = 1_000_000
MAX_MOVES = 120


def scenario(maze, seed):
    rows = MAZES[maze]
    cells = [(x, len(rows) - 1 - row) for row, text in enumerate(rows)
             for x, char in enumerate(text) if char == " " and len(rows) - 1 - row >= 3]
    spawn = random.Random(seed).sample(cells, 3)
    grid = [list(row) for row in rows]
    for char, (x, y) in zip("PGG", spawn):
        grid[len(rows) - 1 - y][x] = char
    return Layout(["".join(row) for row in grid])


class Recorder:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.frames = []

    def snapshot(self, data):
        if self.enabled:
            self.frames.append({"pacman": list(map(int, data.agentStates[0].getPosition())),
                                "ghosts": [list(map(int, agent.getPosition())) for agent in data.agentStates[1:]],
                                "score": int(data.score)})

    def initialize(self, data):
        self.snapshot(data)

    def update(self, data):
        # Only record Pacman's turn; stationary ghost turns add no information.
        if data._agentMoved == 0:
            self.snapshot(data)

    def finish(self):
        pass


def play(agent, maze, seed, max_moves=MAX_MOVES, record=False):
    display = Recorder(record)
    board = scenario(maze, seed)
    game = BustersGameRules().newGame(board, agent, [StaticGhost(1), StaticGhost(2)], display, max_moves)
    game.run()
    moves = sum(index == 0 for index, _ in game.moveHistory)
    return {"maze": maze, "scenario_seed": seed, "win": game.state.isWin(), "moves": moves,
            "captures": 2 - sum(game.state.getLivingGhosts()), "score": int(game.state.getScore()),
            **({"board": board.layoutText, "frames": display.frames} if record else {})}


def training_seed(run_seed, episode):
    # A disjoint seed range per independent run; no evaluation identifiers.
    return run_seed * 10_000 + episode


def train(episodes, run_seed, output):
    if not 1 <= episodes <= 5000:
        raise ValueError("Use 1–5000 episodes; the fixed test seed range must stay disjoint")
    agent = QLearningAgent(seed=run_seed)
    history = []
    mazes = list(MAZES)
    for episode in range(episodes):
        agent.epsilon = 0.8 + (0.05 - 0.8) * min(1, episode / max(1, episodes * 0.8))
        result = play(agent, mazes[episode % len(mazes)], training_seed(run_seed, episode))
        history.append({"episode": episode + 1, "epsilon": round(agent.epsilon, 4), **result})
    model = output / f"q-seed-{run_seed}.json"
    agent.save(model, {"run_seed": run_seed, "episodes": episodes, "max_moves": MAX_MOVES,
                      "mazes": MAZES, "task": "observable-stationary-ghosts"})
    return agent, history, model


def evaluate(agent, count=30):
    rows = []
    for offset, maze in enumerate(MAZES):
        for episode in range(count):
            rows.append(play(agent, maze, TEST_BASE + offset * 10_000 + episode))
    return rows


def aggregate(rows):
    return {"games": len(rows), "wins": sum(row["win"] for row in rows),
            "win_rate": statistics.mean(row["win"] for row in rows),
            "mean_moves": statistics.mean(row["moves"] for row in rows),
            "mean_captures": statistics.mean(row["captures"] for row in rows),
            "mean_score": statistics.mean(row["score"] for row in rows)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_demo(model, output):
    """Export genuine trajectories on identical initial boards for each policy."""
    scenarios = {}
    for offset, maze in enumerate(MAZES):
        seed = TEST_BASE + offset * 10_000
        agents = {"q_learning": QLearningAgent.load(model, seed=11),
                  "random": RandomMazeAgent(11),
                  "untrained_q": QLearningAgent(seed=11, training=False),
                  "greedy_bfs": GreedyMazeAgent()}
        scenarios[maze] = {name: play(agent, maze, seed, record=True) for name, agent in agents.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("window.REPLAYS = " + json.dumps(scenarios) + ";\n", encoding="utf-8")


def benchmark(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    histories, records, models = [], [], []
    for run_seed in TRAIN_SEEDS:
        trained, history, model = train(args.episodes, run_seed, output / "models")
        histories.extend({"run_seed": run_seed, **row} for row in history)
        models.append({"path": str(model.relative_to(output)),
                       "sha256": hashlib.sha256(model.read_bytes()).hexdigest()})
        agents = {"random": RandomMazeAgent(run_seed),
                  "untrained_q": QLearningAgent(seed=run_seed, training=False, epsilon=0),
                  "q_learning": QLearningAgent.load(model, seed=run_seed),
                  "greedy_bfs": GreedyMazeAgent()}
        for name, agent in agents.items():
            before = repr(getattr(agent, "q", {}))
            rows = evaluate(agent, args.test_games)
            if repr(getattr(agent, "q", {})) != before:
                raise RuntimeError("Evaluation modified Q values")
            records.extend({"run_seed": run_seed, "agent": name, **row} for row in rows)
        print(f"Run {run_seed}: {args.episodes} training episodes; evaluation complete", flush=True)
    summary = {name: aggregate([row for row in records if row["agent"] == name])
               for name in agents}
    per_run = {str(seed): {name: aggregate([row for row in records
                     if row["agent"] == name and row["run_seed"] == seed]) for name in agents}
               for seed in TRAIN_SEEDS}
    payload = {"protocol": {"episodes_per_run": args.episodes, "run_seeds": list(TRAIN_SEEDS),
                           "test_games_per_maze": args.test_games, "max_moves": MAX_MOVES,
                           "test_seed_base": TEST_BASE, "mazes": MAZES},
               "models": models, "summary": summary, "per_run": per_run,
               "per_maze": {maze: {name: aggregate([row for row in records
                            if row["agent"] == name and row["maze"] == maze])
                            for name in agents} for maze in MAZES}}
    write_json(output / "benchmark.json", payload)
    for filename, rows in (("evaluation.csv", records), ("training.csv", histories)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("benchmark", help="Train three independent models and compare four agents")
    bench.add_argument("--episodes", type=int, default=3000)
    bench.add_argument("--test-games", type=int, default=30)
    bench.add_argument("--output", type=Path, default=ROOT / "results" / "local")
    training = commands.add_parser("train")
    training.add_argument("--episodes", type=int, default=3000)
    training.add_argument("--seed", type=int, choices=TRAIN_SEEDS, default=11)
    training.add_argument("--output", type=Path, default=ROOT / "results" / "local" / "models")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--games", type=int, default=30)
    replay = commands.add_parser("replay")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--maze", choices=MAZES, default="detour")
    replay.add_argument("--seed", type=int, default=TEST_BASE + 20_000)
    replay.add_argument("--output", type=Path, default=ROOT / "demo" / "replay.js")
    demo = commands.add_parser("demo", help="Export the four policies on three shared scenarios")
    demo.add_argument("--model", type=Path, required=True)
    demo.add_argument("--output", type=Path, default=ROOT / "demo" / "replay.js")
    args = parser.parse_args()
    if args.command == "benchmark":
        if not 1 <= args.test_games <= 1000:
            parser.error("--test-games must be in 1..1000")
        benchmark(args)
    elif args.command == "train":
        print(train(args.episodes, args.seed, args.output)[2])
    elif args.command == "evaluate":
        if not 1 <= args.games <= 1000:
            parser.error("--games must be in 1..1000")
        print(json.dumps(aggregate(evaluate(QLearningAgent.load(args.model), args.games)), indent=2))
    elif args.command == "demo":
        export_demo(args.model, args.output)
        print(args.output)
    else:
        result = play(QLearningAgent.load(args.model), args.maze, args.seed, record=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("window.REPLAY = " + json.dumps(result) + ";\n", encoding="utf-8")
        print(f"{args.output}: {result['moves']} moves, win={result['win']}")


if __name__ == "__main__":
    main()
