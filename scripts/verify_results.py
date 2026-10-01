"""Check the reference CSVs, model hashes and repeat the trained evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import aggregate, evaluate, TRAIN_SEEDS, MAZES
from learningAgents import QLearningAgent

directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "reference"
payload = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
with (directory / "evaluation.csv").open(encoding="utf-8") as stream:
    rows = [{**row, "run_seed": int(row["run_seed"]), "scenario_seed": int(row["scenario_seed"]),
             "win": row["win"] == "True", "moves": int(row["moves"]),
             "captures": int(row["captures"]), "score": int(row["score"])}
            for row in csv.DictReader(stream)]
count = payload["protocol"]["test_games_per_maze"]
assert len(rows) == len(TRAIN_SEEDS) * len(MAZES) * count * 4
assert len({(r["run_seed"], r["agent"], r["maze"], r["scenario_seed"]) for r in rows}) == len(rows)
for name, expected in payload["summary"].items():
    measured = aggregate([row for row in rows if row["agent"] == name])
    assert all(math.isclose(measured[key], value, abs_tol=1e-12) for key, value in expected.items())
for row in rows:
    assert row["score"] == 200 * row["captures"] - row["moves"]
    assert 0 <= row["captures"] <= 2
    assert row["win"] == (row["captures"] == 2)
    assert 1 <= row["moves"] <= payload["protocol"]["max_moves"]
for seed, metadata in zip(TRAIN_SEEDS, payload["models"]):
    model = directory / metadata["path"]
    assert hashlib.sha256(model.read_bytes()).hexdigest() == metadata["sha256"]
    agent = QLearningAgent.load(model, seed=seed)
    before = repr(agent.q)
    measured = evaluate(agent, count)
    expected = [{key: row[key] for key in measured[0]} for row in rows
                if row["agent"] == "q_learning" and row["run_seed"] == seed]
    assert measured == expected, f"Evaluation differs for model seed {seed}"
    assert repr(agent.q) == before
for name in ("comparison.svg", "training.svg"):
    ET.parse(directory / name)
print(f"Verified {len(rows)} evaluation records, three model hashes, frozen replay and SVG charts.")
