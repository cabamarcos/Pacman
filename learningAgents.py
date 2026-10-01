"""Original learning extension: observable, stationary ghost navigation.

The Berkeley inference exercises are deliberately left unchanged. This module
learns goal-conditioned navigation on the existing game engine instead.
"""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import random

from game import Agent, Directions, Actions


def legal_moves(state):
    actions = state.getLegalActions(0)
    moving = [action for action in actions if action != Directions.STOP]
    return moving or actions


def position(state):
    return tuple(map(int, state.getPacmanPosition()))


def maze_id(state):
    walls = state.getWalls()
    return hashlib.sha256(str(walls).encode()).hexdigest()[:16]


def living_targets(state):
    return [(i, tuple(map(int, state.getGhostPosition(i))))
            for i, alive in enumerate(state.getLivingGhosts()) if alive]


def manhattan(first, second):
    return sum(abs(a - b) for a, b in zip(first, second))


class RandomMazeAgent(Agent):
    def __init__(self, seed=0):
        super().__init__(0)
        self.rng = random.Random(seed)

    def getAction(self, state):
        return self.rng.choice(legal_moves(state))


class GreedyMazeAgent(Agent):
    """BFS to the closest reachable ghost; an informed, non-learning baseline."""
    def getAction(self, state):
        start = position(state)
        goals = {target for _, target in living_targets(state)}
        queue = deque([(start, None)])
        visited = {start}
        walls = state.getWalls()
        while queue:
            current, first = queue.popleft()
            if current in goals and first is not None:
                return first
            for action in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
                dx, dy = Actions.directionToVector(action)
                nxt = (current[0] + int(dx), current[1] + int(dy))
                if (0 <= nxt[0] < walls.width and 0 <= nxt[1] < walls.height
                        and not walls[nxt[0]][nxt[1]] and nxt not in visited):
                    visited.add(nxt)
                    queue.append((nxt, first or action))
        return legal_moves(state)[0]


class QLearningAgent(Agent):
    """Tabular Q-learning for successive, fixed target navigation subtasks.

    State: maze fingerprint, Pacman coordinates and selected ghost coordinates.
    A target is retained until captured. Its capture terminates that subtask,
    so neither the bootstrap nor the potential jumps to the next target.
    """
    FORMAT = "pacman-goal-q-v1"

    def __init__(self, seed=0, training=True, alpha=0.3, gamma=0.95, epsilon=0.1):
        super().__init__(0)
        if not (0 < alpha <= 1 and 0 <= gamma < 1 and 0 <= epsilon <= 1):
            raise ValueError("Invalid learning parameters")
        self.rng = random.Random(seed)
        self.training = training
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = {}
        self.pending = None
        self.target = None

    def registerInitialState(self, state):
        self.pending = None
        self.target = None
        self.maze = maze_id(state)

    def key(self, pacman, goal):
        return (self.maze, *pacman, *goal)

    def values(self, key):
        # Evaluation never creates or modifies Q entries.
        return self.q.get(key, {})

    def update(self, key, action, reward, next_key, next_actions, terminal):
        if not self.training:
            return
        row = self.q.setdefault(key, {})
        continuation = 0 if terminal else max(
            (self.values(next_key).get(a, 0.0) for a in next_actions), default=0.0)
        old = row.get(action, 0.0)
        row[action] = old + self.alpha * (reward + self.gamma * continuation - old)

    def observe(self, state):
        if self.pending is None:
            return
        key, action, index, goal, prev_position = self.pending
        captured = not state.getLivingGhosts()[index]
        terminal = captured or state.isWin() or state.isLose()
        # Intrinsic subgoal reward, distinct from the engine's game score.
        reward = -1.0 + (20.0 if captured else 0.0)
        phi_before = -manhattan(prev_position, goal)
        phi_after = 0.0 if terminal else -manhattan(position(state), goal)
        reward += self.gamma * phi_after - phi_before
        self.update(key, action, reward, self.key(position(state), goal),
                    [] if terminal else legal_moves(state), terminal)
        self.pending = None
        if captured:
            self.target = None

    def getAction(self, state):
        self.observe(state)
        if self.target is None:
            self.target = min(living_targets(state),
                              key=lambda item: (manhattan(position(state), item[1]), item[0]))
        index, goal = self.target
        key = self.key(position(state), goal)
        actions = legal_moves(state)
        if self.training and self.rng.random() < self.epsilon:
            action = self.rng.choice(actions)
        else:
            row = self.values(key)
            best = max(row.get(a, 0.0) for a in actions)
            action = self.rng.choice([a for a in actions if row.get(a, 0.0) == best])
        self.pending = (key, action, index, goal, position(state))
        return action

    def final(self, state):
        self.observe(state)

    def save(self, path, metadata=None):
        payload = {"format": self.FORMAT, "alpha": self.alpha, "gamma": self.gamma,
                   "metadata": metadata or {},
                   "rows": [[list(key), row] for key, row in sorted(self.q.items())]}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, seed=0):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.FORMAT:
            raise ValueError("Unsupported model format")
        agent = cls(seed=seed, training=False, alpha=payload["alpha"], gamma=payload["gamma"], epsilon=0)
        for raw_key, row in payload["rows"]:
            if (len(raw_key) != 5 or not isinstance(raw_key[0], str)
                    or not all(type(value) is int for value in raw_key[1:])
                    or not isinstance(row, dict)
                    or not all(action in ("North", "South", "East", "West", "Stop")
                               and type(value) in (int, float) and math.isfinite(value)
                               for action, value in row.items())):
                raise ValueError("Malformed Q-table")
            key = tuple(raw_key)
            if key in agent.q:
                raise ValueError("Duplicate Q-table row")
            agent.q[key] = row
        agent.metadata = payload.get("metadata", {})
        return agent
