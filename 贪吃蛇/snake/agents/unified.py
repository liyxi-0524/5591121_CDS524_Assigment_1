import numpy as np
from .qlearning import QLearningAgent
from ..utils import DIRS, turn_relative

class UnifiedAgent(QLearningAgent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, mode="q"):
        super().__init__(alpha=alpha, gamma=gamma, epsilon=epsilon)
        self.mode = mode  # "q" | "greedy" | "mixed"
        self._env = None

    def set_env(self, env):
        self._env = env

    def _greedy_action(self, env):
        head = env.snake[0]
        fx, fy = env.food
        best = None
        best_dist = None
        for a in env.valid_actions():
            ndir = turn_relative(env.direction, a)
            move = DIRS[ndir]
            nh = (head[0] + int(move[0]), head[1] + int(move[1]))
            if env.collision(nh):
                continue
            d = abs(nh[0] - fx) + abs(nh[1] - fy)
            if best is None or d < best_dist:
                best = a
                best_dist = d
        return 1 if best is None else best

    def act(self, state, training=True):
        if self.mode == "q":
            return super().act(state, training=training)
        if self.mode == "greedy":
            if self._env is None:
                return super().act(state, training=False)
            return self._greedy_action(self._env)
        # mixed：优先用 Q；若该状态 Q 值全相等或接近，则用贪心
        si = self._state_index(state)
        row = self.q[si]
        if np.allclose(row, row[0]) and self._env is not None:
            return self._greedy_action(self._env)
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(row.shape[0])
        return int(np.argmax(row))
