import numpy as np
from ..utils import DIR_ORDER, DIRS, turn_relative

class GreedyAgent:
    # 启发式策略：选择离食物曼哈顿距离最近且不碰撞的下一步；若都危险则直行
    def act(self, env):
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
        if best is None:
            return 1
        return best
