import numpy as np
import random
from .utils import DIRS, DIR_ORDER, turn_relative, opposite, encode_state

class SnakeEnv:
    """
    简化版贪吃蛇环境：
    - 动作：相对动作 {0:左转, 1:直行, 2:右转}
    - 奖励：吃到食物 +1；撞墙/撞身体 -10；每步微惩罚 -0.01；长期未吃食 -5（防止原地打转）
    - 观测：由 utils.encode_state 离散化（危险标志、食物相对方向、朝向）
    """
    def __init__(self, width=20, height=20, seed=None, reward_food=18.0, reward_step=-0.005, reward_death=-10.0, reward_timeout=-5.0, reward_closer=0.2, reward_farther=-0.2, reward_body_farther=0.1, reward_body_closer=-0.1):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.rew_food = reward_food
        self.rew_step = reward_step
        self.rew_death = reward_death
        self.rew_timeout = reward_timeout
        self.rew_closer = reward_closer
        self.rew_farther = reward_farther
        self.rew_body_farther = reward_body_farther
        self.rew_body_closer = reward_body_closer
        self.prev_dist = None
        self.prev_body_min = None
        self.reset()

    def reset(self):
        cx = self.width // 2
        cy = self.height // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = "RIGHT"
        self.done = False
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.place_food()
        head = self.snake[0]
        self.prev_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        self.prev_body_min = self._min_body_distance()
        return self.get_state()

    def place_food(self):
        positions = set(self.snake)
        free = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in positions]
        self.food = self.rng.choice(free)

    def collision(self, pos):
        x, y = pos
        # 撞墙
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # 撞到自身（忽略头部，与移动后的尾部是否移除相匹配）
        if pos in self.snake[1:]:
            return True
        return False

    def step(self, action):
        if self.done:
            # 结束后继续 step 返回 done 状态
            return self.get_state(), 0.0, True, {"score": self.score}
        # 将相对动作转为新的绝对朝向
        self.direction = turn_relative(self.direction, action)
        head = self.snake[0]
        move = DIRS[self.direction]
        new_head = (head[0] + int(move[0]), head[1] + int(move[1]))
        reward = self.rew_step
        self.steps += 1
        self.steps_since_food += 1
        if self.collision(new_head):
            self.done = True
            reward = self.rew_death
            return self.get_state(), reward, True, {"score": self.score}
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            reward += self.rew_food
            self.place_food()
            self.steps_since_food = 0
            head2 = self.snake[0]
            self.prev_dist = abs(head2[0] - self.food[0]) + abs(head2[1] - self.food[1])
        else:
            self.snake.pop()
            d = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if self.prev_dist is not None:
                if d < self.prev_dist:
                    reward += self.rew_closer
                elif d > self.prev_dist:
                    reward += self.rew_farther
            self.prev_dist = d
            bmin = self._min_body_distance()
            if self.prev_body_min is not None:
                if bmin > self.prev_body_min:
                    reward += self.rew_body_farther
                elif bmin < self.prev_body_min:
                    reward += self.rew_body_closer
            self.prev_body_min = bmin
        # 长时间未吃到食物，提前终止一局
        if self.steps_since_food > self.width * self.height * 2:
            self.done = True
            reward += self.rew_timeout
        return self.get_state(), reward, self.done, {"score": self.score}

    def _min_body_distance(self):
        if len(self.snake) <= 1:
            return 9999
        hx, hy = self.snake[0]
        best = 9999
        for (x, y) in self.snake[1:]:
            d = abs(hx - x) + abs(hy - y)
            if d < best:
                best = d
        return best

    def get_state(self):
        head = self.snake[0]
        danger = self._danger_flags()
        return encode_state(head, self.food, self.direction, (self.width, self.height), danger)

    def _danger_flags(self):
        # 计算左/前/右三个位置是否危险（下一步会碰撞）
        dir_idx = DIR_ORDER.index(self.direction)
        left_dir = DIR_ORDER[(dir_idx - 1) % 4]
        right_dir = DIR_ORDER[(dir_idx + 1) % 4]
        front_dir = DIR_ORDER[dir_idx]
        head = self.snake[0]
        left_pos = (head[0] + int(DIRS[left_dir][0]), head[1] + int(DIRS[left_dir][1]))
        right_pos = (head[0] + int(DIRS[right_dir][0]), head[1] + int(DIRS[right_dir][1]))
        front_pos = (head[0] + int(DIRS[front_dir][0]), head[1] + int(DIRS[front_dir][1]))
        return (self.collision(left_pos), self.collision(front_pos), self.collision(right_pos))

    def valid_actions(self):
        return [0, 1, 2]

    def snake_cells(self):
        return list(self.snake)

    def food_cell(self):
        return self.food
