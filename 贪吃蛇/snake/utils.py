import numpy as np

# 网格上四个基本方向向量（x 正向右，y 正向下）
DIRS = {
    "UP": np.array([0, -1]),
    "DOWN": np.array([0, 1]),
    "LEFT": np.array([-1, 0]),
    "RIGHT": np.array([1, 0]),
}

# 固定方向顺序：上→右→下→左（用于相对转向/编码）
DIR_ORDER = ["UP", "RIGHT", "DOWN", "LEFT"]

def turn_relative(curr_dir, action):
    # 将相对动作映射到绝对方向
    # action: 0=左转, 1=直行, 2=右转
    i = DIR_ORDER.index(curr_dir)
    if action == 0:
        i = (i - 1) % 4
    elif action == 1:
        i = i
    elif action == 2:
        i = (i + 1) % 4
    return DIR_ORDER[i]

def opposite(dir_name):
    # 返回相反方向（用于禁止立即反向）
    if dir_name == "UP":
        return "DOWN"
    if dir_name == "DOWN":
        return "UP"
    if dir_name == "LEFT":
        return "RIGHT"
    return "LEFT"

def relative_action(curr_dir, desired_dir):
    # 计算“当前方向→目标绝对方向”需要的相对动作（左转/直行/右转）
    ci = DIR_ORDER.index(curr_dir)
    di = DIR_ORDER.index(desired_dir)
    delta = (di - ci) % 4
    if delta == 0:
        return 1
    if delta == 1:
        return 2
    if delta == 3:
        return 0
    return 1
def sign(x):
    # 将数值压缩到 {-1,0,1}，用于离散化食物相对位置
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def encode_state(head, food, direction, board, danger):
    # 将环境状态离散化为有限维度的元组
    # danger = (左前方危险, 正前方危险, 右前方危险) → {0,1}
    # fx, fy = 食物相对方向的符号 → {-1,0,1}
    # d = 头部朝向在 DIR_ORDER 中的索引 → {0..3}
    dx = food[0] - head[0]
    dy = food[1] - head[1]
    fx = sign(dx)
    fy = sign(dy)
    d = DIR_ORDER.index(direction)
    left, front, right = danger
    return (int(left), int(front), int(right), fx, fy, d)

def state_space_size():
    # 2*2*2（危险标志） * 3*3（食物相对符号） * 4（朝向）
    return 2 * 2 * 2 * 3 * 3 * 4

def action_space_size():
    # 相对动作空间：左转/直行/右转
    return 3
