import numpy as np
import os
from ..utils import state_space_size, action_space_size

class QLearningAgent:
    """
    表格型 Q-Learning：
    - 状态 s 离散化为单一索引（见 _state_index）
    - 策略：ε-贪心（训练时探索，评估时 greedy）
    - 更新：Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros((state_space_size(), action_space_size()), dtype=np.float32)

    def _state_index(self, s):
        # 将 (危险标志, 食物相对方向符号, 朝向) 组合成一维索引
        # 编码顺序与 utils.state_space_size 一致
        l, f, r, fx, fy, d = s
        idx = 0
        idx = idx * 2 + l
        idx = idx * 2 + f
        idx = idx * 2 + r
        idx = idx * 3 + (fx + 1)
        idx = idx * 3 + (fy + 1)
        idx = idx * 4 + d
        return idx

    def act(self, state, training=True):
        # 训练时按 ε-贪心探索；评估时纯 greedy
        si = self._state_index(state)
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(action_space_size())
        return int(np.argmax(self.q[si]))

    def update(self, s, a, r, s2, done):
        # 经典 Q-Learning 更新；终止状态不引入未来回报
        si = self._state_index(s)
        si2 = self._state_index(s2)
        best_next = 0.0 if done else np.max(self.q[si2])
        td = r + self.gamma * best_next - self.q[si, a]
        upd = self.alpha * td
        self.q[si, a] += upd
        return abs(upd)
    def train(self, env, episodes=500, epsilon_decay=0.995, min_epsilon=0.05, max_steps=2000, window_size=50, conv_threshold=0.05, conv_patience=3, print_every=0, alpha_decay=1.0, min_alpha=0.0, conv_qdelta_threshold=0.0):
        rewards = []
        scores = []
        epsilons = []
        q_update_mean = []
        q_delta_mean = []
        conv_count = 0
        last_avg = None
        q_conv_count = 0
        last_qavg = None
        converged_at = None
        converged_reward_at = None
        converged_qdelta_at = None
        for ep in range(episodes):
            q_before = self.q.copy()
            s = env.reset()
            total = 0.0
            ep_score = 0
            upd_acc = []
            for _ in range(max_steps):
                a = self.act(s, training=True)
                s2, r, done, info = env.step(a)
                du = self.update(s, a, r, s2, done)
                upd_acc.append(du)
                s = s2
                total += r
                if done:
                    ep_score = info.get("score", 0)
                    break
            rewards.append(total)
            scores.append(ep_score)
            epsilons.append(self.epsilon)
            q_update_mean.append(float(np.mean(upd_acc)) if upd_acc else 0.0)
            q_delta_mean.append(float(np.mean(np.abs(self.q - q_before))))
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)
            self.alpha = max(min_alpha, self.alpha * alpha_decay)
            r_ma = None
            s_ma = None
            qd_ma = None
            if len(rewards) >= window_size:
                r_ma = float(np.mean(rewards[-window_size:]))
                s_ma = float(np.mean(scores[-window_size:]))
                if last_avg is None:
                    last_avg = r_ma
                    conv_count = 0
                else:
                    if abs(r_ma - last_avg) <= conv_threshold:
                        conv_count += 1
                    else:
                        conv_count = 0
                        last_avg = r_ma
                if len(q_delta_mean) >= window_size and conv_qdelta_threshold > 0.0:
                    qd_ma = float(np.mean(q_delta_mean[-window_size:]))
                    if last_qavg is None:
                        last_qavg = qd_ma
                        q_conv_count = 0
                    else:
                        if qd_ma <= conv_qdelta_threshold:
                            q_conv_count += 1
                        else:
                            q_conv_count = 0
                            last_qavg = qd_ma
                if print_every and ((ep + 1) % print_every == 0):
                    if qd_ma is None:
                        print("ep", ep + 1, "reward", f"{total:.3f}", "score", ep_score, "eps", f"{epsilons[-1]:.3f}", "r_ma", f"{r_ma:.3f}", "s_ma", f"{s_ma:.3f}", "conv", conv_count)
                    else:
                        print("ep", ep + 1, "reward", f"{total:.3f}", "score", ep_score, "eps", f"{epsilons[-1]:.3f}", "r_ma", f"{r_ma:.3f}", "s_ma", f"{s_ma:.3f}", "qd_ma", f"{qd_ma:.4f}", "conv", conv_count, "qconv", q_conv_count)
                if conv_patience > 0 and len(rewards) >= 2 * window_size and converged_at is None:
                    if conv_qdelta_threshold > 0.0:
                        if conv_count >= conv_patience and q_conv_count >= conv_patience:
                            converged_at = ep + 1
                            converged_reward_at = ep + 1
                            converged_qdelta_at = ep + 1
                            break
                    else:
                        if conv_count >= conv_patience:
                            converged_at = ep + 1
                            converged_reward_at = ep + 1
                            break
            else:
                if print_every and ((ep + 1) % print_every == 0):
                    print("ep", ep + 1, "reward", f"{total:.3f}", "score", ep_score, "eps", f"{epsilons[-1]:.3f}")
        return {
            "rewards": rewards,
            "scores": scores,
            "epsilons": epsilons,
            "q_update_mean": q_update_mean,
            "q_delta_mean": q_delta_mean,
            "converged_at": converged_at,
            "converged_reward_at": converged_reward_at,
            "converged_qdelta_at": converged_qdelta_at,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)
        return self

    def evaluate(self, env, episodes=20, max_steps=2000):
        total_r = 0.0
        total_s = 0.0
        for _ in range(episodes):
            s = env.reset()
            ep_r = 0.0
            done = False
            steps = 0
            # 评估时不探索，纯 greedy 前进
            while steps < max_steps and not done:
                a = self.act(s, training=False)
                s, r, done, info = env.step(a)
                ep_r += r
                steps += 1
            total_r += ep_r
            total_s += info.get("score", 0)
        return {"avg_reward": total_r / episodes, "avg_score": total_s / episodes}
