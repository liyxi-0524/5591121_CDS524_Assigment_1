import argparse
import os
from snake.game import SnakeEnv
from snake.agents.unified import UnifiedAgent
from snake.ui.pygame_renderer import GameRenderer, show_menu
import csv
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

def train(args):
    # 训练入口：支持日志、收敛判定、评估与摘要输出
    env = SnakeEnv(width=args.width, height=args.height, seed=args.seed)
    agent = UnifiedAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, mode=getattr(args, "mode", "q"))
    res = agent.train(
        env,
        episodes=args.episodes,
        epsilon_decay=args.eps_decay,
        min_epsilon=args.min_eps,
        max_steps=2000,
        window_size=args.window,
        conv_threshold=args.conv_threshold,
        conv_patience=args.conv_patience,
        print_every=args.print_every,
        alpha_decay=args.alpha_decay,
        min_alpha=args.min_alpha,
    )
    if args.model:
        agent.save(args.model)
    if args.log:
        # 将每个 episode 的指标写入 CSV（含移动平均），便于后续可视化
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        with open(args.log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward", "score", "epsilon", "q_update_mean", "q_delta_mean", "reward_ma", "score_ma"])
            from collections import deque
            win_r = deque(maxlen=args.window)
            win_s = deque(maxlen=args.window)
            q_um = res.get("q_update_mean", [])
            q_dm = res.get("q_delta_mean", [])
            for i in range(len(res["rewards"])):
                r = res["rewards"][i]
                s = res["scores"][i]
                e = res["epsilons"][i] if i < len(res["epsilons"]) else ""
                qu = q_um[i] if i < len(q_um) else ""
                qd = q_dm[i] if i < len(q_dm) else ""
                win_r.append(r)
                win_s.append(s)
                rma = sum(win_r) / len(win_r)
                sma = sum(win_s) / len(win_s)
                w.writerow([i + 1, r, s, e, qu, qd, rma, sma])
    k = min(50, len(res["rewards"]))
    last_r = sum(res["rewards"][-k:]) / k
    last_s = sum(res["scores"][-k:]) / k
    print("episodes", len(res["rewards"]))
    print("last_avg_reward", f"{last_r:.3f}")
    print("last_avg_score", f"{last_s:.3f}")
    if res.get("converged_at"):
        print("converged_at", res["converged_at"])
    if args.eval_episodes > 0:
        eval_env = SnakeEnv(width=args.width, height=args.height, seed=args.seed)
        ev = agent.evaluate(eval_env, episodes=args.eval_episodes)
        print("eval_avg_reward", f"{ev['avg_reward']:.3f}")
        print("eval_avg_score", f"{ev['avg_score']:.3f}")

def play_q(args):
    agent = UnifiedAgent(mode="q")
    if args.model and os.path.exists(args.model):
        agent.load(args.model)
    gr = GameRenderer(width=args.width, height=args.height, cell=args.cell)
    gr.run_agent(agent, fps=args.fps, episodes=1)

def play_greedy(args):
    agent = UnifiedAgent(mode="greedy")
    gr = GameRenderer(width=args.width, height=args.height, cell=args.cell)
    gr.run_agent(agent, fps=args.fps, episodes=1)

def human(args):
    gr = GameRenderer(width=args.width, height=args.height, cell=args.cell)
    gr.run_human(fps=args.fps)

def play_unified(args):
    agent = UnifiedAgent(mode="mixed")
    gr = GameRenderer(width=args.width, height=args.height, cell=args.cell)
    gr.run_agent(agent, fps=args.fps, episodes=1)

def train_and_play(args):
    res = train(args)
    if getattr(args, "log", ""):
        pa = SimpleNamespace(
            log=args.log,
            out="models/train_plot.png",
            window=getattr(args, "window", 50),
            fig_width=8.0,
            fig_height=5.0,
            dpi=120,
        )
        plot_png(pa)
    if not hasattr(args, "model") or not args.model:
        args.model = "models/q_table.npy"
    play_q(args)

def plot_log(args):
    # 终端 ASCII 可视化：快速查看最近趋势（无需图形界面）
    if not args.log or not os.path.exists(args.log):
        print("no log")
        return
    with open(args.log, "r") as f:
        r = csv.DictReader(f)
        rewards = []
        scores = []
        rma = []
        sma = []
        for row in r:
            rv = float(row.get("reward", 0.0))
            sv = float(row.get("score", 0.0))
            rewards.append(rv)
            scores.append(sv)
            if "reward_ma" in row and row["reward_ma"]:
                rma.append(float(row["reward_ma"]))
            if "score_ma" in row and row["score_ma"]:
                sma.append(float(row["score_ma"]))
    if not rma:
        w = deque(maxlen=args.window)
        for v in rewards:
            w.append(v)
            rma.append(sum(w) / len(w))
    if not sma:
        w = deque(maxlen=args.window)
        for v in scores:
            w.append(v)
            sma.append(sum(w) / len(w))
    def ascii_plot(vals, width=60, height=12, ch="█", title=""):
        if not vals:
            print(title, "empty")
            return
        tail = vals[-width:] if len(vals) >= width else vals[:]
        vmin = min(tail)
        vmax = max(tail)
        if vmax == vmin:
            vmax = vmin + 1.0
        grid = [[" " for _ in range(len(tail))] for _ in range(height)]
        for i, v in enumerate(tail):
            y = int(round((v - vmin) / (vmax - vmin) * (height - 1)))
            y = max(0, min(height - 1, y))
            grid[y][i] = ch
        if title:
            print(title)
        for row in range(height - 1, -1, -1):
            print("".join(grid[row]))
        print("min", f"{vmin:.3f}", "max", f"{vmax:.3f}")
    ascii_plot(rma, width=args.plot_width, height=args.plot_height, ch="*", title="reward_ma")
    ascii_plot(sma, width=args.plot_width, height=args.plot_height, ch="+", title="score_ma")

def plot_png(args):
    # 生成 PNG 图片：reward/score 及其移动平均；可叠加 epsilon
    if not args.log or not os.path.exists(args.log):
        print("no log")
        return
    with open(args.log, "r") as f:
        r = csv.DictReader(f)
        ep = []
        rewards = []
        scores = []
        eps = []
        rma = []
        sma = []
        i = 0
        for row in r:
            i += 1
            ep.append(i)
            rewards.append(float(row.get("reward", 0.0)))
            scores.append(float(row.get("score", 0.0)))
            if "epsilon" in row and row["epsilon"]:
                eps.append(float(row["epsilon"]))
            if "reward_ma" in row and row["reward_ma"]:
                rma.append(float(row["reward_ma"]))
            if "score_ma" in row and row["score_ma"]:
                sma.append(float(row["score_ma"]))
    if not rma:
        w = deque(maxlen=args.window)
        for v in rewards:
            w.append(v)
            rma.append(sum(w) / len(w))
    if not sma:
        w = deque(maxlen=args.window)
        for v in scores:
            w.append(v)
            sma.append(sum(w) / len(w))
    # 读取 Q 更新与幅度
    q_update_mean = []
    q_delta_mean = []
    with open(args.log, "r") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, 1):
            try:
                q_update_mean.append(float(row.get("q_update_mean", "") or 0.0))
                q_delta_mean.append(float(row.get("q_delta_mean", "") or 0.0))
            except ValueError:
                q_update_mean.append(0.0)
                q_delta_mean.append(0.0)
    fig, ax = plt.subplots(2, 2, figsize=(args.fig_width, args.fig_height), dpi=args.dpi, constrained_layout=True)
    # 左上：训练收敛曲线（奖励与滑动平均）
    l1, = ax[0][0].plot(ep, rewards, color="#cccccc", label="reward")
    l2, = ax[0][0].plot(ep, rma, color="#1f77b4", label=f"reward_ma(w={args.window})")
    ax[0][0].set_title("reward and moving average")
    ax[0][0].set_xlabel("episode")
    ax[0][0].set_ylabel("reward")
    if eps:
        ax2 = ax[0][0].twinx()
        l3, = ax2.plot(ep[: len(eps)], eps, color="#ff7f0e", alpha=0.5, label="epsilon")
        ax2.set_ylabel("epsilon")
        lines = [l1, l2, l3]
        labels = [ln.get_label() for ln in lines]
        ax[0][0].legend(lines, labels, loc="upper right")
    else:
        ax[0][0].legend(loc="upper right")
    # 右上：探索度衰减
    if eps:
        ax[0][1].plot(ep[: len(eps)], eps, color="#ff7f0e")
    ax[0][1].set_title("epsilon decay")
    ax[0][1].set_xlabel("episode")
    ax[0][1].set_ylabel("epsilon")
    # 左下：Q 表更新幅度（单步平均 |ΔQ|）
    if q_update_mean:
        ax[1][0].plot(ep[: len(q_update_mean)], q_update_mean, color="#9467bd")
    ax[1][0].set_title("mean |ΔQ| per step (episode)")
    ax[1][0].set_xlabel("episode")
    ax[1][0].set_ylabel("|ΔQ| (mean)")
    # 右下：Q 表变化曲线（每集相对 q_before 的 |ΔQ| 平均）
    if q_delta_mean:
        ax[1][1].plot(ep[: len(q_delta_mean)], q_delta_mean, color="#d62728")
    ax[1][1].set_title("Q-table mean |ΔQ| per episode")
    ax[1][1].set_xlabel("episode")
    ax[1][1].set_ylabel("mean |Q - Q_prev_ep|")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("saved", args.out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--cell", type=int, default=24)
    p.add_argument("--fps", type=int, default=12)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--width", type=int, default=20)
    common.add_argument("--height", type=int, default=20)
    common.add_argument("--cell", type=int, default=24)
    common.add_argument("--fps", type=int, default=12)
    sub = p.add_subparsers(dest="mode")
    pt = sub.add_parser("train", parents=[common])
    pt.add_argument("--episodes", type=int, default=5000)
    pt.add_argument("--alpha", type=float, default=0.1)
    pt.add_argument("--gamma", type=float, default=0.9)
    pt.add_argument("--epsilon", type=float, default=0.2)
    pt.add_argument("--mode", type=str, choices=["q", "greedy", "mixed"], default="q")
    pt.add_argument("--model", type=str, default="models/q_table.npy")
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--eval-episodes", type=int, default=0)
    pt.add_argument("--log", type=str, default="")
    pt.add_argument("--window", type=int, default=50)
    pt.add_argument("--conv-threshold", dest="conv_threshold", type=float, default=0.05)
    pt.add_argument("--conv-patience", dest="conv_patience", type=int, default=3)
    pt.add_argument("--print-every", dest="print_every", type=int, default=0)
    pt.add_argument("--eps-decay", dest="eps_decay", type=float, default=0.995)
    pt.add_argument("--min-eps", dest="min_eps", type=float, default=0.05)
    pt.add_argument("--alpha-decay", dest="alpha_decay", type=float, default=1.0)
    pt.add_argument("--min-alpha", dest="min_alpha", type=float, default=0.0)
    pt.set_defaults(func=train)
    pq = sub.add_parser("play-q", parents=[common])
    pq.add_argument("--model", type=str, default="models/q_table.npy")
    pq.set_defaults(func=play_q)
    pg = sub.add_parser("play-greedy", parents=[common])
    pg.set_defaults(func=play_greedy)
    ph = sub.add_parser("human", parents=[common])
    ph.set_defaults(func=human)
    pu = sub.add_parser("play-unified", parents=[common])
    pu.set_defaults(func=play_unified)
    pa = sub.add_parser("auto", parents=[common])
    pa.add_argument("--episodes", type=int, default=5000)
    pa.add_argument("--alpha", type=float, default=0.1)
    pa.add_argument("--gamma", type=float, default=0.9)
    pa.add_argument("--epsilon", type=float, default=0.2)
    pa.add_argument("--model", type=str, default="models/q_table.npy")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--eval-episodes", type=int, default=50)
    pa.add_argument("--log", type=str, default="models/train_log.csv")
    pa.add_argument("--window", type=int, default=50)
    pa.add_argument("--conv-threshold", dest="conv_threshold", type=float, default=0.05)
    pa.add_argument("--conv-patience", dest="conv_patience", type=int, default=3)
    pa.add_argument("--print-every", dest="print_every", type=int, default=100)
    pa.add_argument("--eps-decay", dest="eps_decay", type=float, default=0.995)
    pa.add_argument("--min-eps", dest="min_eps", type=float, default=0.05)
    pa.add_argument("--alpha-decay", dest="alpha_decay", type=float, default=1.0)
    pa.add_argument("--min-alpha", dest="min_alpha", type=float, default=0.0)
    pa.set_defaults(func=train_and_play)
    pl = sub.add_parser("plot-log")
    pl.add_argument("--log", type=str, default="models/train_log.csv")
    pl.add_argument("--window", type=int, default=50)
    pl.add_argument("--plot-width", dest="plot_width", type=int, default=60)
    pl.add_argument("--plot-height", dest="plot_height", type=int, default=12)
    pl.set_defaults(func=plot_log)
    pp = sub.add_parser("plot-png")
    pp.add_argument("--log", type=str, default="models/train_log.csv")
    pp.add_argument("--out", type=str, default="models/train_plot.png")
    pp.add_argument("--window", type=int, default=50)
    pp.add_argument("--fig-width", dest="fig_width", type=float, default=8.0)
    pp.add_argument("--fig-height", dest="fig_height", type=float, default=5.0)
    pp.add_argument("--dpi", type=int, default=120)
    pp.set_defaults(func=plot_png)
    args = p.parse_args()
    if args.mode is None:
        choice = show_menu(width=args.width, height=args.height, cell=args.cell)
        if choice == "human":
            human(args)
            return
        if choice == "play-greedy":
            play_greedy(args)
            return
        if choice == "play-q":
            if not hasattr(args, "model"):
                args.model = "models/q_table.npy"
            play_q(args)
            return
        if choice == "play-unified":
            play_unified(args)
            return
        if choice == "auto":
            if not hasattr(args, "episodes"):
                args.episodes = 5000
            if not hasattr(args, "alpha"):
                args.alpha = 0.1
            if not hasattr(args, "gamma"):
                args.gamma = 0.9
            if not hasattr(args, "epsilon"):
                args.epsilon = 0.2
            if not hasattr(args, "model"):
                args.model = "models/q_table.npy"
            if not hasattr(args, "seed"):
                args.seed = 42
            if not hasattr(args, "eval_episodes"):
                args.eval_episodes = 50
            if not hasattr(args, "log"):
                args.log = "models/train_log.csv"
            if not hasattr(args, "window"):
                args.window = 50
            if not hasattr(args, "conv_threshold"):
                args.conv_threshold = 0.05
            if not hasattr(args, "conv_patience"):
                args.conv_patience = 3
            if not hasattr(args, "print_every"):
                args.print_every = 100
            train_and_play(args)
            return
        return
    args.func(args)

if __name__ == "__main__":
    main()
