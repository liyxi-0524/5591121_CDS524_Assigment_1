import pygame
from ..game import SnakeEnv
from ..agents.unified import UnifiedAgent
from ..utils import relative_action, opposite
import pygame.freetype

class GameRenderer:
    # 基于 pygame 的最小渲染器：负责绘制网格、食物和蛇，并提供两种运行模式
    def __init__(self, width=20, height=20, cell=24):
        self.cell = cell
        self.env = SnakeEnv(width=width, height=height)
        pygame.init()
        self.surface = pygame.display.set_mode((width * cell, height * cell))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

    def draw(self):
        # 单帧绘制：深灰背景、红色食物、绿色蛇体（头部更亮）
        self.surface.fill((30, 30, 30))
        fx, fy = self.env.food
        pygame.draw.rect(self.surface, (220, 50, 50), (fx * self.cell, fy * self.cell, self.cell, self.cell))
        for i, (x, y) in enumerate(self.env.snake):
            color = (50, 200, 50) if i == 0 else (40, 140, 40)
            pygame.draw.rect(self.surface, color, (x * self.cell, y * self.cell, self.cell, self.cell))
        pygame.display.flip()

    def run_human(self, fps=10):
        # 人类控制：方向键表示绝对方向；禁止立即反向
        self.env.reset()
        running = True
        desired = self.env.direction
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        if opposite(self.env.direction) != "UP":
                            desired = "UP"
                    elif event.key == pygame.K_DOWN:
                        if opposite(self.env.direction) != "DOWN":
                            desired = "DOWN"
                    elif event.key == pygame.K_LEFT:
                        if opposite(self.env.direction) != "LEFT":
                            desired = "LEFT"
                    elif event.key == pygame.K_RIGHT:
                        if opposite(self.env.direction) != "RIGHT":
                            desired = "RIGHT"
            a = relative_action(self.env.direction, desired)
            _, _, done, _ = self.env.step(a)
            self.draw()
            if done:
                # 一局结束后自动重新开始
                self.env.reset()
                desired = self.env.direction
            self.clock.tick(fps)
        pygame.quit()

    def run_agent(self, agent, fps=15, episodes=1):
        # Agent 运行：UnifiedAgent 支持 Q 学习与贪心模式
        for _ in range(episodes):
            self.env.reset()
            done = False
            if isinstance(agent, UnifiedAgent):
                agent.set_env(self.env)
            while not done:
                s = self.env.get_state()
                a = agent.act(s, training=False) if isinstance(agent, UnifiedAgent) else agent.act(self.env)
                _, _, done, _ = self.env.step(a)
                self.draw()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                self.clock.tick(fps)
        pygame.quit()

def show_menu(width=20, height=20, cell=24):
    pygame.init()
    W, H = width * cell, height * cell
    surface = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Snake - Menu")
    clock = pygame.time.Clock()
    font = pygame.freetype.SysFont(None, max(14, cell))
    title_font = pygame.freetype.SysFont(None, max(22, int(cell * 1.3)))
    options = [
        ("玩家手动", "human"),
        ("训练至收敛并演示", "auto"),
        ("Agent (Q-learning)", "play-q"),
        ("Agent (Unified/mixed)", "play-unified"),
        ("Agent (Greedy)", "play-greedy"),
        ("退出", "quit"),
    ]
    btn_w = int(W * 0.6)
    btn_h = int(cell * 1.6)
    spacing = int(btn_h * 0.6)
    start_y = (H - (len(options) * btn_h + (len(options) - 1) * spacing)) // 2
    selected = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % len(options)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % len(options)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    pygame.quit()
                    return options[selected][1]
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                for i, (label, value) in enumerate(options):
                    x = (W - btn_w) // 2
                    y = start_y + i * (btn_h + spacing)
                    if x <= mx <= x + btn_w and y <= my <= y + btn_h:
                        pygame.quit()
                        return value
        surface.fill((30, 30, 30))
        title = "选择模式"
        title_rect = title_font.get_rect(title)
        title_font.render_to(surface, ((W - title_rect.width) // 2, start_y - int(1.8 * btn_h)), title, (230, 230, 230))
        mx, my = pygame.mouse.get_pos()
        for i, (label, value) in enumerate(options):
            x = (W - btn_w) // 2
            y = start_y + i * (btn_h + spacing)
            hovered = (x <= mx <= x + btn_w and y <= my <= y + btn_h)
            is_sel = (i == selected)
            bg = (60, 60, 60)
            if hovered or is_sel:
                bg = (80, 120, 80) if value != "quit" else (120, 80, 80)
            pygame.draw.rect(surface, bg, (x, y, btn_w, btn_h), border_radius=8)
            pygame.draw.rect(surface, (100, 100, 100), (x, y, btn_w, btn_h), width=2, border_radius=8)
            txt_rect = font.get_rect(label)
            font.render_to(surface, (x + (btn_w - txt_rect.width) // 2, y + (btn_h - txt_rect.height) // 2), label, (240, 240, 240))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    return "quit"
