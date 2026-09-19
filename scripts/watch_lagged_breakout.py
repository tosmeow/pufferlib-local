"""Watch four trained lag policies play synchronized, unlagged Breakout games."""

import argparse
from collections import deque
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_lagged_breakout import float_view

LAGS = (0, 2, 4, 8)
COLORS = ('#55c7e6', '#87d5aa', '#e6bd68', '#c6a1ed')
BG, CARD, FIELD, FG, MUTED = '#11181f', '#1b2630', '#061818', '#edf3f5', '#9eafbb'


class Player:
    def __init__(self, experiment, lag, policy_seed, game_seed):
        from pufferlib import _C
        from pufferlib.torch_pufferl import load_policy
        if _C.env_name != 'lagged_breakout' or _C.precision_bytes != 4:
            raise RuntimeError('Build lagged_breakout with --cpu or --float first')
        self.folder = experiment / f'lag_{lag}' / f'seed_{policy_seed}'
        self.config = json.loads((self.folder / 'config.json').read_text())
        if self.config['env']['action_lag'] != lag or self.config['env']['frameskip'] != 4:
            raise ValueError('The viewer requires the saved four-microstep policies at their labeled lag')
        self.lag, self.policy_seed = lag, policy_seed
        self.vec = None
        self._create_game(game_seed)
        try:
            self.policy = load_policy(self.eval_config, self.vec, device=torch.device('cpu')).eval()
            self._reset_bookkeeping()
        except BaseException:
            self.close()
            raise

    def _create_game(self, game_seed):
        from pufferlib import _C
        self.eval_config = copy.deepcopy(self.config)
        self.eval_config['vec'].update(total_agents=1, num_buffers=1, num_threads=1)
        self.eval_config['env'].update(action_lag=0, volatility=0.0, seed=game_seed, reward_interval=4)
        self.eval_config['load_model_path'] = str(self.folder / 'policy.bin')
        self.eval_config['load_id'] = None
        self.vec = _C.create_vec(self.eval_config, 0)
        self.obs = float_view(self.vec.obs_ptr, (1, self.vec.obs_size))
        self.rewards = float_view(self.vec.rewards_ptr, (1,))
        self.terminals = float_view(self.vec.terminals_ptr, (1,))
        self.vec.reset()

    def _reset_bookkeeping(self):
        self.state = self.policy.initial_state(1, device=torch.device('cpu'))
        self.screen = self.obs[0].copy()
        self.score = self.steps = self.switches = self.reversals = self.noops = 0
        self.action = 0
        self.finished = self.capped = False
        self.trail = deque(maxlen=14)

    def reset(self, game_seed):
        # Recreate the vector: reset() alone continues the old serve RNG stream.
        self.close()
        self._create_game(game_seed)
        self._reset_bookkeeping()

    def step(self, max_steps):
        if self.finished:
            return
        with torch.inference_mode():
            logits, _, self.state = self.policy.forward_eval(torch.from_numpy(self.obs), self.state)
            action = int(logits.argmax(dim=-1).item())
        if self.steps:
            self.switches += action != self.action
            self.reversals += action != self.action and action != 0 and self.action != 0
        self.noops += action == 0
        self.action = action
        before = self.obs[0].copy()
        actions = np.array([[action]], dtype=np.float32)
        self.vec.cpu_step(actions.ctypes.data)
        self.steps += 1
        self.score += int(round(float(self.rewards[0])))
        if self.terminals[0]:
            self.finished = True
            # The native step auto-resets. Hold the previous visible game instead
            # of displaying a new episode with an old episode's final score.
            self.screen = before
            if self.score == 864:
                self.screen[10:118] = 1
        else:
            self.screen = self.obs[0].copy()
            if self.screen[8] != before[8]:
                self.trail.clear()
            self.finished = self.capped = self.steps >= max_steps
        self.trail.append((float(self.screen[2] * 576 + 16), float(self.screen[3] * 330 + 16)))

    def snapshot(self):
        return dict(lag=self.lag, policy_seed=self.policy_seed, score=self.score,
                    steps=self.steps, action=self.action, finished=self.finished, capped=self.capped,
                    switches=self.switches, reversals=self.reversals, noops=self.noops,
                    observation=self.screen.tolist())

    def close(self):
        if self.vec is not None:
            self.vec.close()
            self.vec = None


class Comparison:
    def __init__(self, experiment, policy_seeds, game_seed=100000, max_steps=120000):
        self.experiment, self.max_steps = experiment, max_steps
        self.seed, self.tick = game_seed, 0
        self.players = []
        try:
            for lag, seed in zip(LAGS, policy_seeds):
                self.players.append(Player(experiment, lag, seed, game_seed))
        except BaseException:
            self.close()
            raise

    def step(self):
        if not all(p.finished for p in self.players):
            for player in self.players:
                player.step(self.max_steps)
            self.tick += 1

    def reset(self, game_seed=None):
        self.seed = self.seed if game_seed is None else game_seed
        self.tick = 0
        for player in self.players:
            player.reset(self.seed)

    def select(self, index, policy_seed):
        replacement = Player(self.experiment, LAGS[index], policy_seed, self.seed)
        self.players[index].close()
        self.players[index] = replacement
        self.reset()

    def snapshot(self):
        return dict(seed=self.seed, tick=self.tick, seconds=self.tick * 4 / 60,
                    eval_lag=0, eval_volatility=0.0, players=[p.snapshot() for p in self.players])

    def close(self):
        for player in self.players:
            player.close()


class Viewer:
    def __init__(self, comparison, speed, paused, status_file):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.game, self.status_file = comparison, status_file
        self.root = tk.Tk()
        self.root.title('Lagged Breakout — live policy comparison')
        self.root.configure(bg=BG)
        width, height = min(1280, self.root.winfo_screenwidth() - 60), min(900, self.root.winfo_screenheight() - 90)
        self.root.geometry(f'{width}x{height}+30+30')
        self.root.minsize(920, 680)
        self.paused = paused
        self.speed = tk.StringVar(value=f'{speed:g}×')
        self.seed = tk.StringVar(value=str(comparison.seed))
        self.auto_next = tk.BooleanVar(value=True)
        self.trails = tk.BooleanVar(value=True)
        self.next_time = time.monotonic()
        self.completed_at = None
        self.last_status = 0.0
        self.closed = False
        self.sprite_cache = {}
        self.sprite = tk.PhotoImage(file=str(ROOT / 'resources/shared/puffers_128.png')).subsample(4, 4)
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground=CARD, background=CARD, foreground=FG,
                        arrowcolor=FG, bordercolor='#40505e', padding=4)
        self.root.option_add('*TCombobox*Listbox.background', CARD)
        self.root.option_add('*TCombobox*Listbox.foreground', FG)
        self.root.option_add('*Font', ('Helvetica Neue', 12))

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', padx=20, pady=(14, 8))
        tk.Label(top, text='One game. Four learned behaviors.', bg=BG, fg=FG,
                 font=('Helvetica Neue', 22, 'bold')).pack(side='left')
        self.clock_label = tk.Label(top, text='', bg=BG, fg='#79d4b0', font=('Menlo', 13))
        self.clock_label.pack(side='right')
        tk.Label(self.root, text='All four play the original game: no action delay, no diffusion. Labels show each policy’s training lag.',
                 bg=BG, fg=MUTED, anchor='w').pack(fill='x', padx=20)
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill='x', padx=20, pady=(12, 10))
        self.pause_button = self.button(controls, 'Play' if paused else 'Pause', self.toggle)
        self.button(controls, 'Step', self.single_step)
        self.button(controls, 'Replay seed', self.restart)
        self.button(controls, 'Next seed', self.next_seed)
        tk.Label(controls, text='Game seed', bg=BG, fg=MUTED).pack(side='left', padx=(12, 5))
        entry = tk.Entry(controls, textvariable=self.seed, width=9, bg=CARD, fg=FG, insertbackground=FG,
                         relief='flat', highlightthickness=1, highlightbackground='#40505e')
        entry.pack(side='left', ipady=6)
        entry.bind('<Return>', lambda _: self.restart())
        tk.Label(controls, text='Speed', bg=BG, fg=MUTED).pack(side='left', padx=(12, 5))
        speed_menu = ttk.Combobox(controls, textvariable=self.speed, values=['0.25×', '0.5×', '1×', '2×', '4×', '8×'], width=5, state='readonly')
        speed_menu.pack(side='left')
        speed_menu.bind('<<ComboboxSelected>>', lambda _: self.reschedule())
        tk.Checkbutton(controls, text='Trails', variable=self.trails, bg=BG, fg=MUTED,
                       selectcolor=CARD, activebackground=BG, activeforeground=FG).pack(side='left', padx=10)

        boards = tk.Frame(self.root, bg=BG)
        boards.pack(fill='both', expand=True, padx=14)
        self.canvases, self.metrics, self.seed_choices = [], [], []
        for i, lag in enumerate(LAGS):
            boards.columnconfigure(i % 2, weight=1, uniform='columns')
            boards.rowconfigure(i // 2, weight=1, uniform='rows')
            card = tk.Frame(boards, bg=CARD, highlightthickness=1, highlightbackground='#34434f')
            card.grid(row=i // 2, column=i % 2, sticky='nsew', padx=6, pady=6)
            header = tk.Frame(card, bg=CARD)
            header.pack(fill='x', padx=12, pady=8)
            tk.Label(header, text=f'TRAINED LAG {lag}  ·  {lag / 60 * 1000:.0f} ms', fg=COLORS[i], bg=CARD,
                     font=('Helvetica Neue', 13, 'bold')).pack(side='left')
            choice = tk.StringVar(value=str(comparison.players[i].policy_seed))
            available = sorted(p.name.removeprefix('seed_') for p in (comparison.experiment / f'lag_{lag}').glob('seed_*') if (p / 'policy.bin').exists())
            menu = ttk.Combobox(header, textvariable=choice, values=available, width=4, state='readonly')
            menu.pack(side='right')
            tk.Label(header, text='Policy seed ', bg=CARD, fg=MUTED).pack(side='right')
            menu.bind('<<ComboboxSelected>>', lambda event, index=i: self.change_policy(index))
            self.seed_choices.append(choice)
            metrics = tk.Label(card, text='', bg=CARD, fg=FG, anchor='w', font=('Menlo', 11))
            metrics.pack(side='bottom', fill='x', padx=12, pady=8)
            self.metrics.append(metrics)
            canvas = tk.Canvas(card, bg=FIELD, highlightthickness=0)
            canvas.pack(fill='both', expand=True, padx=1)
            self.canvases.append(canvas)
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill='x', padx=20, pady=(5, 12))
        self.message = tk.Label(bottom, text='Same initial seed and clock; actions make the paths diverge.  Space: pause  ·  R: replay  ·  N: next seed',
                                bg=BG, fg=MUTED, anchor='w', font=('Helvetica Neue', 10))
        self.message.pack(side='left')
        tk.Checkbutton(bottom, text='Next seed when all finish', variable=self.auto_next, bg=BG, fg=MUTED,
                       selectcolor=CARD, activebackground=BG, activeforeground=FG).pack(side='right')
        self.root.bind('<space>', lambda event: self.shortcut(event, self.toggle))
        self.root.bind('<r>', lambda event: self.shortcut(event, self.restart))
        self.root.bind('<n>', lambda event: self.shortcut(event, self.next_seed))
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(50, self.loop)

    def button(self, parent, text, command):
        button = self.tk.Button(parent, text=text, command=command, bg=CARD, fg=FG,
                                activebackground='#34434f', activeforeground=FG, relief='flat', padx=12, pady=6)
        button.pack(side='left', padx=(0, 6))
        return button

    def shortcut(self, event, action):
        if event.widget.winfo_class() not in ('Entry', 'TCombobox'):
            action()
            return 'break'

    def reschedule(self):
        self.next_time = time.monotonic()

    def toggle(self):
        self.paused = not self.paused
        self.pause_button.configure(text='Play' if self.paused else 'Pause')
        self.reschedule()

    def single_step(self):
        self.paused = True
        self.pause_button.configure(text='Play')
        self.game.step()

    def restart(self):
        try:
            seed = int(self.seed.get())
            if not 0 <= seed < 2**31:
                raise ValueError
        except ValueError:
            self.message.configure(text='Choose an integer game seed from 0 to 2147483647.', fg='#e6bd68')
            return
        self.game.reset(seed)
        self.completed_at = None
        self.reschedule()
        self.message.configure(text='All four restarted together from the same seed.', fg=MUTED)

    def next_seed(self):
        self.seed.set(str((self.game.seed + 1) % 2**31))
        self.restart()

    def change_policy(self, index):
        previous = self.game.players[index].policy_seed
        try:
            self.game.select(index, int(self.seed_choices[index].get()))
        except Exception as exc:
            self.seed_choices[index].set(str(previous))
            self.message.configure(text=str(exc), fg='#e6bd68')
            return
        self.completed_at = None
        self.reschedule()
        self.message.configure(text='Policy changed. All four games restarted together for a fair replay.', fg=MUTED)

    def draw(self, i, player):
        canvas = self.canvases[i]
        width, height = canvas.winfo_width(), canvas.winfo_height()
        scale = min(width / 576, height / 330)
        ox, oy = (width - 576 * scale) / 2, (height - 330 * scale) / 2
        def rectangle(x, y, w, h, color):
            canvas.create_rectangle(ox + x * scale, oy + y * scale,
                                    ox + (x + w) * scale, oy + (y + h) * scale, fill=color, outline='')
        canvas.delete('all')
        obs = player.screen
        brick_colors = ('#e16c70', '#e8965a', '#e4c768', '#80c68a', '#69bfcd', '#7397d7')
        for index, destroyed in enumerate(obs[10:118]):
            if destroyed < .5:
                row, col = divmod(index, 18)
                rectangle(col * 32, 50 + row * 12, 31, 11, brick_colors[row])
        if self.trails.get():
            trail = list(player.trail)
            for j in range(1, len(trail)):
                canvas.create_line(ox + trail[j - 1][0] * scale, oy + trail[j - 1][1] * scale,
                                   ox + trail[j][0] * scale, oy + trail[j][1] * scale,
                                   fill=('#28494d' if j < len(trail) / 2 else '#477078'), width=max(1, 2 * scale))
        rectangle(float(obs[0] * 576), float(obs[1] * 330), float(obs[9] * 62), 8, COLORS[i])
        size = max(8, round(32 * scale))
        if size not in self.sprite_cache:
            self.sprite_cache[size] = self.sprite.zoom(size).subsample(32)
        canvas.create_image(ox + float(obs[2] * 576 + 16) * scale,
                            oy + float(obs[3] * 330 + 16) * scale, image=self.sprite_cache[size])
        balls = max(0, round(float(obs[8]) * 5))
        canvas.create_text(ox + 12 * scale, oy + 18 * scale, anchor='w', text=f'{player.score:03d} / 864',
                           fill=FG, font=('Menlo', max(10, round(15 * scale)), 'bold'))
        canvas.create_text(ox + 564 * scale, oy + 18 * scale, anchor='e', text=f'Balls {balls}',
                           fill=MUTED, font=('Menlo', max(9, round(12 * scale))))
        action = ('HOLD', 'LEFT', 'RIGHT')[player.action]
        rate = player.switches / max(1, player.steps - 1)
        self.metrics[i].configure(text=f'{action:<5}   Changes {rate:.0%}   Hold {player.noops / max(1, player.steps):.0%}   t={player.steps / 15:.1f}s')
        if player.finished:
            label = 'BOARD CLEARED' if player.score == 864 else 'TIME LIMIT' if player.capped else 'EPISODE FINISHED'
            rectangle(100, 175, 376, 65, '#172e34')
            canvas.create_text(width / 2, oy + 196 * scale, text=label, fill=COLORS[i], font=('Helvetica Neue', 14, 'bold'))
            canvas.create_text(width / 2, oy + 223 * scale, text=f'{player.score} points · waiting for the other games', fill=FG, font=('Helvetica Neue', 11))

    def loop(self):
        if self.closed:
            return
        now = time.monotonic()
        if not self.paused:
            if now - self.next_time > .5:
                self.next_time = now
            interval = 4 / 60 / float(self.speed.get().removesuffix('×'))
            for _ in range(8):
                if now < self.next_time:
                    break
                self.game.step()
                self.next_time += interval
        for i, player in enumerate(self.game.players):
            self.draw(i, player)
        seconds = self.game.tick / 15
        mode = 'PAUSED' if self.paused else 'LIVE'
        self.clock_label.configure(text=f'{mode}  {int(seconds // 60):02d}:{seconds % 60:04.1f}  |  seed {self.game.seed}')
        if all(p.finished for p in self.game.players):
            self.completed_at = now if self.completed_at is None else self.completed_at
            if self.auto_next.get() and not self.paused and now - self.completed_at >= 3:
                self.next_seed()
        if self.status_file and now - self.last_status >= 1:
            status = {**self.game.snapshot(), 'paused': self.paused, 'speed': self.speed.get()}
            temporary = self.status_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(status))
            temporary.replace(self.status_file)
            self.last_status = now
        self.root.after(16, self.loop)

    def close(self):
        self.closed = True
        self.game.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'benchmarks/lagged_breakout/experiment_03_gpu')
    parser.add_argument('--policy-seed', type=int, default=42)
    parser.add_argument('--game-seed', type=int, default=100000)
    parser.add_argument('--speed', type=float, default=1.0, choices=(.25, .5, 1, 2, 4, 8))
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--max-steps', type=int, default=120000)
    parser.add_argument('--headless-steps', type=int)
    parser.add_argument('--status-file', type=Path)
    args = parser.parse_args()
    if not 0 <= args.game_seed < 2**31 or args.max_steps <= 0:
        parser.error('game-seed must fit signed 32 bits and max-steps must be positive')
    if args.headless_steps is not None and args.headless_steps < 0:
        parser.error('headless-steps must be nonnegative')
    torch.set_num_threads(1)
    game = Comparison(args.experiment.resolve(), [args.policy_seed] * 4, args.game_seed, args.max_steps)
    try:
        if args.headless_steps is not None:
            for _ in range(args.headless_steps):
                game.step()
            print(json.dumps(game.snapshot()))
        else:
            Viewer(game, args.speed, args.paused, args.status_file).run()
    finally:
        game.close()


if __name__ == '__main__':
    main()
