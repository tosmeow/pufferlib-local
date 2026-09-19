# Lag-0 early scoring and above-wall visits

Final score alone does not distinguish rapid scoring strategies from slower full clears. This diagnostic replays the selected starting policy and its five-minute T4 continuation on the common lag-0, volatility-0 game, with greedy actions and 32 environment seeds (100000–100031). All reported times are simulated game seconds, not evaluation wall time.

| Metric | Before continuation | After continuation |
| --- | ---: | ---: |
| Final score | 864.00 | 864.00 |
| First observed fully-above-wall entry (seconds) | 60.50 | 78.80 |
| Total observed time above wall (seconds) | 6.73 | 6.33 |
| Time to first 432 points (seconds) | 142.93 | 128.85 |
| Full-game completion (seconds) | 270.77 | 266.10 |
| Score after 15 seconds | 13.50 | 13.50 |
| Score after 30 seconds | 54.50 | 49.00 |
| Score after 60 seconds | 183.50 | 187.50 |
| Score after 120 seconds | 421.50 | 418.00 |
| Score after 180 seconds | 547.00 | 571.75 |

The ball was observed entirely above the original brick wall in all 32 episodes for each policy. The criterion is ball_top_y + 32 <= 50 pixels, sampled every four simulation frames. It can miss shorter visits and does not establish intentional aiming, tunnel creation, or preferential use of an available opening.

The native physics permit above-wall play: bricks start at y=50, the ball is 32 pixels tall, and the top wall reflects the ball. Paddle impact position controls outgoing angle. The policy observes the complete brick-state array, ball position and velocity.

Two training choices are relevant hypotheses, not proven causes of the current strategy:

- `pufferlib/torch_pufferl.py` clips rewards to [-1, 1]. Native bricks award 1, 4 or 7 points, so the training reward loses that point-value distinction, and multiple hits within one reward interval may also be compressed. Rapid repeated hits remain rewarded; clipping does not make tunnelling impossible.
- Gamma is 0.97212466 per policy decision, with 15 decisions per simulated second. A reward five seconds later has direct discount weight 0.120; ten seconds later, 0.0144. This can make delayed setup costs harder to justify, while bootstrapped values can still carry information beyond a rollout. There is no hard 2.4-second memory cutoff.

A useful next objective is early raw score (for example score at 30/60 seconds, or area under the score curve over a fixed window), with win rate retained as a constraint. Track first usable gap, time from gap availability to above-wall entry, and repeat entries as diagnostics. Reward scaling that preserves point ratios and a longer discount horizon are candidate controlled changes. Giving a direct tunnel-entry bonus would alter the task to explicitly favor that tactic, so it is not needed merely to measure whether the tactic emerges.

[Raw episodes and metrics](evaluations/lag0_tunnelling_1788712717435756000/results.json)
