#include <assert.h>
#include <stdio.h>
#include "ocean/lagged_breakout/lagged_breakout.h"

static void setup(LaggedBreakout* env, int lag, float volatility, int frameskip) {
    *env = (LaggedBreakout){
        .rng = 123, .action_lag = lag, .volatility = volatility,
        .reward_interval = frameskip,
        .game = {
            .frameskip = frameskip, .width = 576, .height = 330,
            .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
            .ball_width = 32, .ball_height = 32,
            .brick_width = 32, .brick_height = 12, .brick_rows = 6, .brick_cols = 18,
            .initial_ball_speed = 256, .max_ball_speed = 448, .paddle_speed = 620,
        },
    };
    allocate(env);
    c_reset(env);
}

static void open_space(LaggedBreakout* env) {
    Breakout* game = &env->game;
    for (int i = 0; i < game->num_bricks; i++) game->brick_states[i] = 1.0f;
    game->balls_fired = 1;
    game->ball_x = 200.0f;
    game->ball_y = 200.0f;
    game->ball_vx = 1.0f;
    game->ball_vy = 0.25f;
}

static void test_original_parity(void) {
    LaggedBreakout env;
    setup(&env, 0, 0, 1);
    Breakout original = env.game;
    original.client = NULL;
    original.brick_x = original.brick_y = original.brick_states = NULL;
    breakout_allocate(&original);
    breakout_reset(&original);
    int terminals = 0;
    float score = 0;
    for (int i = 0; i < 100000; i++) {
        float target = original.ball_x + original.ball_width / 2.0f;
        float paddle = original.paddle_x + original.paddle_width / 2.0f;
        float action = i % 3000 < 1000 ? NOOP : target < paddle ? LEFT : RIGHT;
        env.actions[0] = original.actions[0] = action;
        c_step(&env);
        breakout_step(&original);
        assert(memcmp(env.observations, original.observations, 118 * sizeof(float)) == 0);
        assert(env.rewards[0] == original.rewards[0]);
        assert(env.terminals[0] == original.terminals[0]);
        assert(env.game.rng == original.rng);
        score += env.rewards[0];
        terminals += env.terminals[0] != 0;
    }
    assert(terminals > 10 && score > 0);
    breakout_free_allocated(&original);
    free_allocated(&env);
    puts("PASS: 100000-frame original parity, including scores and episode boundaries");
}

static void test_delay(void) {
    for (int lag = 0; lag <= LAGGED_MAX_LAG; lag++) {
        LaggedBreakout env;
        setup(&env, lag, 0, 1);
        open_space(&env);
        float start = env.game.paddle_x;
        for (int i = 0; i < lag; i++) {
            env.actions[0] = RIGHT;
            c_step(&env);
            assert(env.game.paddle_x == start);
        }
        env.actions[0] = RIGHT;
        c_step(&env);
        assert(fabsf(env.game.paddle_x - start - env.game.paddle_speed * TICK_RATE) < 0.0001f);
        free_allocated(&env);
    }
    LaggedBreakout env;
    setup(&env, 3, 0, 4);
    open_space(&env);
    float start = env.game.paddle_x;
    float distance = env.game.paddle_speed * TICK_RATE;
    env.actions[0] = RIGHT;
    c_step(&env);
    assert(fabsf(env.game.paddle_x - start - distance) < 0.0001f);
    for (int i = 0; i < 3; i++) assert(env.observations[LAGGED_QUEUE_OBS + i] == 1);
    env.actions[0] = LEFT;
    c_step(&env);
    assert(fabsf(env.game.paddle_x - start - 3 * distance) < 0.0001f);
    for (int i = 0; i < 3; i++) assert(env.observations[LAGGED_QUEUE_OBS + i] == -1);
    for (int i = 3; i < LAGGED_MAX_LAG; i++) assert(env.observations[LAGGED_QUEUE_OBS + i] == 0);
    free_allocated(&env);
    puts("PASS: all 65 lag values, exact delivery, FIFO order, and delivery inside a decision step");
}

static void test_reward_clock_and_reset(void) {
    LaggedBreakout env;
    setup(&env, 2, 0, 1);
    env.reward_interval = 4;
    open_space(&env);
    env.game.brick_states[90] = 0;
    env.game.ball_x = 8;
    env.game.ball_y = 125;
    env.game.ball_vx = 0;
    env.game.ball_vy = -4;
    env.actions[0] = RIGHT;
    c_step(&env);
    assert(env.game.score == 1 && env.rewards[0] == 0 && env.pending_reward == 1);
    c_step(&env);
    assert(env.rewards[0] == 0);
    c_step(&env);
    assert(env.rewards[0] == 0);
    c_step(&env);
    assert(env.rewards[0] == 1 && env.pending_reward == 0 && env.reward_tick == 0);
    // Unpaid points must be returned on termination even off the reward clock.
    env.pending_reward = 5;
    env.game.score = 6;
    env.game.num_balls = 0;
    env.game.ball_y = env.game.paddle_y + env.game.paddle_height;
    env.game.ball_vy = 1;
    env.game.frameskip = 4;
    c_step(&env);
    assert(env.terminals[0] == 1 && env.rewards[0] == 5);
    assert(env.game.tick == 0 && env.game.balls_fired == 0);
    assert(env.pending_reward == 0 && env.reward_tick == 0);
    assert(env.log.n == 1 && env.log.episode_return == 6);
    for (int i = 0; i < LAGGED_MAX_LAG; i++) {
        assert(env.action_queue[i] == 0 && env.observations[LAGGED_QUEUE_OBS + i] == 0);
    }
    c_reset(&env);
    assert(env.rewards[0] == 0 && env.terminals[0] == 0);
    free_allocated(&env);
    puts("PASS: fixed reward clock, terminal payout, and clearing pending actions on reset");
}

static void test_diffusion(void) {
    const int samples = 40000;
    for (int horizon = 1; horizon <= 8; horizon *= 8) {
        LaggedBreakout env;
        setup(&env, 0, 6, 1);
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        for (int sample = 0; sample < samples; sample++) {
            open_space(&env);
            for (int j = 0; j < horizon; j++) c_step(&env);
            assert(env.game.ball_vx == 1.0f && env.game.ball_vy == 0.25f);
            double x = env.game.ball_x - 200.0 - horizon;
            double y = env.game.ball_y - 200.0 - 0.25 * horizon;
            sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y;
        }
        double expected = 36.0 * horizon / 60.0;
        double mean_x = sx / samples, mean_y = sy / samples;
        assert(fabs(mean_x) < 5 * sqrt(expected / samples));
        assert(fabs(mean_y) < 5 * sqrt(expected / samples));
        assert(fabs(sxx / samples - mean_x*mean_x - expected) < 0.03 * expected);
        assert(fabs(syy / samples - mean_y*mean_y - expected) < 0.03 * expected);
        assert(fabs(sxy / samples - mean_x*mean_y) < 0.03 * expected);
        printf("PASS: diffusion horizon=%d mean=(%.4f, %.4f), variance=(%.4f, %.4f), expected=%.4f\n",
            horizon, mean_x, mean_y, sxx/samples-mean_x*mean_x, syy/samples-mean_y*mean_y, expected);
        free_allocated(&env);
    }
}

static void test_seed_and_collisions(void) {
    LaggedBreakout first, second;
    setup(&first, 5, 6, 4);
    setup(&second, 5, 6, 4);
    for (int i = 0; i < 10000; i++) {
        first.actions[0] = second.actions[0] = i % 3;
        c_step(&first); c_step(&second);
        assert(memcmp(first.observations, second.observations, LAGGED_OBS_SIZE * sizeof(float)) == 0);
        assert(first.rewards[0] == second.rewards[0]);
        for (int j = 0; j < LAGGED_OBS_SIZE; j++) assert(isfinite(first.observations[j]));
        assert(first.game.ball_x >= 0 && first.game.ball_x + first.game.ball_width <= first.game.width);
        assert(first.game.ball_y >= 0);
    }
    // A noisy outward step from a contact must not escape the playfield.
    open_space(&first);
    first.game.ball_x = 0;
    step_frame_with_noise(&first.game, NOOP, -5, 0);
    assert(first.game.ball_x >= 0 && first.game.ball_vx == -1);
    first.game.ball_x = first.game.width - first.game.ball_width;
    first.game.ball_vx = 1;
    step_frame_with_noise(&first.game, NOOP, 5, 0);
    assert(first.game.ball_x + first.game.ball_width <= first.game.width);
    assert(first.game.ball_vx == -1);
    uint64_t noise_rng = first.noise_rng;
    first.volatility = 0;
    float x, y;
    lagged_noise(&first, &x, &y);
    assert(x == 0 && y == 0 && first.noise_rng == noise_rng);
    free_allocated(&first); free_allocated(&second);
    puts("PASS: reproducible independent noise, finite rollouts, wall contacts, and no RNG draws at sigma=0");
}

int main(void) {
    test_original_parity();
    test_delay();
    test_reward_clock_and_reset();
    test_diffusion();
    test_seed_and_collisions();
    return 0;
}
