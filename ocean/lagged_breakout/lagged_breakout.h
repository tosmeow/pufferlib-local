#ifndef PUFFER_LAGGED_BREAKOUT_H
#define PUFFER_LAGGED_BREAKOUT_H

#include <stdint.h>
#include <stdio.h>

// Reuse the original game and renderer, namespacing its environment entry points.
#define Log BreakoutLog
#define init breakout_init
#define allocate breakout_allocate
#define free_allocated breakout_free_allocated
#define c_reset breakout_reset
#define c_step breakout_step
#define c_close breakout_close
#define c_render breakout_render
#include "../breakout/breakout.h"
#undef Log
#undef init
#undef allocate
#undef free_allocated
#undef c_reset
#undef c_step
#undef c_close
#undef c_render

#define LAGGED_MAX_LAG 64
#define LAGGED_BASE_OBS 118
#define LAGGED_QUEUE_OBS 126
#define LAGGED_OBS_SIZE (LAGGED_QUEUE_OBS + LAGGED_MAX_LAG)

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float action_switch_rate;
    float paddle_distance;
    float n;
} Log;

typedef struct LaggedBreakout {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;
    Breakout game;
    int action_lag;
    float volatility;
    int reward_interval;
    uint64_t noise_rng;
    float action_queue[LAGGED_MAX_LAG];
    int queue_head;
    float applied_action;
    float pending_reward;
    int reward_tick;
    int decisions;
    int switches;
    float last_command;
    float paddle_distance;
} LaggedBreakout;

static inline void lagged_require(int valid, const char* message) {
    if (!valid) {
        fprintf(stderr, "lagged_breakout: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

static inline float lagged_action_value(float action) {
    return action == LEFT ? -1.0f : action == RIGHT ? 1.0f : 0.0f;
}

// SplitMix64: per-environment noise stream, independent of the serve RNG.
static inline uint64_t lagged_random(LaggedBreakout* env) {
    uint64_t z = (env->noise_rng += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static inline void lagged_noise(LaggedBreakout* env, float* x, float* y) {
    *x = *y = 0.0f;
    if (env->volatility == 0.0f) return;
    double u = ((lagged_random(env) >> 11) + 0.5) * 0x1.0p-53;
    double v = ((lagged_random(env) >> 11) + 0.5) * 0x1.0p-53;
    double radius = env->volatility * sqrt(-2.0 * TICK_RATE * log(u));
    double angle = 2.0 * M_PI * v;
    *x = (float)(radius * cos(angle));
    *y = (float)(radius * sin(angle));
}

static inline void lagged_clear_episode(LaggedBreakout* env) {
    memset(env->action_queue, 0, sizeof(env->action_queue));
    env->queue_head = 0;
    env->applied_action = NOOP;
    env->pending_reward = 0.0f;
    env->reward_tick = 0;
    env->decisions = 0;
    env->switches = 0;
    env->last_command = NOOP;
    env->paddle_distance = 0.0f;
}

// The vector allocator can relocate the outer struct and attach buffers after
// init. Bind the inner game only once those final pointers are available.
static inline void lagged_bind_buffers(LaggedBreakout* env) {
    env->game.observations = env->observations;
    env->game.actions = env->actions;
    env->game.rewards = env->rewards;
    env->game.terminals = env->terminals;
}

static inline void lagged_observations(LaggedBreakout* env) {
    compute_observations(&env->game);
    float* obs = env->observations;
    obs[118] = env->action_lag / (float)LAGGED_MAX_LAG;
    obs[119] = env->volatility / 100.0f;
    obs[120] = env->game.frameskip / 64.0f;
    obs[121] = (env->game.hits % 4) / 4.0f;
    obs[122] = env->game.ball_speed / env->game.max_ball_speed;
    obs[123] = lagged_action_value(env->applied_action);
    obs[124] = env->reward_tick / (float)env->reward_interval;
    obs[125] = env->pending_reward / 864.0f;
    for (int i = 0; i < LAGGED_MAX_LAG; i++) {
        obs[LAGGED_QUEUE_OBS + i] = i < env->action_lag
            ? lagged_action_value(env->action_queue[(env->queue_head + i) % env->action_lag])
            : 0.0f;
    }
}

void init(LaggedBreakout* env) {
    lagged_require(env->action_lag >= 0 && env->action_lag <= LAGGED_MAX_LAG,
        "action_lag must be an integer in [0, 64]");
    lagged_require(isfinite(env->volatility) && env->volatility >= 0.0f,
        "volatility must be finite and nonnegative");
    lagged_require(env->game.frameskip > 0 && env->game.frameskip <= 64,
        "frameskip must be an integer in [1, 64]");
    lagged_require(env->reward_interval >= env->game.frameskip &&
        env->reward_interval % env->game.frameskip == 0,
        "reward_interval must be a positive multiple of frameskip");
    lagged_require(env->game.brick_rows == 6 && env->game.brick_cols == 18,
        "the observation layout requires 6 brick rows and 18 columns");
    lagged_require(!env->game.continuous, "only discrete NOOP/LEFT/RIGHT actions are supported");
    env->num_agents = env->game.num_agents = 1;
    env->game.rng = env->rng;
    env->noise_rng = (uint64_t)env->rng ^ UINT64_C(0xd1b54a32d192ed03);
    breakout_init(&env->game);
    lagged_clear_episode(env);
}

void allocate(LaggedBreakout* env) {
    init(env);
    env->observations = (float*)calloc(LAGGED_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    lagged_bind_buffers(env);
}

void c_reset(LaggedBreakout* env) {
    lagged_bind_buffers(env);
    lagged_clear_episode(env);
    env->rewards[0] = env->terminals[0] = 0.0f;
    breakout_reset(&env->game);
    lagged_observations(env);
}

static inline float lagged_deliver(LaggedBreakout* env, float command) {
    if (env->action_lag == 0) return command;
    float delivered = env->action_queue[env->queue_head];
    env->action_queue[env->queue_head] = command;
    env->queue_head = (env->queue_head + 1) % env->action_lag;
    return delivered;
}

void c_step(LaggedBreakout* env) {
    lagged_bind_buffers(env);
    env->terminals[0] = 0.0f;
    env->rewards[0] = 0.0f;
    float command = env->actions[0];
    if (command != LEFT && command != RIGHT) command = NOOP;
    if (env->decisions > 0 && command != env->last_command) env->switches++;
    env->last_command = command;
    env->decisions++;
    float paid_reward = 0.0f;
    for (int i = 0; i < env->game.frameskip; i++) {
        env->applied_action = lagged_deliver(env, command);
        float noise_x, noise_y;
        lagged_noise(env, &noise_x, &noise_y);
        float paddle_x = env->game.paddle_x;
        int balls = env->game.num_balls;
        env->game.tick++;
        step_frame_with_noise(&env->game, env->applied_action, noise_x, noise_y);
        // Exclude the automatic paddle teleport on a lost ball or episode reset.
        if (!env->terminals[0] && balls == env->game.num_balls) {
            env->paddle_distance += fabsf(env->game.paddle_x - paddle_x);
        }
        env->pending_reward += env->rewards[0];
        env->rewards[0] = 0.0f;
        env->reward_tick++;
        if (env->reward_tick == env->reward_interval || env->terminals[0]) {
            paid_reward += env->pending_reward;
            env->pending_reward = 0.0f;
            env->reward_tick = 0;
        }
        if (env->terminals[0]) {
            BreakoutLog* game_log = &env->game.log;
            env->log.perf += game_log->perf;
            env->log.score += game_log->score;
            env->log.episode_return += game_log->episode_return;
            env->log.episode_length += game_log->episode_length;
            env->log.action_switch_rate += env->switches / (float)(env->decisions > 1 ? env->decisions - 1 : 1);
            env->log.paddle_distance += env->paddle_distance;
            env->log.n += game_log->n;
            memset(game_log, 0, sizeof(*game_log));
            lagged_clear_episode(env);
            // Return the reset observation without applying an old command to
            // the next episode or mixing its rewards into this transition.
            break;
        }
    }
    env->rewards[0] = paid_reward;
    lagged_observations(env);
}

void c_render(LaggedBreakout* env) {
    if (env->game.client == NULL) {
        env->game.client = make_client(&env->game);
        SetWindowTitle(TextFormat("Lagged Breakout | lag=%d microsteps | volatility=%.2f",
            env->action_lag, env->volatility));
    }
    breakout_render(&env->game);
}

void c_close(LaggedBreakout* env) {
    if (env->game.client != NULL) {
        UnloadTexture(env->game.client->ball);
        close_client(env->game.client);
        env->game.client = NULL;
    }
    breakout_close(&env->game);
}

void free_allocated(LaggedBreakout* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

#endif
