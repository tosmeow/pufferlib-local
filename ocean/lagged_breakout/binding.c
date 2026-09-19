#include "lagged_breakout.h"
#define OBS_SIZE LAGGED_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {3}
#define OBS_TENSOR_T FloatTensor
#define Env LaggedBreakout
#include "vecenv.h"

static double option(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : item->value;
}

static int integer_option(Dict* kwargs, const char* key, int fallback, int lo, int hi) {
    double value = option(kwargs, key, fallback);
    if (!isfinite(value) || value < lo || value > hi || value != floor(value)) {
        fprintf(stderr, "lagged_breakout: %s must be an integer in [%d, %d]\n", key, lo, hi);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

void my_init(Env* env, Dict* kwargs) {
    env->action_lag = integer_option(kwargs, "action_lag", 0, 0, LAGGED_MAX_LAG);
    env->volatility = option(kwargs, "volatility", 0.0);
    int frameskip = integer_option(kwargs, "frameskip", 4, 1, 64);
    env->reward_interval = integer_option(kwargs, "reward_interval", frameskip, 1, INT_MAX);
    env->rng += (unsigned int)integer_option(kwargs, "seed", 0, 0, INT_MAX);
    integer_option(kwargs, "num_agents", 1, 1, 1);
    // Geometry and speeds match the original Breakout defaults. Keeping these
    // fixed makes the game and observation layout comparable across all runs.
    env->game = (Breakout){
        .frameskip = frameskip,
        .width = 576, .height = 330,
        .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
        .ball_width = 32, .ball_height = 32,
        .brick_width = 32, .brick_height = 12,
        .brick_rows = 6, .brick_cols = 18,
        .initial_ball_speed = 256, .max_ball_speed = 448, .paddle_speed = 620,
    };
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "action_switch_rate", log->action_switch_rate);
    dict_set(out, "paddle_distance", log->paddle_distance);
}
