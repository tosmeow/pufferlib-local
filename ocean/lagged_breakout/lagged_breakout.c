#include "lagged_breakout.h"

// Native interactive demo: ./lagged_breakout [lag_microsteps] [volatility]
int main(int argc, char** argv) {
    LaggedBreakout env = {
        .action_lag = argc > 1 ? atoi(argv[1]) : 4,
        .volatility = argc > 2 ? atof(argv[2]) : 6.0f,
        .reward_interval = 1,
        .game = {
            .frameskip = 1, .width = 576, .height = 330,
            .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
            .ball_width = 32, .ball_height = 32,
            .brick_width = 32, .brick_height = 12, .brick_rows = 6, .brick_cols = 18,
            .initial_ball_speed = 256, .max_ball_speed = 448, .paddle_speed = 620,
        },
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        env.actions[0] = IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A) ? LEFT
            : IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D) ? RIGHT : NOOP;
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
    return 0;
}
