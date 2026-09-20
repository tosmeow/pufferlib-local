/* Standalone native smoke loop.
 *
 * This file mirrors the PufferLib Ocean demo convention, but GFR is C++.
 * Build with a C++ compiler and the same engine objects/libs used by
 * GFR/gfootball/build_game_engine.sh.
 */

#include "gfr.h"

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    GFR env = {};
    env.num_agents = 1;
    env.scenario = GFR_ACADEMY_EMPTY_GOAL_CLOSE;
    env.rng = 1;
    env.observations = (float*)calloc(GFR_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    for (int i = 0; i < 1000; i++) {
        env.actions[0] = (float)(rand_r(&env.rng) % GFR_DEFAULT_ACTIONS);
        c_step(&env);
        if ((i % 100) == 0) {
            c_render(&env);
        }
    }

    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
