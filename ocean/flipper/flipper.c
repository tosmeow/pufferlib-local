/* Pure C demo for Flipper.
 * Build with:
 *   bash build.sh flipper --local
 */

#include "flipper.h"

/* demo builds a tiny standalone executable for interactive physics debugging.
 * It allocates the same buffers that vecenv would allocate in training, then
 * drives the two flippers from the keyboard:
 *   A/Left Arrow  -> left speed=1, left hold=true
 *   D/Right Arrow -> right speed=1, right hold=true
 */
void demo(void) {
    Flipper env = {
        .num_agents = 1,
        .max_ticks = FLIPPER_DEFAULT_MAX_TICKS,
        .rng = 1,
    };
    env.observations = (unsigned char*)calloc(FLIPPER_OBS_SIZE, sizeof(unsigned char));
    env.actions = (float*)calloc(4, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        env.actions[ACTION_LEFT_SPEED] = IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT) ? 1.0f : 0.0f;
        env.actions[ACTION_RIGHT_SPEED] = IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT) ? 1.0f : 0.0f;
        env.actions[ACTION_LEFT_HOLD] = env.actions[ACTION_LEFT_SPEED] > 0.0f ? 1.0f : -1.0f;
        env.actions[ACTION_RIGHT_HOLD] = env.actions[ACTION_RIGHT_SPEED] > 0.0f ? 1.0f : -1.0f;

        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

/* main only delegates to demo so the standalone build path stays obvious.
 * Training never calls this file; it compiles binding.c instead.
 */
int main(void) {
    demo();
    return 0;
}
