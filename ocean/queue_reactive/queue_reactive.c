#include <stdio.h>
#include <string.h>
#include "queue_reactive.h"

int main(void) {
    QueueReactive env;
    float observations[QR_OBS_SIZE] = {0};
    float actions[4] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    memset(&env, 0, sizeof(env));
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.rng = 1;

    QRConfig config;
    qr_config_defaults(&config);
    qr_configure(&env, &config);
    c_reset(&env);
    for (int step = 0; step < 16; step++) {
        actions[0] = step % QR_NUM_ACTION_TYPES;
        actions[1] = step % QR_NUM_SIDES;
        actions[2] = step % QR_ACTION_PRICE_CHOICES;
        actions[3] = step % QR_SIZE_BUCKETS;
        c_step(&env);
        printf(
            "step=%d time=%lld inv=%d cash=%.0f fill=%d reject=%d qr=%d lost=%d reward=%.1f\n",
            step,
            (long long)env.time_ns,
            env.inventory,
            env.cash,
            env.last_action_filled,
            env.last_action_rejected,
            env.last_qr_type,
            env.last_action_lost_race,
            rewards[0]
        );
    }

    c_close(&env);
    return 0;
}
