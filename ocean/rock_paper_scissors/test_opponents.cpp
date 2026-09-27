#include <cassert>
#include <cmath>
#include <initializer_list>
#include "rock_paper_scissors.h"

int main() {
    for (int bot = 0; bot < BOT_COUNT; bot++) {
        for (int last = -1; last < 3; last++) {
            int counts[3] = {};
            for (int i = 0; i < 3000; i++) {
                int action = opponent_action(&opponents[bot], last, (i + 0.5) / 3000);
                assert(action >= 0 && action < 3);
                counts[action]++;
            }
            for (int action = 0; action < 3; action++) {
                double expected = last == -1 ? 1.0 / 3 : opponents[bot].f[last][action];
                assert(std::abs(counts[action] - 3000 * expected) < 1);
            }
            assert(opponent_action(&opponents[bot], last, 0) >= 0);
            assert(opponent_action(&opponents[bot], last, std::nextafter(1.0, 0.0)) < 3);
        }
    }

    Dict kwargs = {};
    dict_set(&kwargs, "round_observation", 1);
    dict_set(&kwargs, "selfplay_share", 1.0 / 3);
    Env env = {};
    float obs[AGENTS][OBS_SIZE] = {}, actions[AGENTS] = {};
    float rewards[AGENTS] = {}, terminals[AGENTS] = {};
    for (int i = 0; i < AGENTS; i++) {
        env.agents[i].observations = obs[i];
        env.agents[i].actions = &actions[i];
        env.agents[i].rewards = &rewards[i];
        env.agents[i].terminals = &terminals[i];
    }
    auto reset = [&](int mode) {
        dict_set(&kwargs, "bot_policy", mode);
        puf_init(&env, &kwargs);
        assert(env.num_agents == 2);
        assert(env.agents[0].policy == 0 && env.agents[1].policy == 0);
        puf_reset(&env);
        for (int i = 0; i < AGENTS; i++) {
            assert(env.players[i].previous_move == -1);
            assert(env.players[i].opponent_move == -1);
            assert(env.players[i].score == 0);
            for (float value : obs[i]) assert(value == 0);
        }
    };
    auto check_obs = [&](int slot, int mine, int theirs) {
        for (int j = 0; j < 6; j++)
            assert(obs[slot][j] == (j == mine || j == 3 + theirs));
        assert(obs[slot][6] == (float)env.steps / (MAX_STEPS - 1));
    };

    // Every self-play action pair: simultaneous moves and mirrored perspectives.
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            reset(BOT_SELFPLAY);
            actions[0] = a;
            actions[1] = b;
            puf_step(&env);
            int expected = a == b ? 0 : ((a - b + 3) % 3 == 1 ? 1 : -1);
            assert(rewards[0] == expected && rewards[1] == -expected);
            assert(env.players[0].score == expected && env.players[1].score == -expected);
            check_obs(0, a, b);
            check_obs(1, b, a);
        }
    }

    // Each counter uses its own learner's previous move, not the other slot's.
    reset(BOT_COUNTER);
    env.players[0].previous_move = 0;
    env.players[1].previous_move = 1;
    actions[0] = 2;
    actions[1] = 0;
    puf_step(&env);
    assert(rewards[0] == 1 && rewards[1] == 1);
    check_obs(0, 2, 1);
    check_obs(1, 0, 2);
    puf_step(&env);
    assert(rewards[0] == -1 && rewards[1] == -1);

    reset(BOT_ALWAYS_ROCK);
    env.players[0].previous_move = env.players[1].previous_move = 0;
    actions[0] = 1;
    actions[1] = 2;
    puf_step(&env);
    assert(rewards[0] == 1 && rewards[1] == -1);
    check_obs(0, 1, 0);
    check_obs(1, 2, 0);

    // Two independent uniform bots use two draws from the seeded env RNG.
    reset(BOT_UNIFORM);
    unsigned int expected_rng = env.rng;
    int expected_bot[AGENTS];
    for (int i = 0; i < AGENTS; i++)
        expected_bot[i] = (int)(3 * (rand_r(&expected_rng) / ((double)RAND_MAX + 1.0)));
    puf_step(&env);
    assert(env.rng == expected_rng);
    for (int i = 0; i < AGENTS; i++) assert(env.players[i].opponent_move == expected_bot[i]);

    // Random modes keep their selected type for a full match; both slots reset.
    for (int mode : {BOT_RANDOM, BOT_MIXED}) {
        reset(mode);
        unsigned int seen = 0;
        for (unsigned int seed = 0; seed < 100; seed++) {
            env.rng = seed;
            puf_reset(&env);
            int selected = env.match_type;
            seen |= 1u << selected;
            env.rng = seed;
            puf_reset(&env);
            assert(env.match_type == selected);
            env.log = {};
            terminals[0] = terminals[1] = 0;
            for (int step = 0; step < MAX_STEPS - 1; step++) {
                puf_step(&env);
                assert(env.match_type == selected);
                for (int i = 0; i < AGENTS; i++)
                    assert(obs[i][6] == (float)(step + 1) / (MAX_STEPS - 1));
                assert(terminals[0] == 0 && terminals[1] == 0);
            }
            puf_step(&env);
            assert(env.steps == 0 && env.bot_policy == mode);
            assert(env.log.n == 2 && env.boundary_reached == 1);
            assert(terminals[0] == 1 && terminals[1] == 1);
            for (int i = 0; i < AGENTS; i++) {
                assert(env.players[i].previous_move == -1 && env.players[i].opponent_move == -1);
                assert(env.players[i].score == 0);
                for (float value : obs[i]) assert(value == 0);
            }
        }
        unsigned int expected = mode == BOT_RANDOM ? (1u << BOT_COUNT) - 1
            : (1u << BOT_ALWAYS_ROCK) | (1u << BOT_COUNTER) | (1u << BOT_SELFPLAY);
        assert(seen == expected);
    }
    reset(BOT_MIXED);
    int counts[BOT_MIXED + 1] = {};
    for (int i = 0; i < 6000; i++) {
        puf_reset(&env);
        counts[env.match_type]++;
    }
    assert(counts[BOT_UNIFORM] == 0);
    const int mixed_bots[] = {BOT_ALWAYS_ROCK, BOT_COUNTER, BOT_SELFPLAY};
    for (int bot : mixed_bots)
        assert(std::abs(counts[bot] - 2000) < 180);
    for (double share : {0.0, 0.8, 0.95, 1.0}) {
        dict_set(&kwargs, "selfplay_share", share);
        reset(BOT_MIXED);
        int weighted[BOT_MIXED + 1] = {};
        for (int i = 0; i < 12000; i++) {
            puf_reset(&env);
            weighted[env.match_type]++;
        }
        assert(std::abs(weighted[BOT_SELFPLAY] - 12000 * share) < 250);
        for (int bot : {BOT_ALWAYS_ROCK, BOT_COUNTER})
            assert(std::abs(weighted[bot] - 6000 * (1 - share)) < 250);
        if (share == 0) assert(weighted[BOT_SELFPLAY] == 0);
        if (share == 1) assert(weighted[BOT_SELFPLAY] == 12000);
    }
    dict_set(&kwargs, "round_observation", 0);
    reset(BOT_SELFPLAY);
    for (int step = 0; step < MAX_STEPS + 1; step++) {
        puf_step(&env);
        for (int i = 0; i < AGENTS; i++) assert(obs[i][6] == 0);
    }
    dict_clear(&kwargs);
    puts("RPS scripted and self-play tests passed");
}
