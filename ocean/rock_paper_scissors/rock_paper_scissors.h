#include <stdlib.h>
#include <math.h>
typedef float obs_t;
#include "pufferenv.h"

#define AGENTS 2
#define ACT_SIZES {3}
#ifndef RPS_OBS_SIZE
#define RPS_OBS_SIZE 7
#endif
#define OBS_SIZE RPS_OBS_SIZE
#if OBS_SIZE != 6 && OBS_SIZE != 7
#error RPS observations must have 6 legacy inputs or 7 inputs with a round feature
#endif
#define NUM_ATNS 1
#define MAX_STEPS 1000

typedef Env RPS;

// Rows: opponent's previous move. Columns: our rock, paper, scissors.
// Each row must contain nonnegative probabilities summing to one.
typedef struct { float f[3][3]; } Opponent;

enum { BOT_UNIFORM, BOT_COUNTER, BOT_ALWAYS_ROCK, BOT_COUNT };
enum { BOT_RANDOM = BOT_COUNT, BOT_SELFPLAY, BOT_MIXED };

static const Opponent opponents[BOT_COUNT] = {
    [BOT_UNIFORM] = {.f = {
        {1.f/3, 1.f/3, 1.f/3},
        {1.f/3, 1.f/3, 1.f/3},
        {1.f/3, 1.f/3, 1.f/3},
    }},
    [BOT_COUNTER] = {.f = {
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 0},
    }},
    [BOT_ALWAYS_ROCK] = {.f = {
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
    }},
};

// u is uniform in [0, 1). All presets open uniformly when last_move is -1.
static int opponent_action(const Opponent* opponent, int last_move, double u) {
    if (last_move == -1) return (int)(3 * u);
    const float* p = opponent->f[last_move];
    if (u < p[0]) return 0;
    if (u < (double)p[0] + p[1]) return 1;
    return 2;
}


// Required struct. Floats only, n last
struct Log { float perf, score, n; };

typedef struct {int previous_move, opponent_move, score;} Player;

struct Env {
    Log log; int num_agents; unsigned int rng; // Required
    Agent agents[AGENTS]; int tag, boundary_reached; // Required
    Player players[2];
    int bot_policy, match_type, round_observation;
    double selfplay_share;
    const Opponent* opponent;
    int steps;
}; // Required: An env struct

void puf_init(Env* env, Dict* kwargs){
    double bot = dict_get(kwargs, "bot_policy");
    if (!(bot >= 0 && bot <= BOT_MIXED) || bot != (int)bot) {
        fprintf(stderr, "RPS bot_policy must be between 0 and 5\n");
        exit(EXIT_FAILURE);
    }
    env->bot_policy = (int)bot;
    double clock = dict_get(kwargs, "round_observation");
    if (!(clock == 0 || clock == 1)) {
        fprintf(stderr, "RPS round_observation must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }
    env->round_observation = (int)clock;
    env->selfplay_share = dict_get(kwargs, "selfplay_share");
    if (!(env->selfplay_share >= 0 && env->selfplay_share <= 1)) {
        fprintf(stderr, "RPS selfplay_share must be between 0 and 1\n");
        exit(EXIT_FAILURE);
    }
    env->num_agents = AGENTS;
    env->steps = 0;
    for (int i=0; i<AGENTS; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

void puf_log(Log* log, Dict* out){
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}

void compute_observations(Env* env) {
    for (int i = 0; i < AGENTS; i++) {
        obs_t* obs = env->agents[i].observations;

        for (int j = 0; j < OBS_SIZE; j++) {
            obs[j] = 0.0f;
        }

        int mine = env->players[i].previous_move;
        int theirs = env->players[i].opponent_move;

        // -1 means no previous move, at the start of a match.
        if (mine >= 0 && mine < 3) {
            obs[mine] = 1.0f;
        }
        if (theirs >= 0 && theirs < 3) {
            obs[3 + theirs] = 1.0f;
        }
#if OBS_SIZE == 7
        // Completed rounds: 0 at opening, 1 before the final action.
        obs[6] = env->round_observation ? (float)env->steps / (MAX_STEPS - 1) : 0;
#endif
    }
}


void puf_reset(Env* env){
    int bot = env->bot_policy;
    if (bot == BOT_RANDOM || bot == BOT_MIXED) {
        double u = rand_r(&env->rng) / ((double)RAND_MAX + 1.0);
        if (bot == BOT_RANDOM) {
            bot = (int)(u * BOT_COUNT);
        } else {
            double scripted_share = (1.0 - env->selfplay_share) / 2;
            bot = u < scripted_share ? BOT_ALWAYS_ROCK
                : u < 2 * scripted_share ? BOT_COUNTER : BOT_SELFPLAY;
        }
    }
    env->match_type = bot;
    env->opponent = bot == BOT_SELFPLAY ? NULL : &opponents[bot];
    env->steps = 0;
    for (int i = 0; i < 2; i++){
        env->players[i].previous_move = -1;
        env->players[i].opponent_move = -1;
        env->players[i].score = 0;
    }
    compute_observations(env);
}

void puf_step(Env* env) {
    // Both sampled policy actions are meaningful: paired in self-play,
    // or each playing an independent scripted opponent.
    int actions[AGENTS] = {
        (int)*env->agents[0].actions,
        (int)*env->agents[1].actions,
    };
    for (int i = 0; i < AGENTS; i++) {
        int a = actions[i];
        int b;
        if (env->match_type == BOT_SELFPLAY) {
            b = actions[1 - i];
        } else {
            double u = rand_r(&env->rng) / ((double)RAND_MAX + 1.0);
            b = opponent_action(env->opponent, env->players[i].previous_move, u);
        }
        int r = a == b ? 0 : ((a - b + 3) % 3 == 1 ? 1 : -1);
        env->players[i].previous_move = a;
        env->players[i].opponent_move = b;
        env->players[i].score += r;
        env->agents[i].rewards[0] = r;
    }
    env->steps++;

    if (env->steps >= MAX_STEPS){
        for (int i = 0; i<AGENTS; i++){
            int score = env->players[i].score;
            env->log.score += score;
            env->log.perf += score > 0 ? 1.0f : (score == 0 ? 0.5f : 0.0f);
            env->log.n += 1.0f;
            env->agents[i].terminals[0] = 1.0f;
        }

        env->boundary_reached = 1;
        puf_reset(env);
        return;
    }

    //Prepare observations for next round.
    compute_observations(env);
}

void puf_render(Env* env){
    (void)env;
}

void puf_close(Env* env){
    (void)env;
}
