#include <stdlib.h>
#include <math.h>
typedef float obs_t;
#include "pufferenv.h"

#define AGENTS 2
#define ACT_SIZES {3}
#define OBS_SIZE (3 * AGENTS)
#define NUM_ATNS 1
#define MAX_STEPS 1000

typedef Env RPS;


// Required struct. Floats only, n last
struct Log { float perf, score, n; };

typedef struct {int previous_move; int score;} Player;

struct Env {
    Log log; int num_agents; unsigned int rng; // Required
    Agent agents[AGENTS]; int tag, boundary_reached; // Required
    Player players[AGENTS];
    int steps;
}; // Required: An env struct

void puf_init(Env* env, Dict* kwargs){
    env->num_agents = AGENTS;
    env->steps = 0;
    for (int i=0; i<AGENTS; i++) {
        env->agents[i].policy = i;
        env->agents[i].action_mask = NULL;
    }
}

void puf_log(Log* log, Dict* out){
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}

void compute_observations(Env* env) {
    for (int i = 0; i < 2; i++) {
        obs_t* obs = env->agents[i].observations;

        for (int j = 0; j < 6; j++) {
            obs[j] = 0.0f;
        }

        int mine = env->players[i].previous_move;
        int theirs = env->players[1 - i].previous_move;

        // -1 means no previous move, at the start of a match.
        if (mine >= 0 && mine < 3) {
            obs[mine] = 1.0f;
        }
        if (theirs >= 0 && theirs < 3) {
            obs[3 + theirs] = 1.0f;
        }
    }
}


void puf_reset(Env* env){
    env->steps = 0;
    for (int i = 0; i<AGENTS; i++){
        env->players[i].previous_move = -1;
        env->players[i].score = 0;
    }
    compute_observations(env);
}

void puf_step(Env* env) {
    float r = 0;
    int a = (int)*env->agents[0].actions;
    int b = (int)*env->agents[1].actions;
    if (a == b) {r = 0;}
    else if ((a == 0 && b == 2) ||
            (a == 1 && b == 0) ||
            (a == 2 && b == 1)) {
                r = 1;
            }
    else {
        r = -1;
    }
    
    //Player update.
    env->players[0].previous_move = a;
    env->players[0].score += r;
    env->players[1].previous_move = b;
    env->players[1].score -= r;

    //Rewards for learning from this round.
    env->agents[0].rewards[0] = r;
    env->agents[1].rewards[0] = -r;
    env->steps++;

    if (env->steps >= MAX_STEPS){
        int score = env->players[0].score;

        env->log.score += score;
        env->log.perf += score > 0 ? 1.0f : (score == 0 ? 0.5f : 0.0f);
        env->log.n += 1.0f;
        
        for (int i = 0; i<AGENTS; i++){
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