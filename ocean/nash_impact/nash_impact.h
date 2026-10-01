#pragma once

/* Headless permanent-impact game with configurable trader populations.
 * Reference: resources/nash_impact/2205.00494v2.pdf, Section 2, equations 1-2.
 *
 * The first num_directional agents start with Q > 0 each; the remaining
 * num_arbitrageurs start at zero. Every agent must finish flat. Positive trade quantities sell, negative ones buy.
 * Observations: [remaining dates / L, own inventory / Q, own initial inventory / Q].
 * The final feature is a fixed role (directional=1, arbitrageur=0). Neither
 * impacted price, opponent inventory nor previous reward enters the policy.
 * Continuous action z: q = inventory / remaining_dates + (Q / L) * z.
 * All agents choose simultaneously. The last date clears all inventories
 * automatically, and its cost is included in the preceding transition.
 * Thus L market dates produce L-1 policy decisions. Zero actions give uniform
 * directional liquidation and an inactive arbitrageur.
 *
 * No price noise, risk aversion, trade caps, inventory caps, or opponent pool.
 * Priority averages a uniformly random permutation: each remaining trader
 * has probability 1 / remaining_count of executing next, while each pair
 * has probability 1/2 of either ordering. Rewards are each agent's negative
 * cost times a fixed positive reward_scale; accounting uses doubles.
 * IMPORTANT: src/pufferl.cu currently clips training rewards to [-1, 1].
 * This environment does not clip them. Disable trainer reward clipping before
 * claiming a faithful training reproduction; scaling alone is no guarantee.
 * Shared-policy feedback learning is not itself a Nash-equilibrium check.
 */

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef float obs_t;
#include "pufferenv.h"

#define OBS_SIZE 3
#define NUM_ATNS 1
#define ACT_SIZES {1}
#define NASH_IMPACT_MAX_AGENTS 64

// The vectorizer averages these float fields by n (completed joint episodes).
struct Log {
    float perf;
    float score;
    float directional_cost;
    float arbitrageur_cost;
    float directional_return;
    float arbitrageur_return;
    float episode_length;
    float terminal_inventory_error;
    float n;
};

struct Env {
    Log log;
    Agent agents[NASH_IMPACT_MAX_AGENTS];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;

    int num_directional;
    int num_arbitrageurs;
    int num_trades;
    int tick;                  // Number of market dates already executed.
    double initial_inventory;  // Also the common action/observation unit Q.
    double impact;
    double transaction_cost;
    double reward_scale;       // Multiplies negative execution cost.
    double inventory[NASH_IMPACT_MAX_AGENTS];
    double cost[NASH_IMPACT_MAX_AGENTS];
    double last_trade[NASH_IMPACT_MAX_AGENTS];
    double price;              // Displacement from the fixed unaffected price.
};

static void nash_impact_require(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "nash_impact: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

static float nash_impact_float(double value) {
    nash_impact_require(isfinite(value) && fabs(value) <= FLT_MAX,
        "non-finite or overflowing reward/observation/log value");
    return (float)value;
}

static void nash_impact_observe(Env* env) {
    for (int i = 0; i < env->num_agents; i++) {
        obs_t* obs = env->agents[i].observations;
        obs[0] = (float)(env->num_trades - env->tick) / env->num_trades;
        obs[1] = nash_impact_float(env->inventory[i] / env->initial_inventory);
        obs[2] = i < env->num_directional ? 1.0f : 0.0f;
    }
}

void puf_init(Env* env, Dict* kwargs) {
    double directional = dict_find(kwargs, "num_directional") ? dict_get(kwargs, "num_directional") : 1;
    double arbitrageurs = dict_find(kwargs, "num_arbitrageurs") ? dict_get(kwargs, "num_arbitrageurs") : 1;
    nash_impact_require(isfinite(directional) && directional >= 1 && floor(directional) == directional,
        "num_directional must be a positive integer");
    nash_impact_require(isfinite(arbitrageurs) && arbitrageurs >= 0 && floor(arbitrageurs) == arbitrageurs,
        "num_arbitrageurs must be a nonnegative integer");
    nash_impact_require(directional + arbitrageurs <= NASH_IMPACT_MAX_AGENTS,
        "total population exceeds NASH_IMPACT_MAX_AGENTS (64)");
    env->num_directional = (int)directional;
    env->num_arbitrageurs = (int)arbitrageurs;
    env->num_agents = (int)(directional + arbitrageurs);
    // Population counts are authoritative. Legacy 1+1 configurations still work.
    dict_set(kwargs, "num_agents", env->num_agents);
    double dates = dict_get(kwargs, "num_trades");
    nash_impact_require(isfinite(dates) && dates >= 2 && dates <= INT_MAX
        && floor(dates) == dates, "num_trades must be an integer >= 2");
    env->num_trades = (int)dates;
    env->initial_inventory = dict_get(kwargs, "initial_inventory");
    env->impact = dict_get(kwargs, "impact");
    env->transaction_cost = dict_get(kwargs, "transaction_cost");
    env->reward_scale = dict_get(kwargs, "reward_scale");
    nash_impact_require(isfinite(env->initial_inventory) && env->initial_inventory > 0,
        "initial_inventory must be finite and positive");
    nash_impact_require(isfinite(env->impact) && env->impact > 0,
        "impact must be finite and positive");
    nash_impact_require(isfinite(env->transaction_cost) && env->transaction_cost > 0,
        "transaction_cost must be finite and positive");
    nash_impact_require(isfinite(env->reward_scale) && env->reward_scale > 0,
        "reward_scale must be finite and positive");
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

void puf_reset(Env* env) {
    env->tick = 0;
    env->price = 0;
    for (int i = 0; i < env->num_agents; i++) {
        env->inventory[i] = i < env->num_directional ? env->initial_inventory : 0;
        env->cost[i] = 0;
        env->last_trade[i] = 0;
    }
    env->boundary_reached = 0;
    // Preserve outgoing rewards/terminals and accumulated logs on auto-reset.
    nash_impact_observe(env);
}

static void nash_impact_execute(Env* env, const double* trade, double* step_cost) {
    // Uniform permutation priority: any other trader precedes i with P=1/2.
    // Cost_i = theta*q_i^2 + G*q_i^2/2 - price*q_i
    //          + G*q_i*sum(preceding orders), averaged over permutations.
    double total = 0;
    for (int i = 0; i < env->num_agents; i++) total += trade[i];
    for (int i = 0; i < env->num_agents; i++) {
        double q = trade[i];
        double cost = (0.5 * env->impact + env->transaction_cost) * q * q
            - env->price * q + 0.5 * env->impact * q * (total - q);
        nash_impact_require(isfinite(cost), "non-finite execution cost");
        step_cost[i] += cost;
        env->cost[i] += cost;
        env->inventory[i] -= q;
        env->last_trade[i] = q;
    }
    env->price -= env->impact * total;
    nash_impact_require(isfinite(env->price), "non-finite price");
    env->tick++;
}

static void nash_impact_log_episode(Env* env) {
    // Group costs are display diagnostics; each agent keeps its own reward.
    double directional_cost = 0, arbitrageur_cost = 0, terminal_error = 0;
    for (int i = 0; i < env->num_agents; i++) {
        if (i < env->num_directional) directional_cost += env->cost[i];
        else arbitrageur_cost += env->cost[i];
        terminal_error += fabs(env->inventory[i]);
    }
    float score = nash_impact_float(-directional_cost - arbitrageur_cost);
    env->log.score += score;
    env->log.perf += score;
    env->log.directional_cost += nash_impact_float(directional_cost);
    env->log.arbitrageur_cost += nash_impact_float(arbitrageur_cost);
    env->log.directional_return += nash_impact_float(-env->reward_scale * directional_cost);
    env->log.arbitrageur_return += nash_impact_float(-env->reward_scale * arbitrageur_cost);
    env->log.episode_length += env->num_trades - 1;
    env->log.terminal_inventory_error += nash_impact_float(terminal_error);
    env->log.n += 1;
}

void puf_step(Env* env) {
    int remaining = env->num_trades - env->tick;
    nash_impact_require(remaining >= 2, "step called outside a live episode");
    env->boundary_reached = 0;
    double trade[NASH_IMPACT_MAX_AGENTS];
    double step_cost[NASH_IMPACT_MAX_AGENTS] = {0};
    for (int i = 0; i < env->num_agents; i++) {
        double z = env->agents[i].actions[0];
        nash_impact_require(isfinite(z), "action must be finite");
        trade[i] = env->inventory[i] / remaining
            + (env->initial_inventory / env->num_trades) * z;
        env->agents[i].terminals[0] = 0;
    }
    nash_impact_execute(env, trade, step_cost);

    int terminal = env->tick == env->num_trades - 1;
    if (terminal) {
        double clearing[NASH_IMPACT_MAX_AGENTS];
        for (int i = 0; i < env->num_agents; i++) clearing[i] = env->inventory[i];
        nash_impact_execute(env, clearing, step_cost);
        nash_impact_log_episode(env);
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].rewards[0] = nash_impact_float(-env->reward_scale * step_cost[i]);
        env->agents[i].terminals[0] = (float)terminal;
    }
    if (terminal) {
        puf_reset(env);
        env->boundary_reached = 1;
    } else {
        nash_impact_observe(env);
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "directional_cost", log->directional_cost);
    dict_set(out, "arbitrageur_cost", log->arbitrageur_cost);
    dict_set(out, "directional_return", log->directional_return);
    dict_set(out, "arbitrageur_return", log->arbitrageur_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "terminal_inventory_error", log->terminal_inventory_error);
    dict_set(out, "n", log->n);
}

// Rendering and history plots can be added independently of the game rules.
void puf_render(Env* env) { (void)env; }
void puf_close(Env* env) { (void)env; }
