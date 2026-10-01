/* Bounded correctness checks; no training or parameter search.
 * Compile and run through Slurm, e.g. using run_checks.sh in this directory.
 */
#include "../nash_impact.h"

#define CHECK(c) do { \
    if (!(c)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
        exit(1); \
    } \
} while (0)
#define NEAR(a, b, eps) CHECK(isfinite(a) && fabs((a) - (b)) <= (eps))

typedef struct {
    Env env;
    float observations[NASH_IMPACT_MAX_AGENTS][OBS_SIZE];
    float actions[NASH_IMPACT_MAX_AGENTS];
    float rewards[NASH_IMPACT_MAX_AGENTS];
    float terminals[NASH_IMPACT_MAX_AGENTS];
} Fixture;

static void init_population(Fixture* f, int dates, double scale, double quantity, int nd, int na) {
    memset(f, 0, sizeof(*f));
    Ini ini = {0};
    puf_ini_load_env(&ini, "nash_impact", 0, NULL);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "num_trades", dates);
    dict_set(kwargs, "num_directional", nd);
    dict_set(kwargs, "num_arbitrageurs", na);
    dict_set(kwargs, "reward_scale", scale);
    dict_set(kwargs, "initial_inventory", quantity);
    // These hand-computed regression cases use G=theta=1, independent of the run config.
    dict_set(kwargs, "impact", 1);
    dict_set(kwargs, "transaction_cost", 1);
    puf_init(&f->env, kwargs);
    puf_ini_free(&ini);
    for (int i = 0; i < f->env.num_agents; i++) {
        f->env.agents[i].observations = f->observations[i];
        f->env.agents[i].actions = &f->actions[i];
        f->env.agents[i].rewards = &f->rewards[i];
        f->env.agents[i].terminals = &f->terminals[i];
    }
    puf_reset(&f->env);
}

static void init_fixture(Fixture* f, int dates, double scale, double quantity) {
    init_population(f, dates, scale, quantity, 1, 1);
}

static void test_population_and_priority(void) {
    Fixture f;
    init_population(&f, 26, 1, 1, 2, 2);
    CHECK(f.env.num_agents == 4);
    for (int i=0;i<4;i++) {
        NEAR(f.observations[i][1], i<2 ? 1 : 0, 0);
        NEAR(f.observations[i][2], i<2 ? 1 : 0, 0);
    }
    // Explicitly average every uniform execution ordering, with mixed signs.
    double q[4]={.4,-.2,.3,-.5}, expected[4]={0},actual[NASH_IMPACT_MAX_AGENTS]={0};
    int counts=0,first[4]={0},pair[4][4]={{0}},ranks[4][4]={{0}};
    f.env.price=-.7;
    for(int a=0;a<4;a++)for(int b=0;b<4;b++)for(int c=0;c<4;c++)for(int d=0;d<4;d++) {
        if(a==b||a==c||a==d||b==c||b==d||c==d)continue;
        int order[4]={a,b,c,d};double preceding=0;
        counts++;first[a]++;pair[a][b]++;
        for(int k=0;k<4;k++) {
            int i=order[k];ranks[i][k]++;
            expected[i]+=(.5+1)*q[i]*q[i]+.7*q[i]+q[i]*preceding;
            preceding+=q[i];
        }
    }
    CHECK(counts==24);
    nash_impact_execute(&f.env,q,actual);
    for(int i=0;i<4;i++) {
        CHECK(first[i]==6);
        for(int k=0;k<4;k++)CHECK(ranks[i][k]==6);
        for(int j=0;j<4;j++)if(i!=j)CHECK(pair[i][j]==2); // 1/4 then 1/3.
        NEAR(actual[i],expected[i]/24,1e-14);
    }
    // Uniform zero-action liquidation: group costs remain separate, rewards individual.
    init_population(&f,26,1,1,2,2);
    double totals[4]={0};
    for(int k=0;k<25;k++) {
        puf_step(&f.env);
        for(int i=0;i<4;i++)totals[i]+=f.rewards[i];
        if(k<24)NEAR(f.env.price, -2+f.env.inventory[0]+f.env.inventory[1]+f.env.inventory[2]+f.env.inventory[3],1e-13);
    }
    for(int i=0;i<4;i++)CHECK(f.terminals[i]);
    NEAR(totals[0],-(1+1.0/26),1e-6);
    NEAR(totals[1],totals[0],0);
    NEAR(totals[2],0,0);NEAR(totals[3],0,0);
    NEAR(f.env.log.directional_cost,2+2.0/26,1e-6);
    NEAR(f.env.log.terminal_inventory_error,0,0);
    init_population(&f,2,1,1,2,2);
    f.actions[0]=3;f.actions[1]=-2;f.actions[2]=2;f.actions[3]=-4;
    puf_step(&f.env);
    for(int i=0;i<4;i++)CHECK(f.terminals[i]);
    NEAR(f.env.log.terminal_inventory_error,0,0);
    init_population(&f,3,1,1,3,0);
    puf_step(&f.env);puf_step(&f.env);
    NEAR(f.env.log.directional_cost,4.5+1,1e-6);
    NEAR(f.env.log.arbitrageur_cost,0,0);
}

static void test_uniform_liquidation(void) {
    Fixture f;
    init_fixture(&f, 26, 1, 1);
    CHECK(f.env.num_agents == 2);
    CHECK(f.env.agents[0].policy == 0 && f.env.agents[1].policy == 0);
    NEAR(f.observations[0][0], 1, 0);
    NEAR(f.observations[0][1], 1, 0);
    NEAR(f.observations[0][2], 1, 0);
    NEAR(f.observations[1][1], 0, 0);
    NEAR(f.observations[1][2], 0, 0);
    double total_reward = 0;
    for (int k = 0; k < 25; k++) {
        puf_step(&f.env);
        total_reward += f.rewards[0];
        NEAR(f.rewards[1], 0, 0);
        CHECK(f.terminals[0] == (k == 24));
        CHECK(f.terminals[1] == (k == 24));
        if (k < 24) {
            NEAR(f.env.last_trade[0], 1.0 / 26, 1e-14);
            NEAR(f.env.inventory[0], (25.0 - k) / 26, 1e-14);
            NEAR(f.env.price, -1 + f.env.inventory[0] + f.env.inventory[1], 1e-14);
        }
    }
    // Permanent own-impact cost G*Q^2/2 plus theta*Q^2/L.
    NEAR(total_reward, -(0.5 + 1.0 / 26), 1e-7);
    NEAR(f.env.log.directional_cost, 0.5 + 1.0 / 26, 1e-7);
    NEAR(f.env.log.arbitrageur_cost, 0, 0);
    NEAR(f.env.log.terminal_inventory_error, 0, 0);
    NEAR(f.env.log.episode_length, 25, 0);
    NEAR(f.env.log.n, 1, 0);
    CHECK(f.env.boundary_reached == 1);
    CHECK(f.env.tick == 0); // Auto-reset preserves terminal reward/signals.
    NEAR(f.observations[0][1], 1, 0);
    puf_step(&f.env);
    CHECK(f.env.boundary_reached == 0);
    CHECK(f.terminals[0] == 0 && f.terminals[1] == 0);
    NEAR(f.env.log.n, 1, 0);
    puf_close(&f.env);
}

static void test_round_trip_and_final_cost(void) {
    Fixture f;
    init_fixture(&f, 2, 1, 1);
    f.actions[0] = 3;   // Sell 2, then buy 1 to clear.
    f.actions[1] = -2;  // Buy 1, then sell 1 to clear.
    puf_step(&f.env);
    // Direct two-date calculation, including the second date at price -1:
    // directional costs [5, 0], arbitrageur costs [0.5, 2].
    NEAR(f.rewards[0], -5, 0);
    NEAR(f.rewards[1], -2.5, 0);
    NEAR(f.env.log.directional_cost, 5, 0);
    NEAR(f.env.log.arbitrageur_cost, 2.5, 0);
    NEAR(f.env.log.terminal_inventory_error, 0, 0);
    CHECK(f.terminals[0] == 1 && f.terminals[1] == 1);
    CHECK(f.actions[0] == 3 && f.actions[1] == -2); // Keep PPO's latent action.

    Fixture scaled;
    init_fixture(&scaled, 2, 0.25, 1);
    scaled.actions[0] = 3;
    scaled.actions[1] = -2;
    puf_step(&scaled.env);
    NEAR(scaled.rewards[0], -1.25, 0); // Environment must not clip.
    NEAR(scaled.rewards[1], -0.625, 0);
    NEAR(scaled.env.log.directional_cost, 5, 0); // Log raw costs.
    NEAR(scaled.env.log.directional_return, -1.25, 0);
}

static void test_two_date_reference(void) {
    Fixture f;
    init_fixture(&f, 2, 1, 1);
    // G=theta=1 reference: D trades [1/2,1/2], A trades [1/8,-1/8].
    f.actions[0] = 0;
    f.actions[1] = 0.25f;
    puf_step(&f.env);
    NEAR(f.env.log.directional_cost, 1.0625, 0);
    NEAR(f.env.log.arbitrageur_cost, -0.03125, 0);
    NEAR(f.rewards[1], 0.03125, 0); // Arbitrageur profit is its own reward.
    NEAR(f.env.log.score, -1.03125, 0);
    Dict out = {0};
    puf_log(&f.env.log, &out);
    NEAR(dict_get(&out, "directional_cost"), 1.0625, 0);
    NEAR(dict_get(&out, "arbitrageur_cost"), -0.03125, 0);
    dict_clear(&out);

    Fixture larger;
    init_fixture(&larger, 2, 1, 3);
    larger.actions[1] = 0.25f;
    puf_step(&larger.env);
    NEAR(larger.env.log.directional_cost, 9 * 1.0625, 0);
    NEAR(larger.env.log.arbitrageur_cost, -9 * 0.03125, 0);
}

static void test_opponent_deviations_do_not_change_own_information(void) {
    // Counterfactual rollouts: changing either opponent's trades must leave
    // this agent's entire observation history unchanged for fixed own actions.
    for (int own = 0; own < 2; own++) {
        Fixture baseline, deviating;
        init_fixture(&baseline, 26, 1, 1);
        init_fixture(&deviating, 26, 1, 1);
        int rewards_differ = 0;
        for (int k = 0; k < 25; k++) {
            for (int j = 0; j < OBS_SIZE; j++)
                NEAR(baseline.observations[own][j], deviating.observations[own][j], 0);
            NEAR(baseline.observations[own][2], own == 0 ? 1 : 0, 0);
            baseline.actions[own] = deviating.actions[own] = 0.5f + 0.1f*k;
            baseline.actions[1-own] = 0;
            deviating.actions[1-own] = k % 2 ? -3 : 4;
            puf_step(&baseline.env);
            puf_step(&deviating.env);
            NEAR(baseline.env.inventory[own], deviating.env.inventory[own], 0);
            CHECK(baseline.terminals[own] == deviating.terminals[own]);
            rewards_differ |= baseline.rewards[own] != deviating.rewards[own];
        }
        CHECK(rewards_differ); // Payoffs still interact even though inputs do not.
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        Fixture f;
        if (strcmp(argv[1], "invalid_dates") == 0) {
            init_fixture(&f, 1, 1, 1);
        } else if (strcmp(argv[1], "invalid_action") == 0) {
            init_fixture(&f, 26, 1, 1);
            f.actions[0] = NAN;
            puf_step(&f.env);
        } else if (strcmp(argv[1], "invalid_population") == 0) {
            init_population(&f,26,1,1,65,0);
        } else {
            return 2;
        }
        return 0; // Negative cases must exit from validation before this.
    }
    test_population_and_priority();
    test_uniform_liquidation();
    test_round_trip_and_final_cost();
    test_two_date_reference();
    test_opponent_deviations_do_not_change_own_information();
    puts("nash_impact: all checks passed");
    return 0;
}
