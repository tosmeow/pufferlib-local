// Evaluate the default 6 -> 16 -> 4xMinGRU -> (3 logits + value) RPS policy.
// Reuse the repository's CPU inference implementation; no training or argmax.
#include <stdint.h>
#include "../src/puffercpu.c"

enum { H = 16, L = 4, STATE = H * L, MATCHES = 64, ROUNDS = 1000 };
static uint64_t rng_state;
static double uniform01(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return ((rng_state * UINT64_C(2685821657736338717)) >> 11) * 0x1.0p-53;
}
static int sample(const double p[3]) {
    double u = uniform01();
    return u < p[0] ? 0 : (u < p[0] + p[1] ? 1 : 2);
}
static void probabilities(PufferNet* net, int mine, int theirs, double p[3]) {
    float obs[6] = {0};
    if (mine >= 0) obs[mine] = 1;
    if (theirs >= 0) obs[3 + theirs] = 1;
    linear(net->encoder, obs);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    double max_logit = fmax(net->decoder->output[0],
        fmax(net->decoder->output[1], net->decoder->output[2]));
    double sum = 0;
    for (int i = 0; i < 3; i++) {
        p[i] = exp(net->decoder->output[i] - max_logit);
        sum += p[i];
    }
    for (int i = 0; i < 3; i++) {
        p[i] /= sum;
        assert(isfinite(p[i]));
    }
}
static void probes(FILE* out, PufferNet* net, const char* scenario, int match, int step) {
    float saved[STATE];
    memcpy(saved, net->mingru->state, sizeof(saved));
    for (int code = -1; code < 9; code++) {
        memcpy(net->mingru->state, saved, sizeof(saved));
        int mine = code < 0 ? -1 : code / 3;
        int theirs = code < 0 ? -1 : code % 3;
        double p[3];
        probabilities(net, mine, theirs, p);
        fprintf(out, "%s,%d,%d,%d,%d,%.10g,%.10g,%.10g\n",
            scenario, match, step, mine, theirs, p[0], p[1], p[2]);
    }
    memcpy(net->mingru->state, saved, sizeof(saved));
}
int main(int argc, char** argv) {
    assert(argc == 4 || argc == 5); // optional final argument enables parity mode
    Weights* w = load_weights(argv[1]);
    assert(w && w->size - 7 == 3232);
    int sizes[] = {3};
    PufferNet* net = make_puffernet(w, 1, 6, H, L, sizes, 1);
    assert(w->idx == 3232);
    if (argc == 5) {
        FILE* parity = fopen(argv[2], "w");
        assert(parity);
        fprintf(parity, "agent,step,p_rock,p_paper,p_scissors\n");
        for (int b = 0; b < 10; b++) {
            memset(net->mingru->state, 0, STATE * sizeof(float));
            for (int t = 0; t < ROUNDS; t++) {
                int code = t == 0 ? b - 1 : (t * 7 + b * 5 + t / 7) % 9;
                double p[3];
                probabilities(net, code < 0 ? -1 : code / 3, code < 0 ? -1 : code % 3, p);
                fprintf(parity, "%d,%d,%.10g,%.10g,%.10g\n", b, t, p[0], p[1], p[2]);
            }
        }
        fclose(parity);
        free_puffernet(net);
        free(w);
        return 0;
    }
    w->idx = 0;
    PufferNet* opponent = make_puffernet(w, 1, 6, H, L, sizes, 1);
    FILE* out = fopen(argv[2], "w");
    FILE* probe = fopen(argv[3], "w");
    assert(out && probe);
    fprintf(out, "scenario,match,step,previous_self,previous_opponent,p_rock,p_paper,p_scissors,action,opponent,reward\n");
    fprintf(probe, "scenario,match,step,previous_self,previous_opponent,p_rock,p_paper,p_scissors\n");
    probes(probe, net, "zero_memory", 0, 0);
    const char* scenarios[] = {"selfplay", "uniform", "always_rock", "always_paper",
        "always_scissors", "cycle", "copy_previous", "win_stay_lose_shift"};
    for (int s = 0; s < 8; s++) {
        for (int m = 0; m < MATCHES; m++) {
            rng_state = UINT64_C(20260926) + 100003 * s + 997 * m;
            memset(net->mingru->state, 0, STATE * sizeof(float));
            memset(opponent->mingru->state, 0, STATE * sizeof(float));
            int a_prev = -1, b_prev = -1, last_reward = 0;
            for (int t = 0; t < ROUNDS; t++) {
                if (t == 1 || t == 10 || t == 100 || t == 999)
                    probes(probe, net, scenarios[s], m, t);
                double p[3], q[3];
                probabilities(net, a_prev, b_prev, p);
                int a = sample(p), b;
                if (s == 0) {
                    probabilities(opponent, b_prev, a_prev, q);
                    b = sample(q);
                } else if (s == 1) b = (int)(3 * uniform01());
                else if (s <= 4) b = s - 2;
                else if (s == 5) b = (t + m) % 3;
                else if (s == 6) b = a_prev < 0 ? m % 3 : a_prev;
                else b = b_prev < 0 ? m % 3 : (last_reward > 0 ? (b_prev + 1) % 3 : b_prev);
                int reward = a == b ? 0 : ((a - b + 3) % 3 == 1 ? 1 : -1);
                fprintf(out, "%s,%d,%d,%d,%d,%.10g,%.10g,%.10g,%d,%d,%d\n",
                    scenarios[s], m, t, a_prev, b_prev, p[0], p[1], p[2], a, b, reward);
                a_prev = a;
                b_prev = b;
                last_reward = reward;
            }
        }
        fprintf(stderr, "Completed %s: %d matches x %d rounds\n", scenarios[s], MATCHES, ROUNDS);
    }
    fclose(out);
    fclose(probe);
    free_puffernet(net);
    free_puffernet(opponent);
    free(w);
    return 0;
}
