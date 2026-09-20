#include "flipper.h"

/* Binding metadata consumed by src/vecenv.h and the Python extension.
 * OBS_SIZE must be a compile-time constant. The four action heads are exposed
 * as continuous heads for stock PufferLib compatibility; c_step thresholds the
 * last two into binary hold decisions.
 */
#define OBS_SIZE FLIPPER_OBS_SIZE
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T ByteTensor

#define Env Flipper
#include "vecenv.h"

/* my_init is called once per vectorized env instance before buffers are wired.
 * Keep it cheap: set static env parameters, declare single-agent ownership,
 * and initialize mutable bookkeeping. c_reset will place the ball after the
 * observation pointer is assigned by vecenv.
 */
void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_ticks = (int)dict_get(kwargs, "max_ticks")->value;
    init(env);
}

/* my_log controls which aggregate Log fields are visible to Python training.
 * These fields are intentionally early-design diagnostics: survival perf,
 * sparse score, interaction counts, and terminal causes.
 */
void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "bumper_hits", log->bumper_hits);
    dict_set(out, "flipper_hits", log->flipper_hits);
    dict_set(out, "drains", log->drains);
    dict_set(out, "timeouts", log->timeouts);
}
