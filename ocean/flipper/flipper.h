/* Flipper: a simple single-agent pinball-like grid env.
 *
 * First skeleton:
 * - Fixed high-detail byte observation grid
 * - Static walls and bumpers
 * - Point-mass ball with position, velocity, acceleration
 * - Two flipper bars controlled by speed actions and binary hold signals
 */

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

/* Board sizing is fixed at compile time for this first version.
 * PufferLib bindings expose a static OBS_SIZE, so a fixed high-detail grid is
 * the simplest trainable shape. ROWS/COLS are the coarse design layer; DETAIL
 * expands that into the physics/observation grid used by collision logic.
 */
#define FLIPPER_ROWS 16
#define FLIPPER_COLS 12
#define FLIPPER_DETAIL 4
#define FLIPPER_PHYS_ROWS (FLIPPER_ROWS * FLIPPER_DETAIL)
#define FLIPPER_PHYS_COLS (FLIPPER_COLS * FLIPPER_DETAIL)
#define FLIPPER_OBS_SIZE (FLIPPER_PHYS_ROWS * FLIPPER_PHYS_COLS)

/* Physics constants are deliberately centralized here. The goal is to make the
 * first skeleton easy to tune without chasing magic numbers through the step
 * loop. Later, some of these can move into config kwargs if we want sweeps.
 */
#define FLIPPER_DEFAULT_MAX_TICKS 1200
#define FLIPPER_WALL_THICKNESS 2
#define FLIPPER_NUM_BUMPERS 3
#define FLIPPER_BUMPER_RADIUS 4.0f
#define FLIPPER_BUMPER_RADIUS2 (FLIPPER_BUMPER_RADIUS * FLIPPER_BUMPER_RADIUS)
#define FLIPPER_GRAVITY 0.035f
#define FLIPPER_DRAG 0.998f
#define FLIPPER_MAX_SPEED 3.25f
#define FLIPPER_WALL_RESTITUTION 0.84f
#define FLIPPER_BUMPER_RESTITUTION 1.18f
#define FLIPPER_RAISE_RATE 0.16f
#define FLIPPER_RELEASE_RATE 0.12f
#define FLIPPER_BASE_KICK 1.95f
#define FLIPPER_OMEGA_KICK 6.0f
#define FLIPPER_CONTACT_COOLDOWN 6
#define FLIPPER_BUMPER_COOLDOWN 5

/* Action layout:
 *   actions[0], actions[1] are continuous speed controls.
 *   actions[2], actions[3] are interpreted as binary hold controls.
 *
 * The binding advertises all four heads as continuous because the stock
 * PufferLib policy/backend does not currently support mixed continuous and
 * discrete action heads. Inside the env, hold is still treated as a binary
 * decision by thresholding at zero.
 */
#define ACTION_LEFT_SPEED 0
#define ACTION_RIGHT_SPEED 1
#define ACTION_LEFT_HOLD 2
#define ACTION_RIGHT_HOLD 3

/* Byte observation values. Keeping this as a compact grid makes the first
 * version close to my_env/squared and lets us render the same buffer we train
 * on. BALL is the agent-like dynamic object; everything else is map state.
 */
#define EMPTY 0
#define BALL 1
#define WALL 2
#define BUMPER 3
#define LEFT_FLIPPER 4
#define RIGHT_FLIPPER 5

/* Static bumper coordinates are intentionally global rather than stored per
 * env. Every vectorized env instance shares the same map, so per-env memory
 * should be reserved for mutable state only.
 */
static const int BUMPER_ROWS[FLIPPER_NUM_BUMPERS] = {
    FLIPPER_PHYS_ROWS / 4,
    FLIPPER_PHYS_ROWS / 3,
    FLIPPER_PHYS_ROWS / 3,
};

static const int BUMPER_COLS[FLIPPER_NUM_BUMPERS] = {
    FLIPPER_PHYS_COLS / 2,
    FLIPPER_PHYS_COLS / 3,
    2 * FLIPPER_PHYS_COLS / 3,
};

/* Required log struct. PufferLib aggregates this across env instances.
 * The fields are floats by convention, even when they represent counts.
 */
typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float bumper_hits;
    float flipper_hits;
    float drains;
    float timeouts;
    float n;
} Log;

typedef struct Flipper {
    /* PufferLib-required fields. The vecenv allocator fills the observation,
     * action, reward, and terminal pointers for each env instance.
     */
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    /* Mutable episode state. These are the only pieces that change as the
     * episode evolves; the map itself is static and regenerated into
     * observations when needed.
     */
    int tick;
    int max_ticks;
    float episode_return;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float ball_ax;
    float ball_ay;
    float left_phase;
    float right_phase;
    float left_omega;
    float right_omega;
    int left_contact_cooldown;
    int right_contact_cooldown;
    int bumper_cooldown;
    unsigned int rng;
} Flipper;

/* clampf is the small guardrail used throughout the env. It keeps actions,
 * phases, and normalized metrics inside the ranges expected by the rest of
 * the physics. Inlining avoids function-call overhead in the hot loop.
 */
static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

/* cell_idx maps a 2D grid coordinate into the flattened observation buffer.
 * Most ocean envs use flat arrays for speed and Python binding simplicity;
 * this helper keeps the row-major convention explicit.
 */
static inline int cell_idx(int row, int col) {
    return row * FLIPPER_PHYS_COLS + col;
}

/* in_bounds_cell distinguishes "safe to write to the observation buffer" from
 * "physically a wall." Bottom out-of-bounds is a drain, not a wall, so the two
 * concepts need to stay separate.
 */
static inline bool in_bounds_cell(int row, int col) {
    return row >= 0 && col >= 0
        && row < FLIPPER_PHYS_ROWS && col < FLIPPER_PHYS_COLS;
}

/* is_wall_cell defines the current obstacle subset of the map.
 * This is the first extension point for richer maps: add/remove coordinate
 * predicates here, or replace the helper with a static map array later.
 */
static inline bool is_wall_cell(int row, int col) {
    if (row < 0 || col < 0 || col >= FLIPPER_PHYS_COLS) {
        return true;
    }
    if (row >= FLIPPER_PHYS_ROWS) {
        return false;
    }
    if (row < FLIPPER_WALL_THICKNESS
            || col < FLIPPER_WALL_THICKNESS
            || col >= FLIPPER_PHYS_COLS - FLIPPER_WALL_THICKNESS) {
        return true;
    }

    // Two small static ledges make wall collision visible before richer maps.
    int ledge_row = FLIPPER_PHYS_ROWS / 2;
    if (row >= ledge_row && row < ledge_row + 2) {
        if (col >= FLIPPER_PHYS_COLS / 6 && col < FLIPPER_PHYS_COLS / 3) {
            return true;
        }
        if (col > 2 * FLIPPER_PHYS_COLS / 3 && col <= 5 * FLIPPER_PHYS_COLS / 6) {
            return true;
        }
    }
    return false;
}

/* is_bumper_cell defines the current bumper subset of the map for observation
 * drawing. Runtime collision uses the same centers/radius in resolve_bumpers,
 * so changing bumper geometry should happen in both places together.
 */
static inline bool is_bumper_cell(int row, int col) {
    for (int i = 0; i < FLIPPER_NUM_BUMPERS; i++) {
        int dr = row - BUMPER_ROWS[i];
        int dc = col - BUMPER_COLS[i];
        if ((float)(dr * dr + dc * dc) <= FLIPPER_BUMPER_RADIUS2) {
            return true;
        }
    }
    return false;
}

/* is_wall_at converts continuous ball coordinates to grid cells for collision
 * checks. The ball is still modeled as a point mass in this skeleton; a future
 * ball radius would likely replace this with neighborhood checks.
 */
static inline bool is_wall_at(float x, float y) {
    return is_wall_cell((int)floorf(y), (int)floorf(x));
}

/* put_obs_cell is a defensive write helper for render/observation generation.
 * It lets drawing code be a little loose near boundaries without risking an
 * out-of-bounds write into the flat observation array.
 */
static inline void put_obs_cell(Flipper* env, int row, int col, unsigned char value) {
    if (in_bounds_cell(row, col)) {
        env->observations[cell_idx(row, col)] = value;
    }
}

/* draw_flipper_bar writes a coarse line segment into observations.
 * This is visual/observational only: first-pass collision uses simple hit
 * zones below. Keeping drawing separate from physics makes it easy to improve
 * either side independently.
 */
static inline void draw_flipper_bar(Flipper* env, int left, float phase) {
    int len = FLIPPER_PHYS_COLS / 3;
    int lift = FLIPPER_DETAIL * 3;
    int pivot_y = FLIPPER_PHYS_ROWS - FLIPPER_DETAIL - 1;
    int gap = FLIPPER_DETAIL;
    int pivot_x = left
        ? FLIPPER_PHYS_COLS / 2 - gap
        : FLIPPER_PHYS_COLS / 2 + gap;
    unsigned char value = left ? LEFT_FLIPPER : RIGHT_FLIPPER;

    for (int i = 0; i < len; i++) {
        float t = (float)i / (float)(len - 1);
        int col = left ? pivot_x - i : pivot_x + i;
        int row = pivot_y - (int)(phase * (float)lift * t);
        put_obs_cell(env, row, col, value);
        put_obs_cell(env, row - 1, col, value);
    }
}

/* compute_observations rebuilds the byte grid from canonical state.
 * This is intentionally simple and a little redundant: static walls/bumpers
 * are redrawn every time, then dynamic flippers and ball are layered on top.
 * Once the design stabilizes, we can cache the static map and memcpy it.
 */
void compute_observations(Flipper* env) {
    memset(env->observations, 0, FLIPPER_OBS_SIZE * sizeof(unsigned char));

    for (int row = 0; row < FLIPPER_PHYS_ROWS; row++) {
        for (int col = 0; col < FLIPPER_PHYS_COLS; col++) {
            if (is_wall_cell(row, col)) {
                env->observations[cell_idx(row, col)] = WALL;
            } else if (is_bumper_cell(row, col)) {
                env->observations[cell_idx(row, col)] = BUMPER;
            }
        }
    }

    draw_flipper_bar(env, 1, env->left_phase);
    draw_flipper_bar(env, 0, env->right_phase);

    int ball_row = (int)roundf(env->ball_y);
    int ball_col = (int)roundf(env->ball_x);
    put_obs_cell(env, ball_row, ball_col, BALL);
}

/* init resets mutable episode bookkeeping without touching RNG or allocated
 * buffers. It is shared by my_init and c_reset so newly-created envs and reset
 * episodes start from the same internal invariants.
 */
void init(Flipper* env) {
    env->tick = 0;
    env->episode_return = 0.0f;
    env->left_phase = 0.0f;
    env->right_phase = 0.0f;
    env->left_omega = 0.0f;
    env->right_omega = 0.0f;
    env->left_contact_cooldown = 0;
    env->right_contact_cooldown = 0;
    env->bumper_cooldown = 0;
}

/* add_log records one completed episode into the aggregate Log.
 * Rewards are sparse in this skeleton, so perf is currently survival fraction
 * rather than "win rate." That gives us a useful signal before adding goals.
 */
void add_log(Flipper* env, int drained, int timeout) {
    float perf = (float)env->tick / (float)env->max_ticks;
    env->log.perf += clampf(perf, 0.0f, 1.0f);
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.drains += drained ? 1.0f : 0.0f;
    env->log.timeouts += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

/* c_reset is the required PufferLib reset hook.
 * It places the ball near the upper field with a small randomized x velocity,
 * clears flipper state, and immediately writes the first observation.
 */
void c_reset(Flipper* env) {
    init(env);
    env->ball_x = (float)(FLIPPER_PHYS_COLS / 2);
    env->ball_y = (float)(FLIPPER_PHYS_ROWS / 5);
    env->ball_vx = ((float)(rand_r(&env->rng) % 200) / 100.0f - 1.0f) * 0.35f;
    env->ball_vy = 0.65f;
    env->ball_ax = 0.0f;
    env->ball_ay = FLIPPER_GRAVITY;
    compute_observations(env);
}

/* update_flipper turns a speed + binary hold command into a bar phase.
 * Holding raises toward 1.0 at an action-scaled rate; releasing falls toward
 * 0.0 at a fixed spring-back rate. omega stores the phase delta for impulse
 * strength when the ball is hit.
 */
static inline void update_flipper(float speed, int hold, float* phase, float* omega) {
    float prev = *phase;
    speed = clampf(fabsf(speed), 0.0f, 1.0f);
    if (hold) {
        *phase += FLIPPER_RAISE_RATE * speed;
    } else {
        *phase -= FLIPPER_RELEASE_RATE;
    }
    *phase = clampf(*phase, 0.0f, 1.0f);
    *omega = *phase - prev;
}

/* clamp_ball_speed prevents one-step tunneling and keeps the toy physics
 * numerically tame. We still substep movement below, but this cap gives the
 * env a clear maximum velocity budget.
 */
static inline void clamp_ball_speed(Flipper* env) {
    float speed2 = env->ball_vx * env->ball_vx + env->ball_vy * env->ball_vy;
    float max2 = FLIPPER_MAX_SPEED * FLIPPER_MAX_SPEED;
    if (speed2 > max2) {
        float scale = FLIPPER_MAX_SPEED / sqrtf(speed2);
        env->ball_vx *= scale;
        env->ball_vy *= scale;
    }
}

/* move_ball_axis advances the point-mass ball and reflects against walls.
 * X and Y are resolved sequentially because it is easy to reason about and
 * robust enough for a grid skeleton. More exact swept collision can replace
 * this once the wall geometry becomes richer.
 */
static inline void move_ball_axis(Flipper* env, float dx, float dy) {
    float next_x = env->ball_x + dx;
    if (is_wall_at(next_x, env->ball_y)) {
        env->ball_vx = -env->ball_vx * FLIPPER_WALL_RESTITUTION;
    } else {
        env->ball_x = next_x;
    }

    float next_y = env->ball_y + dy;
    if (is_wall_at(env->ball_x, next_y)) {
        env->ball_vy = -env->ball_vy * FLIPPER_WALL_RESTITUTION;
    } else {
        env->ball_y = next_y;
    }
}

/* resolve_bumpers applies a radial impulse when the ball enters a bumper.
 * Bumpers are treated as circular force objects rather than wall cells: the
 * velocity is reflected around the bumper normal, the ball is nudged outside,
 * and a small reward/log count is emitted with cooldown to avoid repeated hits.
 */
static inline void resolve_bumpers(Flipper* env) {
    for (int i = 0; i < FLIPPER_NUM_BUMPERS; i++) {
        float dx = env->ball_x - (float)BUMPER_COLS[i];
        float dy = env->ball_y - (float)BUMPER_ROWS[i];
        float dist2 = dx * dx + dy * dy;
        if (dist2 > FLIPPER_BUMPER_RADIUS2) {
            continue;
        }

        float dist = sqrtf(fmaxf(dist2, 0.0001f));
        float nx = dx / dist;
        float ny = dy / dist;
        float dot = env->ball_vx * nx + env->ball_vy * ny;
        if (dot < 0.0f) {
            env->ball_vx -= (1.0f + FLIPPER_BUMPER_RESTITUTION) * dot * nx;
            env->ball_vy -= (1.0f + FLIPPER_BUMPER_RESTITUTION) * dot * ny;
            env->ball_x = (float)BUMPER_COLS[i] + nx * (FLIPPER_BUMPER_RADIUS + 0.2f);
            env->ball_y = (float)BUMPER_ROWS[i] + ny * (FLIPPER_BUMPER_RADIUS + 0.2f);
            if (env->bumper_cooldown == 0) {
                env->rewards[0] += 0.1f;
                env->episode_return += 0.1f;
                env->log.bumper_hits += 1.0f;
                env->bumper_cooldown = FLIPPER_BUMPER_COOLDOWN;
            }
        }
    }
}

/* ball_in_left_flipper_zone is the temporary collision proxy for the left bar.
 * It deliberately uses a rectangular hit zone so the first skeleton can focus
 * on action/physics wiring before implementing rotating-segment collision.
 */
static inline bool ball_in_left_flipper_zone(Flipper* env) {
    return env->ball_y >= (float)(FLIPPER_PHYS_ROWS - 3 * FLIPPER_DETAIL)
        && env->ball_y <= (float)(FLIPPER_PHYS_ROWS - 1)
        && env->ball_x >= (float)(FLIPPER_DETAIL)
        && env->ball_x <= (float)(FLIPPER_PHYS_COLS / 2 - FLIPPER_DETAIL);
}

/* ball_in_right_flipper_zone mirrors the left proxy zone.
 * When we later model true bar geometry, both zone helpers should collapse
 * into one distance-to-segment collision helper.
 */
static inline bool ball_in_right_flipper_zone(Flipper* env) {
    return env->ball_y >= (float)(FLIPPER_PHYS_ROWS - 3 * FLIPPER_DETAIL)
        && env->ball_y <= (float)(FLIPPER_PHYS_ROWS - 1)
        && env->ball_x >= (float)(FLIPPER_PHYS_COLS / 2 + FLIPPER_DETAIL)
        && env->ball_x <= (float)(FLIPPER_PHYS_COLS - FLIPPER_DETAIL);
}

/* resolve_flippers turns an upward-moving flipper phase into a ball impulse.
 * The impulse direction is hard-coded for now: left sends up-right, right
 * sends up-left. omega contributes extra strength, which is why action speed
 * matters even though hold is binary.
 */
static inline void resolve_flippers(Flipper* env) {
    if (env->left_contact_cooldown == 0
            && env->left_omega > 0.0f
            && ball_in_left_flipper_zone(env)) {
        float kick = FLIPPER_BASE_KICK + FLIPPER_OMEGA_KICK * env->left_omega;
        env->ball_vx = 0.65f + 0.25f * kick;
        env->ball_vy = -kick;
        env->left_contact_cooldown = FLIPPER_CONTACT_COOLDOWN;
        env->rewards[0] += 0.02f;
        env->episode_return += 0.02f;
        env->log.flipper_hits += 1.0f;
    }

    if (env->right_contact_cooldown == 0
            && env->right_omega > 0.0f
            && ball_in_right_flipper_zone(env)) {
        float kick = FLIPPER_BASE_KICK + FLIPPER_OMEGA_KICK * env->right_omega;
        env->ball_vx = -0.65f - 0.25f * kick;
        env->ball_vy = -kick;
        env->right_contact_cooldown = FLIPPER_CONTACT_COOLDOWN;
        env->rewards[0] += 0.02f;
        env->episode_return += 0.02f;
        env->log.flipper_hits += 1.0f;
    }
}

/* c_step is the required PufferLib step hook and the main sequential physics
 * loop. The ordering matters:
 *   actions -> flipper phases -> gravity/drag -> substepped movement
 *   -> bumper impulses -> flipper impulses -> terminal/reset/observation.
 */
void c_step(Flipper* env) {
    env->tick += 1;
    env->terminals[0] = 0.0f;
    env->rewards[0] = 0.0f;

    float left_speed = env->actions[ACTION_LEFT_SPEED];
    float right_speed = env->actions[ACTION_RIGHT_SPEED];
    int left_hold = env->actions[ACTION_LEFT_HOLD] > 0.0f;
    int right_hold = env->actions[ACTION_RIGHT_HOLD] > 0.0f;
    update_flipper(left_speed, left_hold, &env->left_phase, &env->left_omega);
    update_flipper(right_speed, right_hold, &env->right_phase, &env->right_omega);

    if (env->left_contact_cooldown > 0) {
        env->left_contact_cooldown--;
    }
    if (env->right_contact_cooldown > 0) {
        env->right_contact_cooldown--;
    }
    if (env->bumper_cooldown > 0) {
        env->bumper_cooldown--;
    }

    env->ball_ax = 0.0f;
    env->ball_ay = FLIPPER_GRAVITY;
    env->ball_vx = (env->ball_vx + env->ball_ax) * FLIPPER_DRAG;
    env->ball_vy = (env->ball_vy + env->ball_ay) * FLIPPER_DRAG;
    clamp_ball_speed(env);

    int substeps = (int)ceilf(fmaxf(fabsf(env->ball_vx), fabsf(env->ball_vy)));
    if (substeps < 1) {
        substeps = 1;
    }
    float dx = env->ball_vx / (float)substeps;
    float dy = env->ball_vy / (float)substeps;
    for (int i = 0; i < substeps; i++) {
        move_ball_axis(env, dx, dy);
        resolve_bumpers(env);
        resolve_flippers(env);
    }

    int drained = env->ball_y >= (float)FLIPPER_PHYS_ROWS;
    int timeout = env->tick >= env->max_ticks;
    if (drained || timeout) {
        env->terminals[0] = 1.0f;
        if (drained) {
            env->rewards[0] -= 1.0f;
            env->episode_return -= 1.0f;
        }
        add_log(env, drained, timeout);
        c_reset(env);
        return;
    }

    compute_observations(env);
}

/* c_render draws the current observation grid with raylib.
 * Rendering intentionally reads env->observations instead of recomputing
 * geometry, so the visual debug view matches exactly what the policy sees.
 */
void c_render(Flipper* env) {
    int px = 10;
    if (!IsWindowReady()) {
        InitWindow(FLIPPER_PHYS_COLS * px, FLIPPER_PHYS_ROWS * px, "PufferLib Flipper");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int row = 0; row < FLIPPER_PHYS_ROWS; row++) {
        for (int col = 0; col < FLIPPER_PHYS_COLS; col++) {
            unsigned char tex = env->observations[cell_idx(row, col)];
            if (tex == EMPTY) {
                continue;
            }
            Color color = (Color){241, 241, 241, 255};
            if (tex == BALL) {
                color = (Color){0, 187, 187, 255};
            } else if (tex == WALL) {
                color = (Color){100, 100, 120, 255};
            } else if (tex == BUMPER) {
                color = (Color){187, 0, 0, 255};
            } else if (tex == LEFT_FLIPPER || tex == RIGHT_FLIPPER) {
                color = (Color){255, 210, 80, 255};
            }
            DrawRectangle(col * px, row * px, px, px, color);
        }
    }

    DrawText(TextFormat("tick %d", env->tick), 8, 8, 16, (Color){241, 241, 241, 255});
    EndDrawing();
}

/* c_close is the required cleanup hook.
 * The vecenv owns observations/actions/rewards/terminals, so this only closes
 * the optional raylib window if rendering was used.
 */
void c_close(Flipper* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}
