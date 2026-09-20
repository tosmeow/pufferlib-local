/* Native PufferLib wrapper for Google Research Football.
 *
 * This intentionally keeps the PufferLib Ocean three-file shape while calling
 * the GFR C++ engine directly. Build as C++ (for example CC=clang++) and pass
 * the GFR engine sources/libs exactly as the upstream gfootball build does.
 */

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#ifndef PUFFERGF_GFR_ENGINE_HEADER
#define PUFFERGF_GFR_ENGINE_HEADER "../../GFR/third_party/gfootball_engine/src/game_env.hpp"
#endif
#include PUFFERGF_GFR_ENGINE_HEADER

#define GFR_OBS_SIZE 115
#define GFR_DEFAULT_ACTIONS 19

#ifndef PUFFERGF_GFR_DATA_DIR
#define PUFFERGF_GFR_DATA_DIR ""
#endif

#ifndef PUFFERGF_GFR_FONT
#define PUFFERGF_GFR_FONT ""
#endif

enum GFRScenario {
    GFR_ACADEMY_EMPTY_GOAL_CLOSE = 0,
    GFR_ACADEMY_EMPTY_GOAL = 1,
    GFR_ACADEMY_RUN_TO_SCORE = 2,
    GFR_11_VS_11_STOCHASTIC = 3,
};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float goals_for;
    float goals_against;
    float n;
} GFRLog;

typedef struct {
    GFRLog log;
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    GameEnv* game;
    SharedInfo info;
    int scenario;
    int tick;
    int previous_score_diff;
    int previous_game_mode;
    int prev_ball_owned_team;
    float episode_return;
    bool started;
} GFR;

static inline void gfr_add_player(SHARED_PTR<ScenarioConfig>& cfg, bool left,
        float x, float y, e_PlayerRole role, bool lazy=false, bool controllable=true) {
    FormationEntry player(x, y, role, lazy, controllable);
    if (left) {
        cfg->left_team.push_back(player);
    } else {
        cfg->right_team.push_back(player);
    }
}

static inline void gfr_set_ball(SHARED_PTR<ScenarioConfig>& cfg, float x, float y) {
    cfg->ball_position.coords[0] = x;
    cfg->ball_position.coords[1] = y;
    cfg->ball_position.coords[2] = 0.0f;
}

static inline void gfr_build_11_vs_11(SHARED_PTR<ScenarioConfig>& cfg, unsigned int seed) {
    cfg->game_duration = 3000;
    cfg->right_team_difficulty = 0.6f;
    cfg->deterministic = false;
    cfg->game_engine_random_seed = seed;
    cfg->reverse_team_processing = (seed & 1u) != 0;

    bool left_first = (seed & 1u) == 0;
    auto add_team = [&](bool left, bool first) {
        gfr_add_player(cfg, left, -1.000000f,  0.000000f, e_PlayerRole_GK);
        gfr_add_player(cfg, left, first ? 0.000000f : -0.050000f,
            first ? 0.020000f : 0.000000f, e_PlayerRole_RM);
        gfr_add_player(cfg, left, first ? 0.000000f : -0.010000f,
            first ? -0.020000f : 0.216102f, e_PlayerRole_CF);
        gfr_add_player(cfg, left, -0.422000f, -0.195760f, e_PlayerRole_LB);
        gfr_add_player(cfg, left, -0.500000f, -0.063560f, e_PlayerRole_CB);
        gfr_add_player(cfg, left, -0.500000f,  0.063559f, e_PlayerRole_CB);
        gfr_add_player(cfg, left, -0.422000f,  0.195760f, e_PlayerRole_RB);
        gfr_add_player(cfg, left, -0.184212f, -0.105680f, e_PlayerRole_CM);
        gfr_add_player(cfg, left, -0.267574f,  0.000000f, e_PlayerRole_CM);
        gfr_add_player(cfg, left, -0.184212f,  0.105680f, e_PlayerRole_CM);
        gfr_add_player(cfg, left, -0.010000f, -0.216100f, e_PlayerRole_LM);
    };

    add_team(true, left_first);
    add_team(false, !left_first);
}

static inline SHARED_PTR<ScenarioConfig> gfr_make_scenario(GFR* env) {
    auto cfg = ScenarioConfig::make();
    cfg->left_agents = 1;
    cfg->right_agents = 0;
    cfg->real_time = false;
    cfg->game_engine_random_seed = env->rng;
    cfg->reverse_team_processing = (env->rng & 1u) != 0;

    switch (env->scenario) {
    case GFR_ACADEMY_EMPTY_GOAL_CLOSE:
        cfg->game_duration = 400;
        cfg->deterministic = false;
        cfg->offsides = false;
        cfg->end_episode_on_score = true;
        cfg->end_episode_on_out_of_play = true;
        cfg->end_episode_on_possession_change = true;
        gfr_set_ball(cfg, 0.77f, 0.0f);
        gfr_add_player(cfg, true, -1.0f, 0.0f, e_PlayerRole_GK);
        gfr_add_player(cfg, true, 0.75f, 0.0f, e_PlayerRole_CB);
        gfr_add_player(cfg, false, 1.0f, 0.0f, e_PlayerRole_GK);
        break;
    case GFR_ACADEMY_EMPTY_GOAL:
        cfg->game_duration = 400;
        cfg->deterministic = false;
        cfg->offsides = false;
        cfg->end_episode_on_score = true;
        cfg->end_episode_on_out_of_play = true;
        cfg->end_episode_on_possession_change = true;
        gfr_set_ball(cfg, -0.1f, 0.0f);
        gfr_add_player(cfg, true, -1.0f, 0.0f, e_PlayerRole_GK);
        gfr_add_player(cfg, true, -0.2f, 0.0f, e_PlayerRole_CB);
        gfr_add_player(cfg, false, 1.0f, 0.0f, e_PlayerRole_GK);
        break;
    case GFR_ACADEMY_RUN_TO_SCORE:
        cfg->game_duration = 400;
        cfg->deterministic = false;
        cfg->offsides = false;
        cfg->end_episode_on_score = true;
        cfg->end_episode_on_out_of_play = true;
        cfg->end_episode_on_possession_change = true;
        gfr_set_ball(cfg, 0.02f, 0.0f);
        gfr_add_player(cfg, true, -1.0f, 0.0f, e_PlayerRole_GK);
        gfr_add_player(cfg, true, 0.0f, 0.0f, e_PlayerRole_CB);
        gfr_add_player(cfg, false, 1.0f, 0.0f, e_PlayerRole_GK);
        gfr_add_player(cfg, false, 0.12f, 0.2f, e_PlayerRole_LB);
        gfr_add_player(cfg, false, 0.12f, 0.1f, e_PlayerRole_CB);
        gfr_add_player(cfg, false, 0.12f, 0.0f, e_PlayerRole_CM);
        gfr_add_player(cfg, false, 0.12f, -0.1f, e_PlayerRole_CB);
        gfr_add_player(cfg, false, 0.12f, -0.2f, e_PlayerRole_RB);
        break;
    case GFR_11_VS_11_STOCHASTIC:
    default:
        gfr_build_11_vs_11(cfg, env->rng);
        break;
    }
    return cfg;
}

static inline int gfr_backend_action(int action) {
    static const int mapping[GFR_DEFAULT_ACTIONS] = {
        game_idle, game_left, game_top_left, game_top, game_top_right,
        game_right, game_bottom_right, game_bottom, game_bottom_left,
        game_long_pass, game_high_pass, game_short_pass, game_shot,
        game_sprint, game_release_direction, game_release_sprint,
        game_sliding, game_dribble, game_release_dribble,
    };
    action = std::max(0, std::min(action, GFR_DEFAULT_ACTIONS - 1));
    return mapping[action];
}

static inline void gfr_write_players(float*& obs,
        const std::vector<PlayerInfo>& players, bool directions) {
    for (int i = 0; i < 11; i++) {
        if (i < (int)players.size()) {
            const Position& p = directions ? players[i].player_direction
                                           : players[i].player_position;
            *obs++ = p.env_coord(0);
            *obs++ = p.env_coord(1);
        } else {
            *obs++ = -1.0f;
            *obs++ = -1.0f;
        }
    }
}

static inline int gfr_active_player(const SharedInfo& info) {
    if (!info.left_controllers.empty()) {
        return info.left_controllers[0].controlled_player;
    }
    for (int i = 0; i < (int)info.left_team.size(); i++) {
        if (info.left_team[i].designated_player) {
            return i;
        }
    }
    return -1;
}

static inline void gfr_compute_observation(GFR* env) {
    env->info = env->game->get_info();
    float* obs = (float*)env->observations;
    gfr_write_players(obs, env->info.left_team, false);
    gfr_write_players(obs, env->info.left_team, true);
    gfr_write_players(obs, env->info.right_team, false);
    gfr_write_players(obs, env->info.right_team, true);

    *obs++ = env->info.ball_position.env_coord(0);
    *obs++ = env->info.ball_position.env_coord(1);
    *obs++ = env->info.ball_position.env_coord(2);
    *obs++ = env->info.ball_direction.env_coord(0);
    *obs++ = env->info.ball_direction.env_coord(1);
    *obs++ = env->info.ball_direction.env_coord(2);

    *obs++ = env->info.ball_owned_team == -1 ? 1.0f : 0.0f;
    *obs++ = env->info.ball_owned_team == 0 ? 1.0f : 0.0f;
    *obs++ = env->info.ball_owned_team == 1 ? 1.0f : 0.0f;

    int active = gfr_active_player(env->info);
    for (int i = 0; i < 11; i++) {
        *obs++ = active == i ? 1.0f : 0.0f;
    }

    int mode = std::max(0, std::min((int)env->info.game_mode, 6));
    for (int i = 0; i < 7; i++) {
        *obs++ = mode == i ? 1.0f : 0.0f;
    }
}

static inline void gfr_add_log(GFR* env) {
    float result = env->episode_return > 0.0f ? 1.0f :
        (env->episode_return == 0.0f ? 0.5f : 0.0f);
    env->log.perf += result;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.goals_for += (float)env->info.left_goals;
    env->log.goals_against += (float)env->info.right_goals;
    env->log.n += 1.0f;
}

static inline void gfr_ensure_started(GFR* env) {
    if (env->game == nullptr) {
        env->game = new GameEnv();
        env->game->game_config.physics_steps_per_frame = 10;
        env->game->game_config.render = false;
        env->game->game_config.render_resolution_x = 1280;
        env->game->game_config.render_resolution_y = 720;
    }
    if (!env->started) {
        if (PUFFERGF_GFR_DATA_DIR[0] != '\0' && getenv("GFOOTBALL_DATA_DIR") == nullptr) {
            setenv("GFOOTBALL_DATA_DIR", PUFFERGF_GFR_DATA_DIR, 0);
        }
        if (PUFFERGF_GFR_FONT[0] != '\0' && getenv("GFOOTBALL_FONT") == nullptr) {
            setenv("GFOOTBALL_FONT", PUFFERGF_GFR_FONT, 0);
        }
        env->game->start_game();
        env->started = true;
    }
}

static inline void c_reset(GFR* env) {
    gfr_ensure_started(env);
    env->rng = 1664525u * env->rng + 1013904223u;
    auto scenario = gfr_make_scenario(env);
    {
        ContextHolder context(env->game);
        env->game->state = game_running;
        env->game->reset(*scenario, false);
        for (int i = 0; i < 2000; i++) {
            gfr_compute_observation(env);
            if (env->info.is_in_play) {
                break;
            }
            env->game->step();
        }
    }
    env->tick = 0;
    env->episode_return = 0.0f;
    env->previous_score_diff = env->info.left_goals - env->info.right_goals;
    env->previous_game_mode = (int)env->info.game_mode;
    env->prev_ball_owned_team = env->info.ball_owned_team;
}

static inline bool gfr_done(GFR* env) {
    if (env->game->scenario_config.end_episode_on_score &&
            (env->info.left_goals > 0 || env->info.right_goals > 0)) {
        return true;
    }
    if (env->game->scenario_config.end_episode_on_out_of_play &&
            (int)env->info.game_mode != (int)e_GameMode_Normal &&
            env->previous_game_mode == (int)e_GameMode_Normal) {
        return true;
    }
    if (env->game->scenario_config.end_episode_on_possession_change &&
            env->info.ball_owned_team != -1 &&
            env->prev_ball_owned_team != -1 &&
            env->info.ball_owned_team != env->prev_ball_owned_team) {
        return true;
    }
    return env->info.step >= env->game->scenario_config.game_duration;
}

static inline void c_step(GFR* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    env->tick += 1;

    {
        ContextHolder context(env->game);
        int action = (int)env->actions[0];
        env->game->action(gfr_backend_action(action), true, 0);

        for (int i = 0; i < 2000; i++) {
            env->game->step();
            gfr_compute_observation(env);
            if (env->info.is_in_play) {
                break;
            }
        }
    }

    int score_diff = env->info.left_goals - env->info.right_goals;
    float reward = (float)(score_diff - env->previous_score_diff);
    env->previous_score_diff = score_diff;
    env->rewards[0] = reward;
    env->episode_return += reward;

    bool done = gfr_done(env);
    if (env->info.ball_owned_team != -1) {
        env->prev_ball_owned_team = env->info.ball_owned_team;
    }
    env->previous_game_mode = (int)env->info.game_mode;

    if (done) {
        env->terminals[0] = 1.0f;
        env->game->state = game_done;
        gfr_add_log(env);
        c_reset(env);
    }
}

static inline void c_render(GFR* env) {
    printf("GFR step=%d score=%d:%d reward=%.2f ball=(%.3f, %.3f)\n",
        env->info.step, env->info.left_goals, env->info.right_goals,
        env->episode_return,
        env->info.ball_position.env_coord(0), env->info.ball_position.env_coord(1));
}

static inline void c_close(GFR* env) {
    if (env->game != nullptr) {
        /* GFR's full quit_game() path is process-global and can crash when
         * invoked from a headless standalone/Puffer teardown. Match the Python
         * wrapper's engine-pool behavior: leave engine-owned globals alone and
         * let process exit reclaim them after training.
         */
        delete env->game;
        env->game = nullptr;
        env->started = false;
    }
}
