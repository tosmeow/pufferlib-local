#pragma once

#include <stdint.h>

#define QR_LEVELS 4
#define QR_MAX_ORDERS 8
#define QR_SIZE_BUCKETS 6
#define QR_ACTION_PRICE_CHOICES 8

#define QR_ACTION_NOOP 0
#define QR_ACTION_LIMIT 1
#define QR_ACTION_CANCEL 2
#define QR_ACTION_MARKET 3
#define QR_ACTION_IMPROVE 4
#define QR_NUM_ACTION_TYPES 5

#define QR_SIDE_BUY 0
#define QR_SIDE_SELL 1
#define QR_NUM_SIDES 2

#define QR_STATE_OBS 12
#define QR_LAST_OBS 12
#define QR_ORDER_OBS 5
#define QR_OBS_SIZE (QR_LEVELS * 4 + QR_STATE_OBS + QR_LAST_OBS + QR_MAX_ORDERS * QR_ORDER_OBS)

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float agent_fills;
    float agent_orders;
    float agent_cancels;
    float agent_rejected;
    float agent_lost_race;
    float qr_events;
    float qr_trades;
    float inventory;
    float n;
} Log;

typedef struct {
    const char* params_path;
    const char* latency_path;
    int use_calibrated_params;
    int use_mixture_delta_t;
    int use_total_lvl;
    int use_power_law_impact;
    int use_alpha;
    int strategy_impact;
    int lot_size;
    int report_interval;
    int max_events_per_step;
    int initial_bid;
    int initial_ask;
    int size_buckets[QR_SIZE_BUCKETS];
    int mes[QR_LEVELS];
    int bid_volumes[QR_LEVELS];
    int ask_volumes[QR_LEVELS];
    double qr_dt_mean_ns;
    double event_add_prob;
    double event_cancel_prob;
    double event_trade_prob;
    double event_create_prob;
    double latency_mu;
    double latency_sigma;
    double latency_lower;
    double latency_upper;
    double alpha_kappa;
    double alpha_sigma;
    double alpha_scale;
    double impact_beta;
    double impact_tau;
    double impact_m;
    int impact_components;
} QRConfig;

typedef struct {
    int active;
    int side;
    int price;
    int remaining;
    int ahead;
    int id;
} QRAgentOrder;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    void* sim;
    QRAgentOrder orders[QR_MAX_ORDERS];

    int step;
    int configured;
    int config_error;
    int calibration_loaded;
    int next_order_id;
    int inventory;
    int last_action_type;
    int last_action_side;
    int last_action_rejected;
    int last_action_partial;
    int last_action_filled;
    int last_action_price;
    int last_action_lost_race;
    int last_qr_type;
    int last_qr_side;
    int last_qr_size;
    int last_qr_price;
    int last_qr_rejected;
    int last_qr_partial;
    int64_t time_ns;
    int64_t last_qr_dt;
    int64_t last_latency_dt;
    double cash;
    double last_alpha;
    double last_impact_bias;
} QueueReactive;

#ifdef __cplusplus
extern "C" {
#endif

void qr_config_defaults(QRConfig* config);
void qr_configure(QueueReactive* env, const QRConfig* config);
void c_reset(QueueReactive* env);
void c_step(QueueReactive* env);
void c_render(QueueReactive* env);
void c_close(QueueReactive* env);

#ifdef __cplusplus
}
#endif
