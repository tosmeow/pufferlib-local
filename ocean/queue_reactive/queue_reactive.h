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
    QRAgentOrder orders[QR_MAX_ORDERS];
    int next_order_id;
} QROwnOrders;

typedef struct {
    int inventory;
    double cash;
} QRAgentAccount;

typedef struct {
    int type;
    int side;
    int rejected;
    int partial;
    int filled;
    int price;
    int lost_race;
} QRActionTrace;

typedef struct {
    int type;
    int side;
    int size;
    int price;
    int rejected;
    int partial;
} QREventTrace;

typedef struct {
    int step;
    int64_t time_ns;
    int64_t last_qr_dt;
    int64_t last_latency_dt;
} QRClock;

typedef struct {
    int config_error;
    int calibration_loaded;
    double last_alpha;
    double last_impact_bias;
} QRRuntimeStatus;

typedef struct {
    // Puffer vector-env ABI fields. vecenv.h writes these directly.
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    // Opaque C++ simulator internals live in queue_reactive.cpp.
    void* sim;

    QROwnOrders own;
    QRAgentAccount account;
    QRActionTrace last_action;
    QREventTrace last_qr;
    QRClock clock;
    QRRuntimeStatus status;
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
