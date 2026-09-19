#include "queue_reactive.h"

#define OBS_SIZE QR_OBS_SIZE
#define NUM_ATNS 4
#define ACT_SIZES {QR_NUM_ACTION_TYPES, QR_NUM_SIDES, QR_ACTION_PRICE_CHOICES, QR_SIZE_BUCKETS}
#define OBS_TENSOR_T FloatTensor

#define Env QueueReactive
#define MY_DIAGNOSTIC_SIZE 1
static void my_diagnostic(Env* env, float* out);
#include "vecenv.h"

static void my_diagnostic(Env* env, float* out) {
    out[0] = (float)qr_mid_price(env);
}

static int get_int_default(Dict* kwargs, const char* key, int value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? value : (int)item->value;
}

static double get_double_default(Dict* kwargs, const char* key, double value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? value : item->value;
}

static const char* get_str_default(Dict* kwargs, const char* key, const char* value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL || item->ptr == NULL ? value : (const char*)item->ptr;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    QRConfig config;
    qr_config_defaults(&config);

    config.params_path = get_str_default(kwargs, "params_path", config.params_path);
    config.latency_path = get_str_default(kwargs, "latency_path", config.latency_path);
    config.use_calibrated_params = get_int_default(kwargs, "use_calibrated_params", config.use_calibrated_params);
    config.use_mixture_delta_t = get_int_default(kwargs, "use_mixture_delta_t", config.use_mixture_delta_t);
    config.use_total_lvl = get_int_default(kwargs, "use_total_lvl", config.use_total_lvl);
    config.use_power_law_impact = get_int_default(kwargs, "use_power_law_impact", config.use_power_law_impact);
    config.use_alpha = get_int_default(kwargs, "use_alpha", config.use_alpha);
    config.use_constant_alpha = get_int_default(
        kwargs, "use_constant_alpha", config.use_constant_alpha);
    config.strategy_impact = get_int_default(kwargs, "strategy_impact", config.strategy_impact);
    config.market_residual_rests = get_int_default(
        kwargs, "market_residual_rests", config.market_residual_rests);
    config.lot_size = get_int_default(kwargs, "lot_size", config.lot_size);
    config.report_interval = get_int_default(kwargs, "report_interval", config.report_interval);
    config.max_events_per_step = get_int_default(kwargs, "max_events_per_step", config.max_events_per_step);
    config.initial_bid = get_int_default(kwargs, "initial_bid", config.initial_bid);
    config.initial_ask = get_int_default(kwargs, "initial_ask", config.initial_ask);
    config.qr_dt_mean_ns = get_double_default(kwargs, "qr_dt_mean_ns", config.qr_dt_mean_ns);
    config.event_add_prob = get_double_default(kwargs, "event_add_prob", config.event_add_prob);
    config.event_cancel_prob = get_double_default(kwargs, "event_cancel_prob", config.event_cancel_prob);
    config.event_trade_prob = get_double_default(kwargs, "event_trade_prob", config.event_trade_prob);
    config.event_create_prob = get_double_default(kwargs, "event_create_prob", config.event_create_prob);
    config.latency_mu = get_double_default(kwargs, "latency_mu", config.latency_mu);
    config.latency_sigma = get_double_default(kwargs, "latency_sigma", config.latency_sigma);
    config.latency_lower = get_double_default(kwargs, "latency_lower", config.latency_lower);
    config.latency_upper = get_double_default(kwargs, "latency_upper", config.latency_upper);
    config.alpha_kappa = get_double_default(kwargs, "alpha_kappa", config.alpha_kappa);
    config.alpha_sigma = get_double_default(kwargs, "alpha_sigma", config.alpha_sigma);
    config.alpha_scale = get_double_default(kwargs, "alpha_scale", config.alpha_scale);
    config.constant_alpha = get_double_default(
        kwargs, "constant_alpha", config.constant_alpha);
    config.impact_beta = get_double_default(kwargs, "impact_beta", config.impact_beta);
    config.impact_tau = get_double_default(kwargs, "impact_tau", config.impact_tau);
    config.impact_m = get_double_default(kwargs, "impact_m", config.impact_m);
    config.impact_components = get_int_default(kwargs, "impact_components", config.impact_components);
    config.reward_interval_ms = get_int_default(kwargs, "reward_interval_ms", config.reward_interval_ms);
    config.pnl_reward_divisor = get_double_default(
        kwargs, "pnl_reward_divisor", config.pnl_reward_divisor);
    config.inventory_penalty_coef = get_double_default(
        kwargs, "inventory_penalty_coef", config.inventory_penalty_coef);
    config.agent_rejection_penalty = get_double_default(
        kwargs, "agent_rejection_penalty", config.agent_rejection_penalty);
    config.episode_duration_seconds = get_double_default(
        kwargs, "episode_duration_seconds", config.episode_duration_seconds);
    config.terminal_inventory_target = get_double_default(
        kwargs, "terminal_inventory_target", config.terminal_inventory_target);
    config.terminal_inventory_penalty_coef = get_double_default(
        kwargs, "terminal_inventory_penalty_coef", config.terminal_inventory_penalty_coef);

    config.size_buckets[0] = get_int_default(kwargs, "size_0", config.size_buckets[0]);
    config.size_buckets[1] = get_int_default(kwargs, "size_1", config.size_buckets[1]);
    config.size_buckets[2] = get_int_default(kwargs, "size_2", config.size_buckets[2]);
    config.size_buckets[3] = get_int_default(kwargs, "size_3", config.size_buckets[3]);
    config.size_buckets[4] = get_int_default(kwargs, "size_4", config.size_buckets[4]);
    config.size_buckets[5] = get_int_default(kwargs, "size_5", config.size_buckets[5]);
    config.mes[0] = get_int_default(kwargs, "mes_0", config.mes[0]);
    config.mes[1] = get_int_default(kwargs, "mes_1", config.mes[1]);
    config.mes[2] = get_int_default(kwargs, "mes_2", config.mes[2]);
    config.mes[3] = get_int_default(kwargs, "mes_3", config.mes[3]);
    config.bid_volumes[0] = get_int_default(kwargs, "bid_volume_0", config.bid_volumes[0]);
    config.bid_volumes[1] = get_int_default(kwargs, "bid_volume_1", config.bid_volumes[1]);
    config.bid_volumes[2] = get_int_default(kwargs, "bid_volume_2", config.bid_volumes[2]);
    config.bid_volumes[3] = get_int_default(kwargs, "bid_volume_3", config.bid_volumes[3]);
    config.ask_volumes[0] = get_int_default(kwargs, "ask_volume_0", config.ask_volumes[0]);
    config.ask_volumes[1] = get_int_default(kwargs, "ask_volume_1", config.ask_volumes[1]);
    config.ask_volumes[2] = get_int_default(kwargs, "ask_volume_2", config.ask_volumes[2]);
    config.ask_volumes[3] = get_int_default(kwargs, "ask_volume_3", config.ask_volumes[3]);

    qr_configure(env, &config);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "agent_fills", log->agent_fills);
    dict_set(out, "agent_orders", log->agent_orders);
    dict_set(out, "agent_cancels", log->agent_cancels);
    dict_set(out, "agent_rejected", log->agent_rejected);
    dict_set(out, "agent_lost_race", log->agent_lost_race);
    dict_set(out, "qr_events", log->qr_events);
    dict_set(out, "qr_trades", log->qr_trades);
    dict_set(out, "inventory", log->inventory);
    dict_set(out, "inventory_penalty", log->inventory_penalty);
    dict_set(out, "terminal_inventory_penalty", log->terminal_inventory_penalty);
    dict_set(out, "agent_rejection_penalty", log->agent_rejection_penalty);
}
