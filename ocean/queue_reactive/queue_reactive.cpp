#include "queue_reactive.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orderbook.h"
#include "qr_model.h"
#include "raylib.h"

namespace {

constexpr float PRICE_NORM = 16.0f;
constexpr float VOLUME_NORM = 5000.0f;
constexpr float INVENTORY_NORM = 20000.0f;
constexpr float CASH_NORM = 30000000.0f;
constexpr double DEFAULT_LATENCY_MU = 5.0;
constexpr double DEFAULT_LATENCY_SIGMA = 0.25;
constexpr double DEFAULT_LATENCY_LOWER = 3.0;
constexpr double DEFAULT_LATENCY_UPPER = 7.0;

struct LatencyModel {
    double mu = DEFAULT_LATENCY_MU;
    double sigma = DEFAULT_LATENCY_SIGMA;
    double lower = DEFAULT_LATENCY_LOWER;
    double upper = DEFAULT_LATENCY_UPPER;

    void load_csv(const std::string& path) {
        if (path.empty()) return;
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open latency CSV: " + path);
        }
        std::string line;
        std::getline(file, line);
        if (!std::getline(file, line)) return;
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        mu = std::stod(token);
        std::getline(ss, token, ',');
        sigma = std::stod(token);
        std::getline(ss, token, ',');
        lower = std::stod(token);
        std::getline(ss, token, ',');
        upper = std::stod(token);
    }

    int64_t sample(std::mt19937_64& rng) const {
        std::normal_distribution<> normal(mu, sigma);
        double x = mu;
        for (int i = 0; i < 64; i++) {
            x = normal(rng);
            if (x >= lower && x <= upper) break;
        }
        x = std::clamp(x, lower, upper);
        return static_cast<int64_t>(std::pow(10.0, x));
    }
};

struct QRSim {
    explicit QRSim(const QRConfig& cfg, uint64_t seed)
        : cfg(cfg),
          seed(seed),
          latency_rng(seed ^ 0x9e3779b97f4a7c15ULL),
          queue_rng(seed ^ 0xd1b54a32d192ed03ULL) {
        params_path = cfg.params_path ? cfg.params_path : "";
        latency_path = cfg.latency_path ? cfg.latency_path : "";
        setup_static_models();
        try_load_calibrated_models();
        reset_book();
    }

    QRConfig cfg{};
    uint64_t seed = 1;
    std::string params_path;
    std::string latency_path;
    qr::QueueDistributions dists;
    qr::QRParams params;
    qr::SizeDistributions size_dists;
    std::unique_ptr<qr::MixtureDeltaT> mixture_delta;
    std::unique_ptr<qr::OrderBook> lob;
    std::unique_ptr<qr::QRModel> model;
    std::unique_ptr<qr::MarketImpact> impact;
    std::unique_ptr<qr::Alpha> alpha;
    LatencyModel latency;
    std::mt19937_64 latency_rng;
    std::mt19937_64 queue_rng;
    std::array<int32_t, QR_LEVELS> mes{1, 1, 1, 1};
    bool calibrated_loaded = false;

    void setup_static_models() {
        for (int i = 0; i < QR_LEVELS; i++) {
            mes[i] = std::max(1, cfg.mes[i]);
        }
        dists.set_mes(mes);
        fill_default_queue_distributions();
        fill_default_qr_params();
        fill_default_size_distributions();
        latency.mu = cfg.latency_mu;
        latency.sigma = cfg.latency_sigma;
        latency.lower = cfg.latency_lower;
        latency.upper = cfg.latency_upper;
        if (!latency_path.empty()) {
            latency.load_csv(latency_path);
        } else if (!params_path.empty()) {
            std::ifstream f(params_path + "/delta_distrib.csv");
            if (f.good()) latency.load_csv(params_path + "/delta_distrib.csv");
        }
    }

    void try_load_calibrated_models() {
        if (!cfg.use_calibrated_params || params_path.empty()) return;
        qr::QueueDistributions loaded_dists(params_path + "/invariant_distributions_qmax100.csv");
        loaded_dists.set_mes(mes);
        qr::QRParams loaded_params(params_path);
        if (cfg.use_total_lvl) {
            loaded_params.load_total_lvl_quantiles(params_path + "/total_lvl_quantiles.csv");
            loaded_params.load_event_probabilities_3d(params_path + "/event_probabilities_3D.csv");
        }
        qr::SizeDistributions loaded_sizes(params_path + "/size_distrib.csv");
        dists = std::move(loaded_dists);
        params = std::move(loaded_params);
        size_dists = std::move(loaded_sizes);
        if (cfg.use_mixture_delta_t) {
            mixture_delta = std::make_unique<qr::MixtureDeltaT>(params_path + "/delta_t_gmm.csv");
        }
        calibrated_loaded = true;
    }

    void reset_book() {
        lob = std::make_unique<qr::OrderBook>(dists, QR_LEVELS, seed);
        std::vector<int32_t> bid_prices(QR_LEVELS);
        std::vector<int32_t> ask_prices(QR_LEVELS);
        std::vector<int32_t> bid_vols(QR_LEVELS);
        std::vector<int32_t> ask_vols(QR_LEVELS);
        for (int i = 0; i < QR_LEVELS; i++) {
            bid_prices[i] = cfg.initial_bid - (QR_LEVELS - 1 - i);
            ask_prices[i] = cfg.initial_ask + i;
            bid_vols[i] = std::max(1, cfg.bid_volumes[QR_LEVELS - 1 - i] / mes[QR_LEVELS - 1 - i]);
            ask_vols[i] = std::max(1, cfg.ask_volumes[i] / mes[i]);
        }
        lob->init(bid_prices, bid_vols, ask_prices, ask_vols);

        if (mixture_delta) {
            model = std::make_unique<qr::QRModel>(lob.get(), params, size_dists, *mixture_delta, seed ^ 0x51edULL);
        } else {
            model = std::make_unique<qr::QRModel>(lob.get(), params, size_dists, seed ^ 0x51edULL);
        }
        model->set_mes(mes);

        if (cfg.use_power_law_impact) {
            impact = std::make_unique<qr::PowerLawImpact>(
                cfg.impact_beta, cfg.impact_tau, cfg.impact_m, std::max(2, cfg.impact_components));
            impact->set_mes(mes[0]);
        } else {
            impact = std::make_unique<qr::NoImpact>();
        }

        if (cfg.use_alpha) {
            alpha = std::make_unique<qr::OUAlpha>(cfg.alpha_kappa, cfg.alpha_sigma, seed ^ 0xa17aULL, cfg.alpha_scale);
        } else {
            alpha = std::make_unique<qr::NoAlpha>();
        }
    }

    void fill_default_queue_distributions() {
        for (int s = 0; s < 2; s++) {
            for (int level = 0; level < QR_LEVELS; level++) {
                auto& cdf = dists.cum_probs[s][level];
                cdf.resize(qr::QueueDistributions::MAX_Q_SIZE + 1);
                for (int i = 0; i <= qr::QueueDistributions::MAX_Q_SIZE; i++) {
                    cdf[i] = static_cast<double>(i + 1) / static_cast<double>(qr::QueueDistributions::MAX_Q_SIZE + 1);
                }
            }
        }
    }

    void fill_state(qr::StateParams& sp, bool wide_spread) {
        sp.events.clear();
        sp.base_probs.clear();
        auto add_event = [&](qr::OrderType type, qr::Side side, int queue, double prob) {
            if (prob <= 0.0) return;
            sp.events.push_back({type, side, queue});
            sp.base_probs.push_back(prob);
        };
        const double add = std::max(0.0, cfg.event_add_prob);
        const double cancel = std::max(0.0, cfg.event_cancel_prob);
        const double trade = std::max(0.0, cfg.event_trade_prob);
        const double create = wide_spread ? std::max(0.0, cfg.event_create_prob) : 0.0;
        add_event(qr::OrderType::Add, qr::Side::Bid, -1, 0.30 * add);
        add_event(qr::OrderType::Add, qr::Side::Bid, -2, 0.20 * add);
        add_event(qr::OrderType::Add, qr::Side::Ask, 1, 0.30 * add);
        add_event(qr::OrderType::Add, qr::Side::Ask, 2, 0.20 * add);
        add_event(qr::OrderType::Cancel, qr::Side::Bid, -1, 0.30 * cancel);
        add_event(qr::OrderType::Cancel, qr::Side::Bid, -2, 0.20 * cancel);
        add_event(qr::OrderType::Cancel, qr::Side::Ask, 1, 0.30 * cancel);
        add_event(qr::OrderType::Cancel, qr::Side::Ask, 2, 0.20 * cancel);
        add_event(qr::OrderType::Trade, qr::Side::Bid, -1, 0.50 * trade);
        add_event(qr::OrderType::Trade, qr::Side::Ask, 1, 0.50 * trade);
        add_event(qr::OrderType::CreateBid, qr::Side::Bid, 0, 0.50 * create);
        add_event(qr::OrderType::CreateAsk, qr::Side::Ask, 0, 0.50 * create);
        if (sp.events.empty()) {
            add_event(qr::OrderType::Add, qr::Side::Bid, -1, 0.5);
            add_event(qr::OrderType::Add, qr::Side::Ask, 1, 0.5);
        }
        sp.probs = sp.base_probs;
        sp.cum_probs.resize(sp.probs.size());
        sp.total = 0.0;
        for (size_t i = 0; i < sp.probs.size(); i++) {
            sp.total += sp.probs[i];
            sp.cum_probs[i] = sp.total;
        }
        sp.lambda = 1.0 / std::max(1.0, cfg.qr_dt_mean_ns);
    }

    void fill_default_qr_params() {
        params = qr::QRParams();
        for (auto& row : params.state_params) {
            fill_state(row[0], false);
            fill_state(row[1], true);
        }
        params.use_total_lvl = false;
    }

    void fill_default_size_distributions() {
        size_dists = qr::SizeDistributions();
        auto fill_cdf = [](std::vector<double>& cdf) {
            cdf.resize(qr::SizeDistributions::MAX_SIZE);
            for (int i = 0; i < qr::SizeDistributions::MAX_SIZE; i++) {
                cdf[i] = static_cast<double>(i + 1) / static_cast<double>(qr::SizeDistributions::MAX_SIZE);
            }
        };
        for (auto& imb : size_dists.cum_probs) {
            for (auto& type : imb) {
                for (auto& queue : type) fill_cdf(queue);
            }
        }
        for (auto& imb : size_dists.cum_probs_create) {
            for (auto& create : imb) fill_cdf(create);
        }
    }
};

static QRSim* sim(QueueReactive* env) {
    return static_cast<QRSim*>(env->sim);
}

static int clamp_int(int value, int lo, int hi) {
    return std::max(lo, std::min(value, hi));
}

static int action_size(const QRSim* s, float raw) {
    int idx = clamp_int(static_cast<int>(raw), 0, QR_SIZE_BUCKETS - 1);
    return std::max(1, s->cfg.size_buckets[idx]) * std::max(1, s->cfg.lot_size);
}

static qr::Side passive_side(int side) {
    return side == QR_SIDE_BUY ? qr::Side::Bid : qr::Side::Ask;
}

static qr::Side trade_side(int side) {
    return side == QR_SIDE_BUY ? qr::Side::Ask : qr::Side::Bid;
}

static int agent_side(qr::Side side) {
    return side == qr::Side::Bid ? QR_SIDE_BUY : QR_SIDE_SELL;
}

static int safe_volume_at(qr::OrderBook& lob, qr::Side side, int price) {
    try {
        return lob.volume_at(side, price);
    } catch (...) {
        return 0;
    }
}

static int level_price(qr::OrderBook& lob, int side, int level) {
    level = clamp_int(level, 0, QR_LEVELS - 1);
    if (side == QR_SIDE_BUY) {
        return lob.best_bid() - level;
    }
    return lob.best_ask() + level;
}

static int market_limit_price(qr::OrderBook& lob, int side, int level) {
    level = clamp_int(level, 0, QR_LEVELS - 1);
    if (side == QR_SIDE_BUY) {
        return lob.best_ask() + level;
    }
    return lob.best_bid() - level;
}

static int own_remaining_at(QueueReactive* env, int side, int price) {
    int remaining = 0;
    for (int i = 0; i < QR_MAX_ORDERS; i++) {
        const QRAgentOrder& order = env->orders[i];
        if (order.active && order.side == side && order.price == price) {
            remaining += order.remaining;
        }
    }
    return remaining;
}

static int first_free_slot(QueueReactive* env) {
    for (int i = 0; i < QR_MAX_ORDERS; i++) {
        if (!env->orders[i].active) return i;
    }
    return -1;
}

static int collect_orders_at(
        QueueReactive* env,
        int side,
        int price,
        std::array<int, QR_MAX_ORDERS>& slots) {
    int count = 0;
    for (int i = 0; i < QR_MAX_ORDERS; i++) {
        const QRAgentOrder& order = env->orders[i];
        if (order.active && order.side == side && order.price == price) {
            slots[count++] = i;
        }
    }

    std::sort(slots.begin(), slots.begin() + count, [&](int a, int b) {
        const QRAgentOrder& lhs = env->orders[a];
        const QRAgentOrder& rhs = env->orders[b];
        if (lhs.ahead != rhs.ahead) return lhs.ahead < rhs.ahead;
        return lhs.id < rhs.id;
    });
    return count;
}

static bool worse_agent_order(const QRAgentOrder& candidate, const QRAgentOrder& incumbent) {
    if (candidate.price != incumbent.price) {
        if (candidate.side == QR_SIDE_BUY) return candidate.price < incumbent.price;
        return candidate.price > incumbent.price;
    }
    if (candidate.ahead != incumbent.ahead) return candidate.ahead > incumbent.ahead;
    return candidate.id > incumbent.id;
}

static int worst_order_slot(QueueReactive* env, int side) {
    int slot = -1;
    for (int i = 0; i < QR_MAX_ORDERS; i++) {
        const QRAgentOrder& order = env->orders[i];
        if (!order.active || order.side != side) continue;
        if (slot < 0 || worse_agent_order(order, env->orders[slot])) slot = i;
    }
    return slot;
}

static void reduce_orders_behind(QueueReactive* env, int side, int price, int order_id, int qty) {
    if (qty <= 0) return;
    for (int i = 0; i < QR_MAX_ORDERS; i++) {
        QRAgentOrder& order = env->orders[i];
        if (!order.active || order.side != side || order.price != price || order.id <= order_id) continue;
        order.ahead = std::max(0, order.ahead - qty);
    }
}

static void fill_agent_order(QueueReactive* env, QRAgentOrder& order, int fill_size) {
    if (fill_size <= 0) return;
    if (order.side == QR_SIDE_BUY) {
        env->inventory += fill_size;
        env->cash -= static_cast<double>(order.price) * fill_size;
    } else {
        env->inventory -= fill_size;
        env->cash += static_cast<double>(order.price) * fill_size;
    }
    order.remaining -= fill_size;
    env->last_action_filled += fill_size;
    env->log.agent_fills += 1.0f;
    reduce_orders_behind(env, order.side, order.price, order.id, fill_size);
    if (order.remaining <= 0) {
        order = {};
    }
}

static void apply_passive_fills(QueueReactive* env, int resting_side, int price, int traded) {
    int remaining_trade = traded;
    std::array<int, QR_MAX_ORDERS> slots{};
    int count = collect_orders_at(env, resting_side, price, slots);
    for (int i = 0; i < count && remaining_trade > 0; i++) {
        QRAgentOrder& order = env->orders[slots[i]];
        if (!order.active) continue;
        int ahead_take = std::min(remaining_trade, order.ahead);
        order.ahead -= ahead_take;
        remaining_trade -= ahead_take;
        if (remaining_trade <= 0) break;
        int fill = std::min(remaining_trade, order.remaining);
        remaining_trade -= fill;
        fill_agent_order(env, order, fill);
    }
}

static int sample_public_cancel_segment(
        std::mt19937_64& rng,
        int segment,
        int remaining_public,
        int remaining_cancel) {
    if (segment <= 0 || remaining_public <= 0 || remaining_cancel <= 0) return 0;
    if (segment >= remaining_public) return remaining_cancel;
    double p = static_cast<double>(segment) / static_cast<double>(remaining_public);
    std::binomial_distribution<int> binom(remaining_cancel, p);
    return std::min(segment, binom(rng));
}

static void sample_public_cancel_decrements(
        QueueReactive* env,
        QRSim* s,
        int side,
        int price,
        int level_volume,
        int cancel_size,
        std::array<int, QR_MAX_ORDERS>& ahead_decrements) {
    // The QR book is aggregate, so infer public queue segments around our
    // fixed-size own-order overlay and sample where public cancels landed.
    ahead_decrements.fill(0);
    if (cancel_size <= 0 || level_volume <= 0) return;

    int own_total = own_remaining_at(env, side, price);
    int public_total = std::max(0, level_volume - own_total);
    int remaining_cancel = std::min(cancel_size, public_total);
    if (remaining_cancel <= 0) return;

    std::array<int, QR_MAX_ORDERS> slots{};
    int count = collect_orders_at(env, side, price, slots);
    if (count == 0) return;

    int remaining_public = public_total;
    int previous_end = 0;
    int prefix_cancel = 0;
    for (int i = 0; i < count; i++) {
        QRAgentOrder& order = env->orders[slots[i]];
        int order_ahead = clamp_int(order.ahead, 0, level_volume);
        int public_segment = std::max(0, order_ahead - previous_end);
        public_segment = std::min(public_segment, remaining_public);
        int segment_cancel = sample_public_cancel_segment(
            s->queue_rng, public_segment, remaining_public, remaining_cancel);

        prefix_cancel += segment_cancel;
        remaining_cancel -= segment_cancel;
        remaining_public -= public_segment;
        ahead_decrements[slots[i]] = std::min(order.ahead, prefix_cancel);
        previous_end = std::max(previous_end, order_ahead + order.remaining);
    }
}

static void apply_ahead_decrements(
        QueueReactive* env,
        const std::array<int, QR_MAX_ORDERS>& ahead_decrements) {
    for (int i = 0; i < QR_MAX_ORDERS; i++) {
        if (!env->orders[i].active || ahead_decrements[i] <= 0) continue;
        env->orders[i].ahead = std::max(0, env->orders[i].ahead - ahead_decrements[i]);
    }
}

static int filled_size(const std::vector<qr::Fill>& fills) {
    int total = 0;
    for (const auto& fill : fills) total += fill.size;
    return total;
}

static void update_inventory_from_aggressive_fills(
        QueueReactive* env, int side, const std::vector<qr::Fill>& fills) {
    for (const auto& fill : fills) {
        if (side == QR_SIDE_BUY) {
            env->inventory += fill.size;
            env->cash -= static_cast<double>(fill.price) * fill.size;
        } else {
            env->inventory -= fill.size;
            env->cash += static_cast<double>(fill.price) * fill.size;
        }
    }
}

static void maybe_add_residual_order(
        QueueReactive* env, int side, const qr::Order& order, int fill) {
    if (!order.partial || fill >= order.size) return;
    int slot = first_free_slot(env);
    if (slot < 0) return;
    int passive_agent_side = side == QR_SIDE_BUY ? QR_SIDE_BUY : QR_SIDE_SELL;
    env->orders[slot] = {
        1,
        passive_agent_side,
        order.price,
        order.size - fill,
        0,
        env->next_order_id++,
    };
    env->log.agent_orders += 1.0f;
}

static void cancel_worst_agent_order(QueueReactive* env) {
    // First-attempt agent cancel logic. This is deliberately shaky: for now a
    // cancel action only chooses a side and we pop the worst-positioned order.
    // Eventually cancellation should be an explicit agent-controlled target
    // over own order id/slot, not this env-side heuristic.
    QRSim* s = sim(env);
    qr::OrderBook& lob = *s->lob;
    int side = env->last_action_side;
    int slot = worst_order_slot(env, side);
    if (slot < 0) {
        env->last_action_rejected = 1;
        env->log.agent_rejected += 1.0f;
        return;
    }

    QRAgentOrder& own = env->orders[slot];
    int original_remaining = own.remaining;
    int level_volume = safe_volume_at(lob, passive_side(side), own.price);
    int cancel_size = std::min(own.remaining, level_volume);
    if (cancel_size <= 0) {
        env->last_action_rejected = 1;
        env->log.agent_rejected += 1.0f;
        own = {};
        return;
    }
    qr::Order cancel(qr::OrderType::Cancel, passive_side(side), own.price, cancel_size, env->time_ns);
    lob.process(cancel);
    if (cancel.rejected) {
        env->last_action_rejected = 1;
        env->log.agent_rejected += 1.0f;
        return;
    }

    reduce_orders_behind(env, own.side, own.price, own.id, cancel_size);
    env->last_action_price = own.price;
    env->last_action_partial = cancel.partial ? 1 : 0;
    own.remaining -= cancel_size;
    if (own.remaining <= 0 || cancel.partial || cancel_size < original_remaining) own = {};
    env->log.agent_cancels += 1.0f;
}

static void process_agent_intervention(QueueReactive* env) {
    QRSim* s = sim(env);
    qr::OrderBook& lob = *s->lob;
    int type = clamp_int(static_cast<int>(env->actions[0]), QR_ACTION_NOOP, QR_ACTION_IMPROVE);
    int side = static_cast<int>(env->actions[1]) == QR_SIDE_SELL ? QR_SIDE_SELL : QR_SIDE_BUY;
    int level_or_slot = clamp_int(static_cast<int>(env->actions[2]), 0, QR_ACTION_PRICE_CHOICES - 1);
    int size = action_size(s, env->actions[3]);
    env->last_action_type = type;
    env->last_action_side = side;
    env->last_action_rejected = 0;
    env->last_action_partial = 0;
    env->last_action_filled = 0;
    env->last_action_price = 0;
    env->last_action_lost_race = 0;

    if (type == QR_ACTION_NOOP) return;

    try {
        if (type == QR_ACTION_CANCEL) {
            // Price/slot action input is ignored for now. See
            // cancel_worst_agent_order for the temporary policy.
            (void)level_or_slot;
            cancel_worst_agent_order(env);
            return;
        }

        if (type == QR_ACTION_LIMIT || type == QR_ACTION_IMPROVE) {
            int slot = first_free_slot(env);
            if (slot < 0) {
                env->last_action_rejected = 1;
                env->log.agent_rejected += 1.0f;
                return;
            }
            qr::OrderType order_type = qr::OrderType::Add;
            int price = level_price(lob, side, level_or_slot);
            if (type == QR_ACTION_IMPROVE) {
                if (lob.spread() <= 1) {
                    env->last_action_rejected = 1;
                    env->log.agent_rejected += 1.0f;
                    return;
                }
                if (side == QR_SIDE_BUY) {
                    order_type = qr::OrderType::CreateBid;
                    price = lob.best_bid() + 1;
                } else {
                    order_type = qr::OrderType::CreateAsk;
                    price = lob.best_ask() - 1;
                }
            }
            int ahead = type == QR_ACTION_LIMIT ? safe_volume_at(lob, passive_side(side), price) : 0;
            qr::Order add(order_type, passive_side(side), price, size, env->time_ns);
            lob.process(add);
            if (add.rejected) {
                env->last_action_rejected = 1;
                env->log.agent_rejected += 1.0f;
                return;
            }
            env->orders[slot] = {1, side, price, size, ahead, env->next_order_id++};
            env->last_action_price = price;
            env->log.agent_orders += 1.0f;
            return;
        }

        if (type == QR_ACTION_MARKET) {
            int price = market_limit_price(lob, side, level_or_slot);
            qr::Order order(qr::OrderType::Trade, trade_side(side), price, size, env->time_ns);
            std::vector<qr::Fill> fills;
            lob.process(order, &fills);
            int fill = filled_size(fills);
            int resting_side = side == QR_SIDE_BUY ? QR_SIDE_SELL : QR_SIDE_BUY;
            for (const auto& f : fills) apply_passive_fills(env, resting_side, f.price, f.size);
            update_inventory_from_aggressive_fills(env, side, fills);
            env->last_action_filled = fill;
            env->last_action_price = fill > 0
                ? static_cast<int>(std::llround([&]() {
                    double notional = 0.0;
                    for (const auto& f : fills) notional += static_cast<double>(f.price) * f.size;
                    return notional / std::max(1, fill);
                }()))
                : price;
            env->last_action_partial = fill < size;
            if (fill == 0) {
                env->last_action_rejected = 1;
                env->log.agent_rejected += 1.0f;
            }
            maybe_add_residual_order(env, side, order, fill);
            if (s->cfg.strategy_impact && s->impact && fill > 0) {
                s->impact->add_trade(trade_side(side), fill);
            }
            if (fill > 0) env->log.agent_fills += 1.0f;
        }
    } catch (...) {
        env->last_action_rejected = 1;
        env->config_error = 1;
        env->log.agent_rejected += 1.0f;
    }
}

static void process_qr_order(QueueReactive* env, qr::Order order) {
    QRSim* s = sim(env);
    qr::OrderBook& lob = *s->lob;
    env->last_qr_type = static_cast<int>(order.type);
    env->last_qr_side = agent_side(order.side);
    env->last_qr_size = order.size;
    env->last_qr_price = order.price;
    env->last_qr_rejected = 0;
    env->last_qr_partial = 0;
    try {
        if (order.type == qr::OrderType::Cancel) {
            int side = agent_side(order.side);
            int level_volume = safe_volume_at(lob, order.side, order.price);
            int own = own_remaining_at(env, side, order.price);
            int public_volume = std::max(0, level_volume - own);
            order.size = std::min(order.size, public_volume);
            if (order.size <= 0) {
                env->last_qr_rejected = 1;
                return;
            }
            std::array<int, QR_MAX_ORDERS> ahead_decrements{};
            sample_public_cancel_decrements(
                env, s, side, order.price, level_volume, order.size, ahead_decrements);
            lob.process(order);
            apply_ahead_decrements(env, ahead_decrements);
        } else if (order.type == qr::OrderType::Trade) {
            std::vector<qr::Fill> fills;
            lob.process(order, &fills);
            int fill = filled_size(fills);
            for (const auto& f : fills) {
                apply_passive_fills(env, agent_side(order.side), f.price, f.size);
            }
            if (s->impact && fill > 0) s->impact->add_trade(order.side, fill);
            env->last_qr_partial = fill < order.size;
            env->log.qr_trades += fill > 0 ? 1.0f : 0.0f;
        } else {
            lob.process(order);
        }
        env->last_qr_rejected = order.rejected ? 1 : 0;
        env->last_qr_partial = order.partial ? 1 : env->last_qr_partial;
        env->log.qr_events += 1.0f;
    } catch (...) {
        env->last_qr_rejected = 1;
        env->config_error = 1;
    }
}

static bool wants_intervention(QueueReactive* env) {
    int type = static_cast<int>(env->actions[0]);
    return type >= QR_ACTION_LIMIT && type <= QR_ACTION_IMPROVE;
}

static void record_submitted_action(QueueReactive* env) {
    int type = clamp_int(static_cast<int>(env->actions[0]), QR_ACTION_NOOP, QR_ACTION_IMPROVE);
    int side = static_cast<int>(env->actions[1]) == QR_SIDE_SELL ? QR_SIDE_SELL : QR_SIDE_BUY;
    env->last_action_type = type;
    env->last_action_side = side;
    env->last_action_rejected = 0;
    env->last_action_partial = 0;
    env->last_action_filled = 0;
    env->last_action_price = 0;
    env->last_action_lost_race = 0;
}

static void advance_models(QueueReactive* env, int64_t dt) {
    QRSim* s = sim(env);
    env->time_ns += dt;
    if (s->alpha) s->alpha->step(dt);
    if (s->impact) s->impact->step(env->time_ns);
}

static float compute_reward(QueueReactive* env) {
    (void)env;
    return 0.0f;
}

static float compute_terminal(QueueReactive* env) {
    (void)env;
    return 0.0f;
}

static void compute_observations(QueueReactive* env) {
    QRSim* s = sim(env);
    qr::OrderBook& lob = *s->lob;
    float mid = 0.5f * static_cast<float>(lob.best_bid() + lob.best_ask());
    int idx = 0;
    for (int level = 0; level < QR_LEVELS; level++) {
        int bid = lob.best_bid() - level;
        int ask = lob.best_ask() + level;
        env->observations[idx++] = (static_cast<float>(bid) - mid) / PRICE_NORM;
        env->observations[idx++] = static_cast<float>(safe_volume_at(lob, qr::Side::Bid, bid)) / VOLUME_NORM;
        env->observations[idx++] = (static_cast<float>(ask) - mid) / PRICE_NORM;
        env->observations[idx++] = static_cast<float>(safe_volume_at(lob, qr::Side::Ask, ask)) / VOLUME_NORM;
    }

    double pnl = env->cash + static_cast<double>(env->inventory) * mid;
    int active_orders = 0;
    for (const auto& order : env->orders) active_orders += order.active ? 1 : 0;
    env->last_alpha = s->alpha ? s->alpha->value() : 0.0;
    env->last_impact_bias = s->impact ? s->impact->bias_factor() : 0.0;

    env->observations[idx++] = static_cast<float>(lob.spread()) / PRICE_NORM;
    env->observations[idx++] = static_cast<float>(lob.imbalance());
    env->observations[idx++] = static_cast<float>(env->inventory) / INVENTORY_NORM;
    env->observations[idx++] = static_cast<float>(env->cash / CASH_NORM);
    env->observations[idx++] = static_cast<float>(pnl / CASH_NORM);
    env->observations[idx++] = static_cast<float>(env->last_alpha);
    env->observations[idx++] = static_cast<float>(env->last_impact_bias);
    env->observations[idx++] = static_cast<float>(env->last_qr_dt / 1e9);
    env->observations[idx++] = static_cast<float>(env->last_latency_dt / 1e9);
    env->observations[idx++] = static_cast<float>((env->time_ns % 1000000000LL) / 1e9);
    env->observations[idx++] = static_cast<float>(env->last_action_filled) / VOLUME_NORM;
    env->observations[idx++] = static_cast<float>(active_orders) / static_cast<float>(QR_MAX_ORDERS);

    env->observations[idx++] = static_cast<float>(env->last_action_type) / static_cast<float>(QR_NUM_ACTION_TYPES - 1);
    env->observations[idx++] = env->last_action_side == QR_SIDE_BUY ? 1.0f : -1.0f;
    env->observations[idx++] = static_cast<float>(env->last_action_rejected);
    env->observations[idx++] = static_cast<float>(env->last_action_partial);
    env->observations[idx++] = static_cast<float>(env->last_action_lost_race);
    env->observations[idx++] = env->last_action_price == 0
        ? 0.0f
        : (static_cast<float>(env->last_action_price) - mid) / PRICE_NORM;
    env->observations[idx++] = static_cast<float>(env->last_qr_type) / 4.0f;
    env->observations[idx++] = env->last_qr_side == QR_SIDE_BUY ? 1.0f : -1.0f;
    env->observations[idx++] = static_cast<float>(env->last_qr_size) / VOLUME_NORM;
    env->observations[idx++] = env->last_qr_price == 0
        ? 0.0f
        : (static_cast<float>(env->last_qr_price) - mid) / PRICE_NORM;
    env->observations[idx++] = static_cast<float>(env->last_qr_rejected);
    env->observations[idx++] = static_cast<float>(env->last_qr_partial);

    for (int i = 0; i < QR_MAX_ORDERS; i++) {
        const QRAgentOrder& order = env->orders[i];
        env->observations[idx++] = static_cast<float>(order.active);
        env->observations[idx++] = order.side == QR_SIDE_BUY ? 1.0f : -1.0f;
        env->observations[idx++] = (static_cast<float>(order.price) - mid) / PRICE_NORM;
        env->observations[idx++] = static_cast<float>(order.remaining) / VOLUME_NORM;
        env->observations[idx++] = static_cast<float>(order.ahead) / VOLUME_NORM;
    }
}

static const char* qr_event_name(int type) {
    switch (type) {
    case 0: return "Add";
    case 1: return "Cancel";
    case 2: return "Trade";
    case 3: return "CreateBid";
    case 4: return "CreateAsk";
    default: return "None";
    }
}

static const char* agent_action_name(int type) {
    switch (type) {
    case QR_ACTION_NOOP: return "Noop";
    case QR_ACTION_LIMIT: return "Limit";
    case QR_ACTION_CANCEL: return "Cancel";
    case QR_ACTION_MARKET: return "Market";
    case QR_ACTION_IMPROVE: return "Improve";
    default: return "Unknown";
    }
}

static const char* side_name(int side) {
    return side == QR_SIDE_SELL ? "Ask" : "Bid";
}

static int active_order_count(const QueueReactive* env) {
    int count = 0;
    for (int i = 0; i < QR_MAX_ORDERS; i++) {
        count += env->orders[i].active ? 1 : 0;
    }
    return count;
}

static void ensure_render_window() {
    if (IsWindowReady()) return;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(980, 720, "queue_reactive order book");
    SetWindowMinSize(760, 580);
    SetTargetFPS(60);
}

static void draw_panel(Rectangle rect, Color fill, Color border) {
    DrawRectangleRounded(rect, 0.04f, 8, fill);
    DrawRectangleRoundedLines(rect, 0.04f, 8, border);
}

static void draw_book_level(
        int x, int y, int width, int height, int center_x,
        int price, int volume, int max_volume, bool bid) {
    const Color row_bg = bid ? (Color){19, 42, 44, 255} : (Color){46, 30, 34, 255};
    const Color bar = bid ? (Color){55, 168, 134, 220} : (Color){213, 91, 98, 220};
    const Color price_color = bid ? (Color){112, 224, 185, 255} : (Color){255, 143, 150, 255};
    const Color text_color = (Color){229, 233, 239, 255};
    const Color muted = (Color){142, 154, 170, 255};

    DrawRectangleRounded((Rectangle){(float)x, (float)y, (float)width, (float)height}, 0.06f, 8, row_bg);

    int bar_max = std::max(80, width / 3);
    int bar_width = max_volume > 0
        ? std::max(2, (int)std::round((double)volume / (double)max_volume * (double)bar_max))
        : 2;
    int bar_y = y + 8;
    int bar_h = height - 16;
    if (bid) {
        DrawRectangle(center_x - 116 - bar_width, bar_y, bar_width, bar_h, bar);
        DrawText(TextFormat("%d", volume), center_x - 108 - bar_width, y + 14, 18, text_color);
    } else {
        DrawRectangle(center_x + 116, bar_y, bar_width, bar_h, bar);
        DrawText(TextFormat("%d", volume), center_x + 124 + bar_width, y + 14, 18, text_color);
    }

    DrawText(TextFormat("%d", price), center_x - 38, y + 14, 20, price_color);
    DrawText(bid ? "BID" : "ASK", bid ? x + 18 : x + width - 52, y + 15, 16, muted);
}

static void draw_queue_reactive_render(QueueReactive* env) {
    QRSim* s = sim(env);
    qr::OrderBook& lob = *s->lob;

    int width = GetScreenWidth();
    int height = GetScreenHeight();
    int margin = 28;
    int ladder_x = margin;
    int ladder_y = 86;
    int ladder_w = width - 2 * margin;
    int row_h = std::max(38, (height - 260) / (QR_LEVELS * 2 + 1));
    int center_x = ladder_x + ladder_w / 2;
    int best_bid = lob.best_bid();
    int best_ask = lob.best_ask();
    double mid = 0.5 * (double)(best_bid + best_ask);

    int max_volume = 1;
    for (int level = 0; level < QR_LEVELS; level++) {
        max_volume = std::max(max_volume, safe_volume_at(lob, qr::Side::Bid, best_bid - level));
        max_volume = std::max(max_volume, safe_volume_at(lob, qr::Side::Ask, best_ask + level));
    }

    const Color bg = (Color){9, 14, 18, 255};
    const Color panel = (Color){16, 23, 30, 255};
    const Color border = (Color){51, 63, 78, 255};
    const Color text = (Color){232, 236, 242, 255};
    const Color muted = (Color){145, 157, 172, 255};
    const Color accent = (Color){243, 190, 87, 255};
    const Color warning = env->config_error ? (Color){255, 143, 150, 255} : (Color){112, 224, 185, 255};

    BeginDrawing();
    ClearBackground(bg);

    DrawText("queue_reactive", margin, 24, 28, text);
    DrawText(TextFormat(
        "step %d   t %.6fs   mid %.1f   spread %d   imbalance %.3f",
        env->step,
        (double)env->time_ns / 1e9,
        mid,
        lob.spread(),
        lob.imbalance()
    ), margin, 56, 18, muted);
    DrawText(env->calibration_loaded ? "calibrated" : "static params", width - 190, 28, 18, warning);

    DrawLine(center_x, ladder_y - 12, center_x, ladder_y + row_h * (QR_LEVELS * 2 + 1) + 12, border);
    DrawText("VOLUME", center_x - 230, ladder_y - 28, 16, muted);
    DrawText("PRICE", center_x - 31, ladder_y - 28, 16, muted);
    DrawText("VOLUME", center_x + 170, ladder_y - 28, 16, muted);

    int row = 0;
    for (int level = QR_LEVELS - 1; level >= 0; level--) {
        int price = best_ask + level;
        int volume = safe_volume_at(lob, qr::Side::Ask, price);
        draw_book_level(ladder_x, ladder_y + row * row_h, ladder_w, row_h - 4,
            center_x, price, volume, max_volume, false);
        row++;
    }

    int mid_y = ladder_y + row * row_h;
    DrawRectangle(ladder_x, mid_y + row_h / 2 - 1, ladder_w, 2, border);
    DrawText(TextFormat("mid %.1f", mid), center_x - 42, mid_y + row_h / 2 + 8, 16, accent);
    row++;

    for (int level = 0; level < QR_LEVELS; level++) {
        int price = best_bid - level;
        int volume = safe_volume_at(lob, qr::Side::Bid, price);
        draw_book_level(ladder_x, ladder_y + row * row_h, ladder_w, row_h - 4,
            center_x, price, volume, max_volume, true);
        row++;
    }

    int panel_y = ladder_y + row * row_h + 22;
    int panel_h = std::max(96, height - panel_y - margin);
    int panel_w = (width - 3 * margin) / 2;
    Rectangle left = {(float)margin, (float)panel_y, (float)panel_w, (float)panel_h};
    Rectangle right = {(float)(2 * margin + panel_w), (float)panel_y, (float)panel_w, (float)panel_h};
    draw_panel(left, panel, border);
    draw_panel(right, panel, border);

    DrawText("last market event", (int)left.x + 18, (int)left.y + 14, 19, text);
    DrawText(TextFormat("%s %s   price %d   size %d",
        qr_event_name(env->last_qr_type),
        side_name(env->last_qr_side),
        env->last_qr_price,
        env->last_qr_size
    ), (int)left.x + 18, (int)left.y + 42, 18, muted);
    DrawText(TextFormat("dt %.6fs   rejected %d   partial %d",
        (double)env->last_qr_dt / 1e9,
        env->last_qr_rejected,
        env->last_qr_partial
    ), (int)left.x + 18, (int)left.y + 68, 18, muted);

    DrawText("agent state", (int)right.x + 18, (int)right.y + 14, 19, text);
    DrawText(TextFormat("%s %s   price %d   filled %d",
        agent_action_name(env->last_action_type),
        side_name(env->last_action_side),
        env->last_action_price,
        env->last_action_filled
    ), (int)right.x + 18, (int)right.y + 42, 18, muted);
    DrawText(TextFormat("inventory %d   cash %.0f   active orders %d",
        env->inventory,
        env->cash,
        active_order_count(env)
    ), (int)right.x + 18, (int)right.y + 68, 18, muted);

    EndDrawing();
}

} // namespace

extern "C" {

void qr_config_defaults(QRConfig* config) {
    *config = {};
    config->params_path = "";
    config->latency_path = "";
    config->use_calibrated_params = 0;
    config->use_mixture_delta_t = 0;
    config->use_total_lvl = 0;
    config->use_power_law_impact = 0;
    config->use_alpha = 0;
    config->strategy_impact = 1;
    config->lot_size = 1;
    config->report_interval = 1024;
    config->max_events_per_step = 1;
    config->initial_bid = 1519;
    config->initial_ask = 1520;
    config->qr_dt_mean_ns = 100000000.0;
    config->event_add_prob = 0.45;
    config->event_cancel_prob = 0.35;
    config->event_trade_prob = 0.20;
    config->event_create_prob = 0.30;
    config->latency_mu = DEFAULT_LATENCY_MU;
    config->latency_sigma = DEFAULT_LATENCY_SIGMA;
    config->latency_lower = DEFAULT_LATENCY_LOWER;
    config->latency_upper = DEFAULT_LATENCY_UPPER;
    config->alpha_kappa = 0.5;
    config->alpha_sigma = 0.5;
    config->alpha_scale = 1.0;
    config->impact_beta = 1.5;
    config->impact_tau = 50.0;
    config->impact_m = 4.0;
    config->impact_components = 20;
    const int sizes[QR_SIZE_BUCKETS] = {100, 200, 400, 800, 1600, 3200};
    const int mes[QR_LEVELS] = {400, 300, 200, 100};
    const int bid_volumes[QR_LEVELS] = {2000, 3000, 200, 400};
    const int ask_volumes[QR_LEVELS] = {1200, 5100, 4400, 2300};
    for (int i = 0; i < QR_SIZE_BUCKETS; i++) config->size_buckets[i] = sizes[i];
    for (int i = 0; i < QR_LEVELS; i++) {
        config->mes[i] = mes[i];
        config->bid_volumes[i] = bid_volumes[i];
        config->ask_volumes[i] = ask_volumes[i];
    }
}

void qr_configure(QueueReactive* env, const QRConfig* config) {
    c_close(env);
    env->sim = nullptr;
    env->configured = 1;
    env->config_error = 0;
    try {
        env->sim = new QRSim(*config, env->rng ? env->rng : 1);
        env->calibration_loaded = sim(env)->calibrated_loaded ? 1 : 0;
    } catch (...) {
        QRConfig fallback;
        qr_config_defaults(&fallback);
        env->sim = new QRSim(fallback, env->rng ? env->rng : 1);
        env->calibration_loaded = 0;
        env->config_error = 1;
    }
}

void c_reset(QueueReactive* env) {
    if (env->sim == nullptr) {
        QRConfig config;
        qr_config_defaults(&config);
        qr_configure(env, &config);
    }
    QRSim* s = sim(env);
    s->reset_book();
    for (auto& order : env->orders) order = {};
    env->step = 0;
    env->next_order_id = 1;
    env->inventory = 0;
    env->cash = 0.0;
    env->time_ns = 0;
    env->last_qr_dt = 0;
    env->last_latency_dt = 0;
    env->last_action_type = QR_ACTION_NOOP;
    env->last_action_side = QR_SIDE_BUY;
    env->last_action_rejected = 0;
    env->last_action_partial = 0;
    env->last_action_filled = 0;
    env->last_action_price = s->cfg.initial_bid;
    env->last_action_lost_race = 0;
    env->last_qr_type = QR_ACTION_NOOP;
    env->last_qr_side = QR_SIDE_BUY;
    env->last_qr_size = 0;
    env->last_qr_price = s->cfg.initial_bid;
    env->last_qr_rejected = 0;
    env->last_qr_partial = 0;
    compute_observations(env);
}

void c_step(QueueReactive* env) {
    QRSim* s = sim(env);
    env->step++;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    record_submitted_action(env);

    double alpha_val = s->alpha ? s->alpha->value() : 0.0;
    double impact_val = s->impact ? s->impact->bias_factor() : 0.0;
    double alpha_scale = s->alpha ? s->alpha->scale() : 1.0;
    double total_bias = -alpha_scale * alpha_val + impact_val;
    s->model->bias(total_bias);

    qr::Order qr_order = s->model->sample_order(env->time_ns);
    int64_t dt_qr = s->model->sample_dt(s->model->last_event());
    int64_t dt_agent = wants_intervention(env)
        ? s->latency.sample(s->latency_rng)
        : std::numeric_limits<int64_t>::max();
    env->last_qr_dt = dt_qr;
    env->last_latency_dt = dt_agent == std::numeric_limits<int64_t>::max() ? 0 : dt_agent;

    if (wants_intervention(env) && dt_agent < dt_qr) {
        advance_models(env, dt_agent);
        process_agent_intervention(env);
        advance_models(env, dt_qr - dt_agent);
        qr_order.ts = env->time_ns;
        process_qr_order(env, qr_order);
    } else {
        if (wants_intervention(env)) {
            env->last_action_lost_race = 1;
            env->log.agent_lost_race += 1.0f;
        }
        advance_models(env, dt_qr);
        qr_order.ts = env->time_ns;
        process_qr_order(env, qr_order);
    }

    env->rewards[0] = compute_reward(env);
    env->terminals[0] = compute_terminal(env);
    compute_observations(env);

    env->log.episode_length += 1.0f;
    env->log.episode_return += env->rewards[0];
    env->log.inventory += std::fabs(static_cast<float>(env->inventory));
    env->log.score += env->rewards[0];
    env->log.perf += 0.0f;
    env->log.n += 1.0f;
}

void c_render(QueueReactive* env) {
    if (env == nullptr || env->sim == nullptr) return;
    ensure_render_window();
    if (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE)) {
        CloseWindow();
        std::exit(0);
    }
    draw_queue_reactive_render(env);
}

void c_close(QueueReactive* env) {
    if (env->sim != nullptr) {
        delete sim(env);
        env->sim = nullptr;
    }
}

} // extern "C"
