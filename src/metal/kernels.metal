#include <metal_stdlib>
using namespace metal;

kernel void fill_f32(
        device float* dst [[buffer(0)]],
        constant float& value [[buffer(1)]],
        constant uint& n [[buffer(2)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx < n) {
        dst[idx] = value;
    }
}

kernel void add_f32(
        device float* dst [[buffer(0)]],
        device const float* src [[buffer(1)]],
        constant uint& n [[buffer(2)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

kernel void scale_f32(
        device float* dst [[buffer(0)]],
        constant float& scale [[buffer(1)]],
        constant uint& n [[buffer(2)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx < n) {
        dst[idx] *= scale;
    }
}

struct ClampParams {
    float lower;
    float upper;
    uint n;
};

kernel void clamp_f32(
        device float* dst [[buffer(0)]],
        constant ClampParams& params [[buffer(1)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx < params.n) {
        // Match clamp_precision_kernel in kernels.cu.
        dst[idx] = fmin(fmax(dst[idx], params.lower), params.upper);
    }
}

kernel void cast_u8_f32(
        device float* dst [[buffer(0)]],
        device const uchar* src [[buffer(1)]],
        constant uint& n [[buffer(2)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx < n) {
        dst[idx] = float(src[idx]);
    }
}

struct Transpose102Params {
    uint a;
    uint b;
    uint c;
    uint total;
};

struct MinGRUGateParams {
    uint batch_size;
    uint hidden_size;
    uint total;
};

struct MinGRUScanParams {
    uint batch_size;
    uint horizon;
    uint hidden_size;
    uint total;
};

inline float puf_sigmoid(float x) {
    float z = exp(-fabs(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

inline float puf_fast_tanh(float x) {
    float v1 = fmin(fmax(x, -9.0f), 9.0f);
    float v2 = v1 * v1;
    float p = v2 * -2.76076847742355e-16f + 2.00018790482477e-13f;
    p = v2 * p + -8.60467152213735e-11f;
    p = v2 * p + 5.12229709037114e-08f;
    p = v2 * p + 1.48572235717979e-05f;
    p = v2 * p + 6.37261928875436e-04f;
    p = v2 * p + 4.89352455891786e-03f;
    p = v1 * p;
    float q = v2 * 1.19825839466702e-06f + 1.18534705686654e-04f;
    q = v2 * q + 2.26843463243900e-03f;
    q = v2 * q + 4.89352518554385e-03f;
    return p / q;
}

inline float puf_fast_sigmoid(float x) {
    return fmin(1.0f, fmax(0.0f, (puf_fast_tanh(x * 0.5f) + 1.0f) * 0.5f));
}

inline float puf_lerp(float a, float b, float w) {
    float diff = b - a;
    return fabs(w) < 0.5f ? a + w * diff : b - diff * (1.0f - w);
}

// MSL has no log1p intrinsic. The correction term recovers the low bits lost
// when 1+x is rounded before log, including the y==1 case for tiny x.
inline float puf_log1p(float x) {
    float y = 1.0f + x;
    return log(y) - ((y - 1.0f) - x) / y;
}

inline float puf_softplus(float x) {
    return x > 20.0f ? x : puf_log1p(exp(x));
}

inline float puf_logaddexp(float a, float b) {
    float m = fmax(a, b);
    float diff = fmin(a, b) - m;
    return diff < -88.0f ? m : m + puf_log1p(exp(diff));
}

inline void puf_log_coeffs_and_values(float gate, float hidden,
        thread float& log_coeff, thread float& log_value) {
    float abs_gate = fabs(gate);
    float sp_neg = puf_log1p(exp(-abs_gate));
    float softplus_gate = gate >= 0.0f ? gate + sp_neg : sp_neg;
    float softplus_neg_gate = gate >= 0.0f ? sp_neg : -gate + sp_neg;
    log_coeff = -softplus_gate;
    float log_tilde_h = hidden >= 0.0f
        ? log(hidden + 0.5f) : -puf_softplus(-hidden);
    log_value = -softplus_neg_gate + log_tilde_h;
}

kernel void mingru_gate_f32(
        device float* output [[buffer(0)]],
        device float* next_state [[buffer(1)]],
        device const float* combined [[buffer(2)]],
        device const float* state [[buffer(3)]],
        device const float* input [[buffer(4)]],
        constant MinGRUGateParams& params [[buffer(5)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= params.total) return;
    uint b = idx / params.hidden_size;
    uint h = idx % params.hidden_size;
    uint combined_base = b * 3 * params.hidden_size;
    float hidden = combined[combined_base + h];
    float gate = combined[combined_base + params.hidden_size + h];
    float projection = combined[combined_base + 2 * params.hidden_size + h];
    float hidden_tilde = hidden >= 0.0f
        ? hidden + 0.5f : puf_fast_sigmoid(hidden);
    float mingru_output = puf_lerp(
        state[idx], hidden_tilde, puf_sigmoid(gate));
    next_state[idx] = mingru_output;
    float projection_gate = puf_sigmoid(projection);
    output[idx] = projection_gate * mingru_output
        + (1.0f - projection_gate) * input[idx];
}

kernel void mingru_scan_forward_f32(
        device float* output [[buffer(0)]],
        device float* next_state [[buffer(1)]],
        device float* a_star_buffer [[buffer(2)]],
        device float* s_buffer [[buffer(3)]],
        device float* log_values_buffer [[buffer(4)]],
        device const float* combined [[buffer(5)]],
        device const float* state [[buffer(6)]],
        device const float* input [[buffer(7)]],
        constant MinGRUScanParams& params [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= params.total) return;
    uint b = idx / params.hidden_size;
    uint h = idx % params.hidden_size;
    uint batch_hidden = b * params.hidden_size;
    uint hidden3 = 3 * params.hidden_size;
    uint batch_sequence = batch_hidden * params.horizon;
    uint output_base = batch_sequence + h;
    uint combined_base = 3 * batch_sequence;

    float a_star = 0.0f;
    float log_value = log(state[batch_hidden + h]);
    float s = log_value;
    uint checkpoint_index = b * (params.horizon + 1) * params.hidden_size + h;
    a_star_buffer[checkpoint_index] = a_star;
    s_buffer[checkpoint_index] = s;
    log_values_buffer[checkpoint_index] = log_value;

    uint hidden_base = combined_base + h;
    uint gate_base = combined_base + params.hidden_size + h;
    uint projection_base = combined_base + 2 * params.hidden_size + h;
    float scan_result = 0.0f;
    uint output_index = output_base;
    uint time_offset = 0;
    for (uint t = 1; t <= params.horizon; ++t) {
        float hidden = combined[hidden_base + time_offset];
        float gate = combined[gate_base + time_offset];
        float projection = combined[projection_base + time_offset];
        float x = input[output_base + (t - 1) * params.hidden_size];
        float log_coeff;
        puf_log_coeffs_and_values(gate, hidden, log_coeff, log_value);
        a_star += log_coeff;
        s = puf_logaddexp(s, log_value - a_star);
        scan_result = exp(a_star + s);
        float projection_gate = puf_sigmoid(projection);
        output[output_index] = projection_gate * scan_result
            + (1.0f - projection_gate) * x;

        checkpoint_index += params.hidden_size;
        output_index += params.hidden_size;
        time_offset += hidden3;
        if (t % 4 == 0) {
            a_star_buffer[checkpoint_index] = a_star;
            s_buffer[checkpoint_index] = s;
            log_values_buffer[checkpoint_index] = log_value;
        }
    }
    next_state[batch_hidden + h] = scan_result;
}

kernel void mingru_scan_backward_f32(
        device float* grad_combined [[buffer(0)]],
        device float* grad_state [[buffer(1)]],
        device float* grad_input [[buffer(2)]],
        device const float* combined [[buffer(3)]],
        device const float* a_star_buffer [[buffer(4)]],
        device const float* s_buffer [[buffer(5)]],
        device const float* input [[buffer(6)]],
        device const float* grad_output [[buffer(7)]],
        device const float* grad_next_state [[buffer(8)]],
        constant MinGRUScanParams& params [[buffer(9)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= params.total) return;
    uint b = idx / params.hidden_size;
    uint h = idx % params.hidden_size;
    uint state_index = b * params.hidden_size + h;
    uint sequence_base = b * params.horizon * params.hidden_size + h;
    uint combined_batch_base = 3 * b * params.horizon * params.hidden_size;
    uint checkpoint_base = b * (params.horizon + 1) * params.hidden_size + h;
    uint hidden3 = 3 * params.hidden_size;

    float carry = grad_next_state[state_index];
    int chunk_end = int(params.horizon);
    while (chunk_end > 0) {
        // Checkpoints are anchored at 0,4,8,...; this also handles horizons
        // that are not a multiple of four.
        int chunk_start = ((chunk_end - 1) / 4) * 4;
        int chunk_length = chunk_end - chunk_start;
        uint checkpoint_index = checkpoint_base
            + uint(chunk_start) * params.hidden_size;
        float previous = exp(
            a_star_buffer[checkpoint_index] + s_buffer[checkpoint_index]);
        float previous_states[4];
        float recurrent_states[4];

        for (int i = 0; i < chunk_length; ++i) {
            uint t = uint(chunk_start + i);
            uint combined_offset = combined_batch_base + t * hidden3;
            float hidden = combined[combined_offset + h];
            float gate = combined[combined_offset + params.hidden_size + h];
            float hidden_tilde = hidden >= 0.0f
                ? hidden + 0.5f : puf_sigmoid(hidden);
            previous_states[i] = previous;
            previous = puf_lerp(previous, hidden_tilde, puf_sigmoid(gate));
            recurrent_states[i] = previous;
        }

        for (int i = chunk_length - 1; i >= 0; --i) {
            uint t = uint(chunk_start + i);
            uint combined_offset = combined_batch_base + t * hidden3;
            uint sequence_index = sequence_base + t * params.hidden_size;
            float hidden = combined[combined_offset + h];
            float gate = combined[combined_offset + params.hidden_size + h];
            float projection = combined[
                combined_offset + 2 * params.hidden_size + h];
            float x = input[sequence_index];
            float grad_out = grad_output[sequence_index];
            float projection_gate = puf_sigmoid(projection);
            float recurrent = recurrent_states[i];

            float grad_recurrent = carry + grad_out * projection_gate;
            float grad_projection = grad_out * (recurrent - x)
                * projection_gate * (1.0f - projection_gate);
            grad_input[sequence_index] = grad_out * (1.0f - projection_gate);

            float gate_value = puf_sigmoid(gate);
            float hidden_tilde = hidden >= 0.0f
                ? hidden + 0.5f : puf_sigmoid(hidden);
            float hidden_derivative = hidden >= 0.0f
                ? 1.0f : hidden_tilde * (1.0f - hidden_tilde);
            float grad_hidden = grad_recurrent * gate_value * hidden_derivative;
            float grad_gate = grad_recurrent
                * (hidden_tilde - previous_states[i])
                * gate_value * (1.0f - gate_value);
            carry = grad_recurrent * (1.0f - gate_value);

            grad_combined[combined_offset + h] = grad_hidden;
            grad_combined[combined_offset + params.hidden_size + h] = grad_gate;
            grad_combined[combined_offset + 2 * params.hidden_size + h] =
                grad_projection;
        }
        chunk_end = chunk_start;
    }
    // Unlike the CUDA log-space quotient, this remains finite for a zero
    // initial state because carry is the direct recurrence derivative.
    grad_state[state_index] = carry;
}

// Transpose [A, B, C] -> [B, A, C], matching transpose_102 in kernels.cu.
kernel void transpose_102_f32(
        device float* dst [[buffer(0)]],
        device const float* src [[buffer(1)]],
        constant Transpose102Params& params [[buffer(2)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= params.total) {
        return;
    }
    uint a = idx / (params.b * params.c);
    uint rem = idx % (params.b * params.c);
    uint b = rem / params.c;
    uint c = rem % params.c;
    dst[b * params.a * params.c + a * params.c + c] = src[idx];
}
