#ifndef PUFFERLIB_METAL_BACKEND_H
#define PUFFERLIB_METAL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PufMetalContext PufMetalContext;
typedef struct PufMetalBuffer PufMetalBuffer;
typedef struct PufMetalCommandBatch PufMetalCommandBatch;

typedef enum PufMetalStatus {
    PUF_METAL_OK = 0,
    PUF_METAL_INVALID_ARGUMENT = 1,
    PUF_METAL_UNAVAILABLE = 2,
    PUF_METAL_IO_ERROR = 3,
    PUF_METAL_COMPILE_ERROR = 4,
    PUF_METAL_PIPELINE_ERROR = 5,
    PUF_METAL_EXECUTION_ERROR = 6,
} PufMetalStatus;

// library_path may point to Metal source (.metal) or a compiled .metallib.
// Source loading is intended for development and machines without the full
// Xcode command-line Metal compiler.
PufMetalContext* puf_metal_create(
    const char* library_path, char* error, size_t error_capacity);
void puf_metal_destroy(PufMetalContext* context);

int puf_metal_is_available(void);
const char* puf_metal_device_name(const PufMetalContext* context);
const char* puf_metal_last_error(const PufMetalContext* context);

// Shared buffers are directly visible to both the CPU and Apple GPU. The CPU
// must only access contents after a submitted GPU operation has completed.
PufMetalBuffer* puf_metal_buffer_create_shared(
    PufMetalContext* context, size_t size_bytes);
void puf_metal_buffer_destroy(PufMetalBuffer* buffer);
void* puf_metal_buffer_contents(PufMetalBuffer* buffer);
size_t puf_metal_buffer_size(const PufMetalBuffer* buffer);

int puf_metal_fill_f32(
    PufMetalContext* context, PufMetalBuffer* dst, float value, uint32_t n);
int puf_metal_add_f32(
    PufMetalContext* context, PufMetalBuffer* dst,
    const PufMetalBuffer* src, uint32_t n);
int puf_metal_clamp_f32(
    PufMetalContext* context, PufMetalBuffer* dst,
    float lower, float upper, uint32_t n);
int puf_metal_cast_u8_f32(
    PufMetalContext* context, PufMetalBuffer* dst,
    const PufMetalBuffer* src, uint32_t n);
int puf_metal_transpose_102_f32(
    PufMetalContext* context, PufMetalBuffer* dst,
    const PufMetalBuffer* src, uint32_t a, uint32_t b, uint32_t c);

// Dense, tightly packed, row-major GEMM:
//   dst[M,N] = alpha * op(lhs) @ op(rhs) + beta * dst[M,N]
// lhs is stored as [M,K] or [K,M], and rhs as [K,N] or [N,K], according
// to the transpose flags. The three buffers may not alias one another.
int puf_metal_gemm_f32(
    PufMetalContext* context, PufMetalBuffer* dst,
    const PufMetalBuffer* lhs, const PufMetalBuffer* rhs,
    uint32_t m, uint32_t n, uint32_t k,
    int transpose_lhs, int transpose_rhs, float alpha, float beta);

// A command batch records multiple GPU operations before one asynchronous
// commit. Buffers referenced by a batch must remain alive until wait succeeds.
PufMetalCommandBatch* puf_metal_command_batch_create(PufMetalContext* context);
void puf_metal_command_batch_destroy(PufMetalCommandBatch* batch);
int puf_metal_command_batch_gemm_f32(
    PufMetalCommandBatch* batch, PufMetalBuffer* dst,
    const PufMetalBuffer* lhs, const PufMetalBuffer* rhs,
    uint32_t m, uint32_t n, uint32_t k,
    int transpose_lhs, int transpose_rhs, float alpha, float beta);

// Bias-free dense layer matching the encoder/decoder matrix layout in
// models.cu: input[B,I], weight[O,I], output[B,O]. Backward writes both
// grad_input[B,I] and grad_weight[O,I].
int puf_metal_command_batch_linear_forward_f32(
    PufMetalCommandBatch* batch, PufMetalBuffer* output,
    const PufMetalBuffer* input, const PufMetalBuffer* weight,
    uint32_t batch_size, uint32_t input_size, uint32_t output_size);
int puf_metal_command_batch_linear_backward_f32(
    PufMetalCommandBatch* batch,
    PufMetalBuffer* grad_input, PufMetalBuffer* grad_weight,
    const PufMetalBuffer* grad_output, const PufMetalBuffer* input,
    const PufMetalBuffer* weight,
    uint32_t batch_size, uint32_t input_size, uint32_t output_size);

// MinGRU kernels matching models.cu. `combined` stores the projected
// hidden/gate/highway values in [B,3H] (rollout) or [B,T,3H] (training).
int puf_metal_command_batch_mingru_gate_f32(
    PufMetalCommandBatch* batch,
    PufMetalBuffer* output, PufMetalBuffer* next_state,
    const PufMetalBuffer* combined, const PufMetalBuffer* state,
    const PufMetalBuffer* input, uint32_t batch_size, uint32_t hidden_size);
int puf_metal_command_batch_mingru_scan_forward_f32(
    PufMetalCommandBatch* batch,
    PufMetalBuffer* output, PufMetalBuffer* next_state,
    PufMetalBuffer* a_star, PufMetalBuffer* s_values,
    PufMetalBuffer* log_values,
    const PufMetalBuffer* combined, const PufMetalBuffer* state,
    const PufMetalBuffer* input,
    uint32_t batch_size, uint32_t horizon, uint32_t hidden_size);
int puf_metal_command_batch_mingru_scan_backward_f32(
    PufMetalCommandBatch* batch,
    PufMetalBuffer* grad_combined, PufMetalBuffer* grad_state,
    PufMetalBuffer* grad_input,
    const PufMetalBuffer* combined,
    const PufMetalBuffer* a_star, const PufMetalBuffer* s_values,
    const PufMetalBuffer* input, const PufMetalBuffer* grad_output,
    const PufMetalBuffer* grad_next_state,
    uint32_t batch_size, uint32_t horizon, uint32_t hidden_size);
int puf_metal_command_batch_commit(PufMetalCommandBatch* batch);
int puf_metal_command_batch_wait(PufMetalCommandBatch* batch);

#ifdef __cplusplus
}
#endif

#endif
