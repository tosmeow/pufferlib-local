#!/usr/bin/env python3
"""Benchmark the synchronous native Metal GEMM against NumPy on CPU."""

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / 'build' / 'metal-bench' / 'libpuffer_metal.dylib'
KERNELS = ROOT / 'src' / 'metal' / 'kernels.metal'

# (label, M, N, K, transpose_lhs, transpose_rhs)
CASES = (
    ('rollout_projection', 4096, 384, 128, 0, 1),
    ('minibatch_projection', 8192, 384, 128, 0, 1),
    ('weight_gradient', 384, 128, 8192, 1, 0),
    ('input_gradient', 8192, 128, 384, 0, 0),
)


class Buffer:
    def __init__(self, lib, context, elements):
        self.lib = lib
        self.elements = elements
        self.handle = lib.puf_metal_buffer_create_shared(
            context, elements * np.dtype(np.float32).itemsize)
        if not self.handle:
            raise RuntimeError('Metal buffer allocation failed')

    def array(self):
        address = self.lib.puf_metal_buffer_contents(self.handle)
        raw = (ctypes.c_float * self.elements).from_address(address)
        return np.ctypeslib.as_array(raw)

    def close(self):
        if self.handle:
            self.lib.puf_metal_buffer_destroy(self.handle)
            self.handle = None


def configure_api(lib):
    lib.puf_metal_create.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.puf_metal_create.restype = ctypes.c_void_p
    lib.puf_metal_destroy.argtypes = [ctypes.c_void_p]
    lib.puf_metal_device_name.argtypes = [ctypes.c_void_p]
    lib.puf_metal_device_name.restype = ctypes.c_char_p
    lib.puf_metal_last_error.argtypes = [ctypes.c_void_p]
    lib.puf_metal_last_error.restype = ctypes.c_char_p
    lib.puf_metal_buffer_create_shared.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t]
    lib.puf_metal_buffer_create_shared.restype = ctypes.c_void_p
    lib.puf_metal_buffer_destroy.argtypes = [ctypes.c_void_p]
    lib.puf_metal_buffer_contents.argtypes = [ctypes.c_void_p]
    lib.puf_metal_buffer_contents.restype = ctypes.c_void_p
    lib.puf_metal_gemm_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float]
    lib.puf_metal_gemm_f32.restype = ctypes.c_int
    lib.puf_metal_command_batch_create.argtypes = [ctypes.c_void_p]
    lib.puf_metal_command_batch_create.restype = ctypes.c_void_p
    lib.puf_metal_command_batch_destroy.argtypes = [ctypes.c_void_p]
    lib.puf_metal_command_batch_gemm_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float]
    lib.puf_metal_command_batch_linear_forward_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32]
    lib.puf_metal_command_batch_mingru_scan_forward_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.puf_metal_command_batch_commit.argtypes = [ctypes.c_void_p]
    lib.puf_metal_command_batch_wait.argtypes = [ctypes.c_void_p]


def timed(call, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        call()
    return (time.perf_counter() - start) / iterations


def benchmark_projection_mix(lib, context, rng, warmup, iterations):
    batch_size, observation_size, hidden_size = 8192, 121, 128
    recurrent_size, decoder_size, layers = 3 * hidden_size, 6, 4
    obs_values = rng.uniform(
        -0.5, 0.5, (batch_size, observation_size)).astype(np.float32)
    encoder_weights = rng.uniform(
        -0.5, 0.5, (hidden_size, observation_size)).astype(np.float32)
    recurrent_weights = rng.uniform(
        -0.5, 0.5, (recurrent_size, hidden_size)).astype(np.float32)
    decoder_weights = rng.uniform(
        -0.5, 0.5, (decoder_size, hidden_size)).astype(np.float32)
    cpu_encoder = np.empty((batch_size, hidden_size), dtype=np.float32)
    cpu_recurrent = np.empty((batch_size, recurrent_size), dtype=np.float32)
    cpu_decoder = np.empty((batch_size, decoder_size), dtype=np.float32)

    buffers = [
        Buffer(lib, context, obs_values.size),
        Buffer(lib, context, encoder_weights.size),
        Buffer(lib, context, recurrent_weights.size),
        Buffer(lib, context, decoder_weights.size),
        Buffer(lib, context, cpu_encoder.size),
        Buffer(lib, context, cpu_recurrent.size),
        Buffer(lib, context, cpu_decoder.size),
    ]
    (obs, encoder_weight, recurrent_weight, decoder_weight,
     encoder_out, recurrent_out, decoder_out) = buffers
    try:
        obs.array()[:] = obs_values.ravel()
        encoder_weight.array()[:] = encoder_weights.ravel()
        recurrent_weight.array()[:] = recurrent_weights.ravel()
        decoder_weight.array()[:] = decoder_weights.ravel()

        def check(status):
            if status != 0:
                raise RuntimeError(lib.puf_metal_last_error(context).decode())

        def metal_call():
            batch = lib.puf_metal_command_batch_create(context)
            if not batch:
                raise RuntimeError(lib.puf_metal_last_error(context).decode())
            try:
                check(lib.puf_metal_command_batch_linear_forward_f32(
                    batch, encoder_out.handle, obs.handle,
                    encoder_weight.handle, batch_size,
                    observation_size, hidden_size))
                for _ in range(layers):
                    check(lib.puf_metal_command_batch_linear_forward_f32(
                        batch, recurrent_out.handle, encoder_out.handle,
                        recurrent_weight.handle, batch_size,
                        hidden_size, recurrent_size))
                check(lib.puf_metal_command_batch_linear_forward_f32(
                    batch, decoder_out.handle, encoder_out.handle,
                    decoder_weight.handle, batch_size,
                    hidden_size, decoder_size))
                check(lib.puf_metal_command_batch_commit(batch))
                check(lib.puf_metal_command_batch_wait(batch))
            finally:
                lib.puf_metal_command_batch_destroy(batch)

        def cpu_call():
            np.matmul(obs_values, encoder_weights.T, out=cpu_encoder)
            for _ in range(layers):
                np.matmul(cpu_encoder, recurrent_weights.T, out=cpu_recurrent)
            np.matmul(cpu_encoder, decoder_weights.T, out=cpu_decoder)

        for _ in range(warmup):
            metal_call()
            cpu_call()
        np.testing.assert_allclose(
            encoder_out.array().reshape(cpu_encoder.shape), cpu_encoder,
            rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(
            recurrent_out.array().reshape(cpu_recurrent.shape), cpu_recurrent,
            rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(
            decoder_out.array().reshape(cpu_decoder.shape), cpu_decoder,
            rtol=1e-3, atol=1e-3)

        metal_seconds = timed(metal_call, iterations)
        cpu_seconds = timed(cpu_call, iterations)
        operations = 2.0 * batch_size * (
            observation_size * hidden_size
            + layers * hidden_size * recurrent_size
            + hidden_size * decoder_size)
        return metal_seconds, cpu_seconds, operations
    finally:
        for buffer in buffers:
            buffer.close()


def benchmark_mingru_forward(lib, context, rng, warmup, iterations):
    segments, horizon, observation_size = 128, 64, 121
    hidden_size, recurrent_size, decoder_size, layers = 128, 384, 6, 4
    rows = segments * horizon
    obs_values = rng.uniform(
        -0.5, 0.5, (rows, observation_size)).astype(np.float32)
    encoder_weights = rng.normal(
        0, np.sqrt(2 / observation_size),
        (hidden_size, observation_size)).astype(np.float32)
    recurrent_weights = rng.normal(
        0, 1 / np.sqrt(hidden_size),
        (recurrent_size, hidden_size)).astype(np.float32)
    decoder_weights = rng.normal(
        0, 1 / np.sqrt(hidden_size),
        (decoder_size, hidden_size)).astype(np.float32)
    cpu_hidden = np.empty((segments, horizon, hidden_size), dtype=np.float32)
    cpu_combined = np.empty(
        (segments, horizon, recurrent_size), dtype=np.float32)
    cpu_decoder = np.empty((rows, decoder_size), dtype=np.float32)

    sequence_elements = rows * hidden_size
    state_elements = segments * hidden_size
    checkpoint_elements = segments * (horizon + 1) * hidden_size
    buffers = [
        Buffer(lib, context, obs_values.size),
        Buffer(lib, context, encoder_weights.size),
        Buffer(lib, context, recurrent_weights.size),
        Buffer(lib, context, decoder_weights.size),
        Buffer(lib, context, sequence_elements),
        Buffer(lib, context, rows * recurrent_size),
        Buffer(lib, context, state_elements),
        Buffer(lib, context, state_elements),
        Buffer(lib, context, checkpoint_elements),
        Buffer(lib, context, checkpoint_elements),
        Buffer(lib, context, checkpoint_elements),
        Buffer(lib, context, cpu_decoder.size),
    ]
    (obs, encoder_weight, recurrent_weight, decoder_weight,
     hidden, combined, state, next_state, a_star, s_values, log_values,
     decoder_out) = buffers
    try:
        obs.array()[:] = obs_values.ravel()
        encoder_weight.array()[:] = encoder_weights.ravel()
        recurrent_weight.array()[:] = recurrent_weights.ravel()
        decoder_weight.array()[:] = decoder_weights.ravel()
        state.array().fill(0.0)

        def check(status):
            if status != 0:
                raise RuntimeError(lib.puf_metal_last_error(context).decode())

        def metal_call():
            batch = lib.puf_metal_command_batch_create(context)
            if not batch:
                raise RuntimeError(lib.puf_metal_last_error(context).decode())
            try:
                check(lib.puf_metal_command_batch_linear_forward_f32(
                    batch, hidden.handle, obs.handle, encoder_weight.handle,
                    rows, observation_size, hidden_size))
                for _ in range(layers):
                    check(lib.puf_metal_command_batch_linear_forward_f32(
                        batch, combined.handle, hidden.handle,
                        recurrent_weight.handle,
                        rows, hidden_size, recurrent_size))
                    check(lib.puf_metal_command_batch_mingru_scan_forward_f32(
                        batch, hidden.handle, next_state.handle,
                        a_star.handle, s_values.handle, log_values.handle,
                        combined.handle, state.handle, hidden.handle,
                        segments, horizon, hidden_size))
                check(lib.puf_metal_command_batch_linear_forward_f32(
                    batch, decoder_out.handle, hidden.handle,
                    decoder_weight.handle,
                    rows, hidden_size, decoder_size))
                check(lib.puf_metal_command_batch_commit(batch))
                check(lib.puf_metal_command_batch_wait(batch))
            finally:
                lib.puf_metal_command_batch_destroy(batch)

        def cpu_call():
            np.matmul(
                obs_values, encoder_weights.T,
                out=cpu_hidden.reshape(rows, hidden_size))
            for _ in range(layers):
                np.matmul(
                    cpu_hidden.reshape(rows, hidden_size),
                    recurrent_weights.T,
                    out=cpu_combined.reshape(rows, recurrent_size))
                recurrent_state = np.zeros(
                    (segments, hidden_size), dtype=np.float32)
                for t in range(horizon):
                    x = cpu_hidden[:, t, :].copy()
                    projected = cpu_combined[:, t, :]
                    hidden_value = projected[:, :hidden_size]
                    gate = projected[:, hidden_size:2 * hidden_size]
                    highway = projected[:, 2 * hidden_size:]
                    gate_sigmoid = 1.0 / (1.0 + np.exp(-gate))
                    hidden_tilde = np.where(
                        hidden_value >= 0, hidden_value + 0.5,
                        1.0 / (1.0 + np.exp(-hidden_value)))
                    recurrent_state[:] = ((1.0 - gate_sigmoid) * recurrent_state
                        + gate_sigmoid * hidden_tilde)
                    highway_sigmoid = 1.0 / (1.0 + np.exp(-highway))
                    cpu_hidden[:, t, :] = (highway_sigmoid * recurrent_state
                        + (1.0 - highway_sigmoid) * x)
            np.matmul(
                cpu_hidden.reshape(rows, hidden_size), decoder_weights.T,
                out=cpu_decoder)

        for _ in range(warmup):
            metal_call()
        cpu_call()
        np.testing.assert_allclose(
            hidden.array().reshape(cpu_hidden.shape), cpu_hidden,
            rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(
            decoder_out.array().reshape(cpu_decoder.shape), cpu_decoder,
            rtol=2e-3, atol=2e-3)
        metal_seconds = timed(metal_call, iterations)
        cpu_seconds = timed(cpu_call, iterations)
        gemm_operations = 2.0 * rows * (
            observation_size * hidden_size
            + layers * hidden_size * recurrent_size
            + hidden_size * decoder_size)
        return metal_seconds, cpu_seconds, gemm_operations
    finally:
        for buffer in buffers:
            buffer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error('iterations must be positive and warmup nonnegative')

    args.library.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['bash', str(ROOT / 'scripts' / 'build_metal_backend.sh'),
         str(args.library)], cwd=ROOT, check=True, capture_output=True)
    lib = ctypes.CDLL(str(args.library))
    configure_api(lib)
    error = ctypes.create_string_buffer(4096)
    context = lib.puf_metal_create(os.fsencode(KERNELS), error, len(error))
    if not context:
        raise RuntimeError(error.value.decode())

    rng = np.random.default_rng(20260820)
    print(f'Device: {lib.puf_metal_device_name(context).decode()}')
    print(f'Iterations: {args.iterations} (after {args.warmup} warmups)')
    print('case                     shape             Metal ms   CPU ms   speedup  Metal GF/s  max |err|')
    batch_results = []
    projection_result = None
    forward_result = None
    try:
        for label, m, n, k, transpose_lhs, transpose_rhs in CASES:
            lhs_shape = (k, m) if transpose_lhs else (m, k)
            rhs_shape = (n, k) if transpose_rhs else (k, n)
            lhs_values = rng.uniform(-0.5, 0.5, lhs_shape).astype(np.float32)
            rhs_values = rng.uniform(-0.5, 0.5, rhs_shape).astype(np.float32)
            cpu_out = np.empty((m, n), dtype=np.float32)
            lhs = Buffer(lib, context, m * k)
            rhs = Buffer(lib, context, k * n)
            dst = Buffer(lib, context, m * n)
            try:
                lhs.array()[:] = lhs_values.ravel()
                rhs.array()[:] = rhs_values.ravel()
                lhs_op = lhs_values.T if transpose_lhs else lhs_values
                rhs_op = rhs_values.T if transpose_rhs else rhs_values

                def metal_call():
                    status = lib.puf_metal_gemm_f32(
                        context, dst.handle, lhs.handle, rhs.handle,
                        m, n, k, transpose_lhs, transpose_rhs, 1.0, 0.0)
                    if status != 0:
                        raise RuntimeError(lib.puf_metal_last_error(context).decode())

                def cpu_call():
                    np.matmul(lhs_op, rhs_op, out=cpu_out)

                for _ in range(args.warmup):
                    metal_call()
                    cpu_call()
                actual = dst.array().reshape(m, n)
                max_error = float(np.max(np.abs(actual - cpu_out)))
                np.testing.assert_allclose(
                    actual, cpu_out, rtol=1e-3, atol=1e-3)

                metal_seconds = timed(metal_call, args.iterations)
                cpu_seconds = timed(cpu_call, args.iterations)
                operations = 2.0 * m * n * k
                print(f'{label:24} {m:5}x{n:<4}x{k:<5} '
                      f'{metal_seconds * 1e3:9.3f} '
                      f'{cpu_seconds * 1e3:8.3f} '
                      f'{cpu_seconds / metal_seconds:8.2f}x '
                      f'{operations / metal_seconds / 1e9:11.1f} '
                      f'{max_error:10.3g}')

                if label == 'minibatch_projection':
                    def metal_batch_call(batch_size):
                        batch = lib.puf_metal_command_batch_create(context)
                        if not batch:
                            raise RuntimeError(
                                lib.puf_metal_last_error(context).decode())
                        try:
                            for _ in range(batch_size):
                                status = lib.puf_metal_command_batch_gemm_f32(
                                    batch, dst.handle, lhs.handle, rhs.handle,
                                    m, n, k, transpose_lhs, transpose_rhs,
                                    1.0, 0.0)
                                if status != 0:
                                    raise RuntimeError(lib.puf_metal_last_error(
                                        context).decode())
                            status = lib.puf_metal_command_batch_commit(batch)
                            if status == 0:
                                status = lib.puf_metal_command_batch_wait(batch)
                            if status != 0:
                                raise RuntimeError(lib.puf_metal_last_error(
                                    context).decode())
                        finally:
                            lib.puf_metal_command_batch_destroy(batch)

                    for batch_size in (1, 4, 16):
                        for _ in range(args.warmup):
                            metal_batch_call(batch_size)
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            metal_batch_call(batch_size)
                        per_operation = ((time.perf_counter() - start)
                            / args.iterations / batch_size)
                        batch_results.append(
                            (batch_size, per_operation,
                             operations / per_operation / 1e9))
            finally:
                lhs.close()
                rhs.close()
                dst.close()
        projection_result = benchmark_projection_mix(
            lib, context, rng, args.warmup, args.iterations)
        forward_result = benchmark_mingru_forward(
            lib, context, rng, args.warmup, args.iterations)
    finally:
        lib.puf_metal_destroy(context)

    print('\nminibatch_projection command batching (amortized per GEMM)')
    print('GEMMs/commit   Metal ms/op   Metal GF/s')
    for batch_size, seconds, gflops in batch_results:
        print(f'{batch_size:12} {seconds * 1e3:13.3f} {gflops:12.1f}')

    metal_seconds, cpu_seconds, operations = projection_result
    print('\nencoder + 4 MinGRU + decoder projection mix (6 GEMMs/commit)')
    print(f'Metal: {metal_seconds * 1e3:.3f} ms  '
          f'CPU: {cpu_seconds * 1e3:.3f} ms  '
          f'speedup: {cpu_seconds / metal_seconds:.2f}x  '
          f'Metal: {operations / metal_seconds / 1e9:.1f} GF/s')

    metal_seconds, cpu_seconds, operations = forward_result
    print('\nencoder + 4 MinGRU scans + decoder forward (one command buffer)')
    print(f'Metal: {metal_seconds * 1e3:.3f} ms  '
          f'NumPy CPU reference: {cpu_seconds * 1e3:.3f} ms  '
          f'speedup: {cpu_seconds / metal_seconds:.2f}x  '
          f'GEMM-equivalent: {operations / metal_seconds / 1e9:.1f} GF/s')


if __name__ == '__main__':
    main()
