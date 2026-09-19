import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build' / 'metal-tests'
DYLIB = BUILD / 'libpuffer_metal.dylib'
KERNELS = ROOT / 'src' / 'metal' / 'kernels.metal'

PUF_METAL_OK = 0
PUF_METAL_INVALID_ARGUMENT = 1


def sigmoid(value):
    z = math.exp(-abs(value))
    return 1.0 / (1.0 + z) if value >= 0 else z / (1.0 + z)


def fast_sigmoid(value):
    x = min(9.0, max(-9.0, value * 0.5))
    x2 = x * x
    numerator = x2 * -2.76076847742355e-16 + 2.00018790482477e-13
    numerator = x2 * numerator + -8.60467152213735e-11
    numerator = x2 * numerator + 5.12229709037114e-08
    numerator = x2 * numerator + 1.48572235717979e-05
    numerator = x2 * numerator + 6.37261928875436e-04
    numerator = x2 * numerator + 4.89352455891786e-03
    numerator = x * numerator
    denominator = x2 * 1.19825839466702e-06 + 1.18534705686654e-04
    denominator = x2 * denominator + 2.26843463243900e-03
    denominator = x2 * denominator + 4.89352518554385e-03
    return min(1.0, max(0.0, (numerator / denominator + 1.0) * 0.5))


def mingru_rollout_reference(combined, state, inputs):
    batch_size, hidden3 = combined.shape
    hidden_size = hidden3 // 3
    output = np.empty_like(inputs)
    next_state = np.empty_like(state)
    for b in range(batch_size):
        for h in range(hidden_size):
            hidden = float(combined[b, h])
            gate = float(combined[b, hidden_size + h])
            projection = float(combined[b, 2 * hidden_size + h])
            hidden_tilde = hidden + 0.5 if hidden >= 0 else fast_sigmoid(hidden)
            gate_value = sigmoid(gate)
            recurrent = ((1.0 - gate_value) * float(state[b, h])
                + gate_value * hidden_tilde)
            next_state[b, h] = recurrent
            projection_gate = sigmoid(projection)
            output[b, h] = (projection_gate * recurrent
                + (1.0 - projection_gate) * float(inputs[b, h]))
    return output, next_state


def mingru_scan_reference(combined, state, inputs, sentinel):
    batch_size, horizon, hidden3 = combined.shape
    hidden_size = hidden3 // 3
    output = np.empty_like(inputs)
    next_state = np.empty_like(state)
    checkpoints = tuple(
        np.full((batch_size, horizon + 1, hidden_size), sentinel,
                dtype=np.float32)
        for _ in range(3))
    a_buffer, s_buffer, log_buffer = checkpoints
    for b in range(batch_size):
        for h in range(hidden_size):
            recurrent = float(state[b, h])
            log_value = math.log(recurrent) if recurrent > 0 else -math.inf
            a_value = 0.0
            s_value = log_value
            a_buffer[b, 0, h] = a_value
            s_buffer[b, 0, h] = s_value
            log_buffer[b, 0, h] = log_value
            scan_result = 0.0
            for t in range(horizon):
                hidden = float(combined[b, t, h])
                gate = float(combined[b, t, hidden_size + h])
                projection = float(combined[b, t, 2 * hidden_size + h])

                abs_gate = abs(gate)
                softplus_tail = math.log1p(math.exp(-abs_gate))
                softplus_gate = (gate + softplus_tail
                    if gate >= 0 else softplus_tail)
                softplus_negative_gate = (softplus_tail
                    if gate >= 0 else -gate + softplus_tail)
                log_coeff = -softplus_gate
                log_hidden = (math.log(hidden + 0.5) if hidden >= 0
                    else -math.log1p(math.exp(-hidden)))
                log_value = -softplus_negative_gate + log_hidden
                a_value += log_coeff
                s_value = float(np.logaddexp(s_value, log_value - a_value))
                scan_result = math.exp(a_value + s_value)

                # An independent direct recurrence is the output baseline.
                gate_value = sigmoid(gate)
                hidden_tilde = hidden + 0.5 if hidden >= 0 else sigmoid(hidden)
                recurrent = ((1.0 - gate_value) * recurrent
                    + gate_value * hidden_tilde)
                projection_gate = sigmoid(projection)
                output[b, t, h] = (projection_gate * recurrent
                    + (1.0 - projection_gate) * float(inputs[b, t, h]))
                if (t + 1) % 4 == 0:
                    a_buffer[b, t + 1, h] = a_value
                    s_buffer[b, t + 1, h] = s_value
                    log_buffer[b, t + 1, h] = log_value
            next_state[b, h] = recurrent if horizon else 0.0
    return output, next_state, checkpoints


def mingru_scan_backward_reference(
        combined, state, inputs, grad_output, grad_next_state):
    batch_size, horizon, hidden3 = combined.shape
    hidden_size = hidden3 // 3
    grad_combined = np.zeros_like(combined, dtype=np.float64)
    grad_state = np.zeros_like(state, dtype=np.float64)
    grad_input = np.zeros_like(inputs, dtype=np.float64)
    for b in range(batch_size):
        for h in range(hidden_size):
            recurrent = float(state[b, h])
            previous_states = []
            recurrent_states = []
            for t in range(horizon):
                hidden = float(combined[b, t, h])
                gate = float(combined[b, t, hidden_size + h])
                hidden_tilde = hidden + 0.5 if hidden >= 0 else sigmoid(hidden)
                previous_states.append(recurrent)
                gate_value = sigmoid(gate)
                recurrent = ((1.0 - gate_value) * recurrent
                    + gate_value * hidden_tilde)
                recurrent_states.append(recurrent)

            carry = float(grad_next_state[b, h])
            for t in range(horizon - 1, -1, -1):
                hidden = float(combined[b, t, h])
                gate = float(combined[b, t, hidden_size + h])
                projection = float(combined[b, t, 2 * hidden_size + h])
                x = float(inputs[b, t, h])
                upstream = float(grad_output[b, t, h])
                projection_gate = sigmoid(projection)
                grad_recurrent = carry + upstream * projection_gate
                grad_combined[b, t, 2 * hidden_size + h] = (
                    upstream * (recurrent_states[t] - x)
                    * projection_gate * (1.0 - projection_gate))
                grad_input[b, t, h] = upstream * (1.0 - projection_gate)

                gate_value = sigmoid(gate)
                hidden_tilde = hidden + 0.5 if hidden >= 0 else sigmoid(hidden)
                hidden_derivative = (1.0 if hidden >= 0
                    else hidden_tilde * (1.0 - hidden_tilde))
                grad_combined[b, t, h] = (
                    grad_recurrent * gate_value * hidden_derivative)
                grad_combined[b, t, hidden_size + h] = (
                    grad_recurrent
                    * (hidden_tilde - previous_states[t])
                    * gate_value * (1.0 - gate_value))
                carry = grad_recurrent * (1.0 - gate_value)
            grad_state[b, h] = carry
    return grad_combined, grad_state, grad_input


class MetalBuffer:
    def __init__(self, case, size_bytes):
        self.case = case
        self.handle = case.lib.puf_metal_buffer_create_shared(
            case.context, size_bytes)
        if not self.handle:
            case.fail(case.last_error())
        self.size_bytes = size_bytes

    def close(self):
        if self.handle:
            self.case.lib.puf_metal_buffer_destroy(self.handle)
            self.handle = None

    def array(self, dtype, count):
        self.case.assertLessEqual(count * np.dtype(dtype).itemsize, self.size_bytes)
        address = self.case.lib.puf_metal_buffer_contents(self.handle)
        self.case.assertTrue(address)
        ctype = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
        raw = (ctype * count).from_address(address)
        return np.ctypeslib.as_array(raw)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


@unittest.skipUnless(sys.platform == 'darwin', 'Metal requires macOS')
class MetalBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BUILD.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ['bash', str(ROOT / 'scripts' / 'build_metal_backend.sh'), str(DYLIB)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        cls.lib = ctypes.CDLL(str(DYLIB))
        cls._configure_api()
        if not cls.lib.puf_metal_is_available():
            raise unittest.SkipTest('No Metal device is available')

        error = ctypes.create_string_buffer(4096)
        cls.context = cls.lib.puf_metal_create(
            os.fsencode(KERNELS), error, len(error))
        if not cls.context:
            raise RuntimeError(error.value.decode())

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, 'context', None):
            cls.lib.puf_metal_destroy(cls.context)
            cls.context = None

    @classmethod
    def _configure_api(cls):
        lib = cls.lib
        lib.puf_metal_is_available.restype = ctypes.c_int
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
        lib.puf_metal_buffer_size.argtypes = [ctypes.c_void_p]
        lib.puf_metal_buffer_size.restype = ctypes.c_size_t
        lib.puf_metal_fill_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_uint32]
        lib.puf_metal_add_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        lib.puf_metal_clamp_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
            ctypes.c_float, ctypes.c_uint32]
        lib.puf_metal_cast_u8_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        lib.puf_metal_transpose_102_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        lib.puf_metal_gemm_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float]
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
        lib.puf_metal_command_batch_linear_backward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        lib.puf_metal_command_batch_mingru_gate_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32]
        lib.puf_metal_command_batch_mingru_scan_forward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        lib.puf_metal_command_batch_mingru_scan_backward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32]
        lib.puf_metal_command_batch_commit.argtypes = [ctypes.c_void_p]
        lib.puf_metal_command_batch_wait.argtypes = [ctypes.c_void_p]

    def last_error(self):
        return self.lib.puf_metal_last_error(self.context).decode()

    def assert_ok(self, status):
        self.assertEqual(status, PUF_METAL_OK, self.last_error())

    def test_device_and_shared_buffer_lifecycle(self):
        name = self.lib.puf_metal_device_name(self.context).decode()
        self.assertTrue(name)
        with MetalBuffer(self, 37) as buffer:
            self.assertEqual(self.lib.puf_metal_buffer_size(buffer.handle), 37)
            self.assertTrue(self.lib.puf_metal_buffer_contents(buffer.handle))

    def test_empty_dispatches(self):
        with MetalBuffer(self, 0) as empty:
            self.assert_ok(self.lib.puf_metal_fill_f32(
                self.context, empty.handle, 3.0, 0))
            self.assert_ok(self.lib.puf_metal_add_f32(
                self.context, empty.handle, empty.handle, 0))

    def test_primitives_match_cpu_at_boundary_sizes(self):
        rng = np.random.default_rng(20260820)
        for n in (1, 31, 32, 33, 255, 256, 257, 4097):
            with self.subTest(n=n), \
                    MetalBuffer(self, n * 4) as dst, \
                    MetalBuffer(self, n * 4) as src, \
                    MetalBuffer(self, n) as src_u8:
                dst_array = dst.array(np.float32, n)
                src_array = src.array(np.float32, n)
                u8_array = src_u8.array(np.uint8, n)

                self.assert_ok(self.lib.puf_metal_fill_f32(
                    self.context, dst.handle, 1.25, n))
                np.testing.assert_array_equal(
                    dst_array, np.full(n, 1.25, dtype=np.float32))

                initial = rng.uniform(-4, 4, n).astype(np.float32)
                source = rng.uniform(-2, 2, n).astype(np.float32)
                dst_array[:] = initial
                src_array[:] = source
                self.assert_ok(self.lib.puf_metal_add_f32(
                    self.context, dst.handle, src.handle, n))
                np.testing.assert_array_equal(dst_array, initial + source)

                unclamped = rng.uniform(-10, 10, n).astype(np.float32)
                dst_array[:] = unclamped
                self.assert_ok(self.lib.puf_metal_clamp_f32(
                    self.context, dst.handle, -1.5, 2.25, n))
                np.testing.assert_array_equal(
                    dst_array, np.clip(unclamped, -1.5, 2.25))

                u8_values = rng.integers(0, 256, n, dtype=np.uint8)
                u8_array[:] = u8_values
                self.assert_ok(self.lib.puf_metal_cast_u8_f32(
                    self.context, dst.handle, src_u8.handle, n))
                np.testing.assert_array_equal(
                    dst_array, u8_values.astype(np.float32))

    def test_transpose_102_matches_cpu(self):
        rng = np.random.default_rng(1973)
        for shape in ((1, 1, 1), (2, 3, 1), (3, 2, 7), (17, 5, 3)):
            with self.subTest(shape=shape):
                n = int(np.prod(shape))
                with MetalBuffer(self, n * 4) as src, \
                        MetalBuffer(self, n * 4) as dst:
                    values = rng.standard_normal(n).astype(np.float32)
                    src.array(np.float32, n)[:] = values
                    self.assert_ok(self.lib.puf_metal_transpose_102_f32(
                        self.context, dst.handle, src.handle, *shape))
                    expected = values.reshape(shape).transpose(1, 0, 2).reshape(-1)
                    np.testing.assert_array_equal(
                        dst.array(np.float32, n), expected)

    def test_gemm_matches_cpu_for_all_transpose_modes(self):
        rng = np.random.default_rng(314159)
        shapes = ((1, 1, 1), (2, 3, 4), (17, 13, 19), (65, 33, 17))
        scales = ((1.0, 0.0), (0.375, -0.25))
        for m, n, k in shapes:
            for transpose_lhs in (0, 1):
                for transpose_rhs in (0, 1):
                    for alpha, beta in scales:
                        label = (m, n, k, transpose_lhs, transpose_rhs,
                            alpha, beta)
                        with self.subTest(gemm=label), \
                                MetalBuffer(self, m * k * 4) as lhs, \
                                MetalBuffer(self, k * n * 4) as rhs, \
                                MetalBuffer(self, m * n * 4) as dst:
                            lhs_shape = ((k, m) if transpose_lhs else (m, k))
                            rhs_shape = ((n, k) if transpose_rhs else (k, n))
                            lhs_values = rng.uniform(
                                -1, 1, lhs_shape).astype(np.float32)
                            rhs_values = rng.uniform(
                                -1, 1, rhs_shape).astype(np.float32)
                            dst_initial = rng.uniform(
                                -1, 1, (m, n)).astype(np.float32)
                            lhs.array(np.float32, m * k)[:] = lhs_values.ravel()
                            rhs.array(np.float32, k * n)[:] = rhs_values.ravel()
                            actual = dst.array(np.float32, m * n).reshape(m, n)
                            actual[:] = dst_initial

                            self.assert_ok(self.lib.puf_metal_gemm_f32(
                                self.context, dst.handle, lhs.handle, rhs.handle,
                                m, n, k, transpose_lhs, transpose_rhs,
                                alpha, beta))
                            lhs_op = lhs_values.T if transpose_lhs else lhs_values
                            rhs_op = rhs_values.T if transpose_rhs else rhs_values
                            expected = np.float32(alpha) * (lhs_op @ rhs_op)
                            expected += np.float32(beta) * dst_initial
                            np.testing.assert_allclose(
                                actual, expected, rtol=2e-5, atol=2e-5)

    def test_gemm_identity_is_exact(self):
        rng = np.random.default_rng(2718)
        m, n, k = 23, 11, 11
        values = rng.integers(-16, 17, (m, k)).astype(np.float32)
        identity = np.eye(k, n, dtype=np.float32)
        with MetalBuffer(self, values.nbytes) as lhs, \
                MetalBuffer(self, identity.nbytes) as rhs, \
                MetalBuffer(self, m * n * 4) as dst:
            lhs.array(np.float32, values.size)[:] = values.ravel()
            rhs.array(np.float32, identity.size)[:] = identity.ravel()
            self.assert_ok(self.lib.puf_metal_gemm_f32(
                self.context, dst.handle, lhs.handle, rhs.handle,
                m, n, k, 0, 0, 1.0, 0.0))
            np.testing.assert_array_equal(
                dst.array(np.float32, m * n).reshape(m, n), values[:, :n])

    def test_gemm_empty_dimensions_follow_blas_semantics(self):
        with MetalBuffer(self, 0) as lhs, MetalBuffer(self, 0) as rhs, \
                MetalBuffer(self, 3 * 5 * 4) as dst:
            actual = dst.array(np.float32, 15)
            for beta in (0.0, 1.0, -0.25):
                initial = np.arange(15, dtype=np.float32) - 7
                actual[:] = initial
                self.assert_ok(self.lib.puf_metal_gemm_f32(
                    self.context, dst.handle, lhs.handle, rhs.handle,
                    3, 5, 0, 0, 0, 42.0, beta))
                np.testing.assert_array_equal(
                    actual, np.float32(beta) * initial)

        with MetalBuffer(self, 0) as empty, MetalBuffer(self, 4) as scalar:
            scalar.array(np.float32, 1)[0] = 9.0
            self.assert_ok(self.lib.puf_metal_gemm_f32(
                self.context, empty.handle, empty.handle, scalar.handle,
                0, 7, 3, 0, 0, 1.0, 0.0))

    def test_gemm_invalid_arguments_are_reported(self):
        with MetalBuffer(self, 4 * 4) as small, \
                MetalBuffer(self, 4 * 16) as rhs, \
                MetalBuffer(self, 4 * 16) as dst:
            status = self.lib.puf_metal_gemm_f32(
                self.context, dst.handle, small.handle, rhs.handle,
                4, 4, 4, 0, 0, 1.0, 0.0)
            self.assertEqual(status, PUF_METAL_INVALID_ARGUMENT)
            self.assertIn('smaller', self.last_error())

            status = self.lib.puf_metal_gemm_f32(
                self.context, dst.handle, rhs.handle, rhs.handle,
                4, 4, 4, 0, 0, 1.0, 0.0)
            self.assertEqual(status, PUF_METAL_INVALID_ARGUMENT)
            self.assertIn('alias', self.last_error())

            status = self.lib.puf_metal_gemm_f32(
                self.context, dst.handle, rhs.handle, small.handle,
                4, 4, 4, 2, 0, 1.0, 0.0)
            self.assertEqual(status, PUF_METAL_INVALID_ARGUMENT)
            self.assertIn('transpose', self.last_error())

    def test_command_batch_preserves_dependent_gemm_order(self):
        rng = np.random.default_rng(1618)
        m, k, n, p = 29, 17, 23, 13
        a_values = rng.uniform(-0.5, 0.5, (m, k)).astype(np.float32)
        b_values = rng.uniform(-0.5, 0.5, (k, n)).astype(np.float32)
        c_values = rng.uniform(-0.5, 0.5, (n, p)).astype(np.float32)
        with MetalBuffer(self, a_values.nbytes) as a, \
                MetalBuffer(self, b_values.nbytes) as b, \
                MetalBuffer(self, c_values.nbytes) as c, \
                MetalBuffer(self, m * n * 4) as intermediate, \
                MetalBuffer(self, m * p * 4) as output:
            a.array(np.float32, a_values.size)[:] = a_values.ravel()
            b.array(np.float32, b_values.size)[:] = b_values.ravel()
            c.array(np.float32, c_values.size)[:] = c_values.ravel()
            batch = self.lib.puf_metal_command_batch_create(self.context)
            self.assertTrue(batch, self.last_error())
            try:
                self.assert_ok(self.lib.puf_metal_command_batch_gemm_f32(
                    batch, intermediate.handle, a.handle, b.handle,
                    m, n, k, 0, 0, 1.0, 0.0))
                self.assert_ok(self.lib.puf_metal_command_batch_gemm_f32(
                    batch, output.handle, intermediate.handle, c.handle,
                    m, p, n, 0, 0, 1.0, 0.0))
                self.assert_ok(self.lib.puf_metal_command_batch_commit(batch))
                self.assert_ok(self.lib.puf_metal_command_batch_wait(batch))
                self.assert_ok(self.lib.puf_metal_command_batch_wait(batch))
            finally:
                self.lib.puf_metal_command_batch_destroy(batch)
            expected = (a_values @ b_values) @ c_values
            np.testing.assert_allclose(
                output.array(np.float32, m * p).reshape(m, p),
                expected, rtol=3e-5, atol=3e-5)

    def test_command_batch_state_errors_are_reported(self):
        batch = self.lib.puf_metal_command_batch_create(self.context)
        self.assertTrue(batch, self.last_error())
        try:
            status = self.lib.puf_metal_command_batch_wait(batch)
            self.assertEqual(status, PUF_METAL_INVALID_ARGUMENT)
            self.assertIn('committed', self.last_error())

            self.assert_ok(self.lib.puf_metal_command_batch_commit(batch))
            status = self.lib.puf_metal_command_batch_commit(batch)
            self.assertEqual(status, PUF_METAL_INVALID_ARGUMENT)
            self.assertIn('already committed', self.last_error())

            with MetalBuffer(self, 4) as a, MetalBuffer(self, 4) as b, \
                    MetalBuffer(self, 4) as dst:
                status = self.lib.puf_metal_command_batch_gemm_f32(
                    batch, dst.handle, a.handle, b.handle,
                    1, 1, 1, 0, 0, 1.0, 0.0)
                self.assertEqual(status, PUF_METAL_INVALID_ARGUMENT)
                self.assertIn('committed', self.last_error())
            self.assert_ok(self.lib.puf_metal_command_batch_wait(batch))
        finally:
            self.lib.puf_metal_command_batch_destroy(batch)

    def test_linear_forward_and_backward_match_cpu(self):
        rng = np.random.default_rng(4242)
        for batch_size, input_size, output_size in (
                (1, 1, 1), (11, 7, 5), (257, 33, 65)):
            with self.subTest(shape=(batch_size, input_size, output_size)):
                input_values = rng.uniform(
                    -0.5, 0.5, (batch_size, input_size)).astype(np.float32)
                weight_values = rng.uniform(
                    -0.5, 0.5, (output_size, input_size)).astype(np.float32)
                grad_values = rng.uniform(
                    -0.5, 0.5, (batch_size, output_size)).astype(np.float32)
                with MetalBuffer(self, input_values.nbytes) as input_buffer, \
                        MetalBuffer(self, weight_values.nbytes) as weight, \
                        MetalBuffer(self, grad_values.nbytes) as grad_output, \
                        MetalBuffer(self, batch_size * output_size * 4) as output, \
                        MetalBuffer(self, input_values.nbytes) as grad_input, \
                        MetalBuffer(self, weight_values.nbytes) as grad_weight:
                    input_buffer.array(
                        np.float32, input_values.size)[:] = input_values.ravel()
                    weight.array(
                        np.float32, weight_values.size)[:] = weight_values.ravel()
                    grad_output.array(
                        np.float32, grad_values.size)[:] = grad_values.ravel()
                    command_batch = self.lib.puf_metal_command_batch_create(
                        self.context)
                    self.assertTrue(command_batch, self.last_error())
                    try:
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_linear_forward_f32(
                                command_batch, output.handle,
                                input_buffer.handle, weight.handle,
                                batch_size, input_size, output_size))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_linear_backward_f32(
                                command_batch, grad_input.handle,
                                grad_weight.handle, grad_output.handle,
                                input_buffer.handle, weight.handle,
                                batch_size, input_size, output_size))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_commit(command_batch))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_wait(command_batch))
                    finally:
                        self.lib.puf_metal_command_batch_destroy(command_batch)

                    expected_output = input_values @ weight_values.T
                    expected_grad_input = grad_values @ weight_values
                    expected_grad_weight = grad_values.T @ input_values
                    np.testing.assert_allclose(
                        output.array(np.float32, expected_output.size).reshape(
                            expected_output.shape),
                        expected_output, rtol=3e-5, atol=3e-5)
                    np.testing.assert_allclose(
                        grad_input.array(np.float32, expected_grad_input.size).reshape(
                            expected_grad_input.shape),
                        expected_grad_input, rtol=3e-5, atol=3e-5)
                    np.testing.assert_allclose(
                        grad_weight.array(
                            np.float32, expected_grad_weight.size).reshape(
                                expected_grad_weight.shape),
                        expected_grad_weight, rtol=3e-5, atol=3e-5)

    def test_linear_gradients_match_finite_differences(self):
        rng = np.random.default_rng(99)
        batch_size, input_size, output_size = 4, 3, 5
        inputs = rng.uniform(-0.5, 0.5, (batch_size, input_size))
        weights = rng.uniform(-0.5, 0.5, (output_size, input_size))
        upstream = rng.uniform(-0.5, 0.5, (batch_size, output_size))
        expected_grad_input = upstream @ weights
        expected_grad_weight = upstream.T @ inputs

        epsilon = 1e-6
        for row, column in ((0, 0), (2, 1), (3, 2)):
            plus = inputs.copy()
            minus = inputs.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical = (np.sum((plus @ weights.T) * upstream)
                - np.sum((minus @ weights.T) * upstream)) / (2 * epsilon)
            self.assertAlmostEqual(
                numerical, expected_grad_input[row, column], places=8)
        for row, column in ((0, 0), (3, 1), (4, 2)):
            plus = weights.copy()
            minus = weights.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical = (np.sum((inputs @ plus.T) * upstream)
                - np.sum((inputs @ minus.T) * upstream)) / (2 * epsilon)
            self.assertAlmostEqual(
                numerical, expected_grad_weight[row, column], places=8)

        inputs_f32 = inputs.astype(np.float32)
        weights_f32 = weights.astype(np.float32)
        upstream_f32 = upstream.astype(np.float32)
        with MetalBuffer(self, inputs_f32.nbytes) as input_buffer, \
                MetalBuffer(self, weights_f32.nbytes) as weight_buffer, \
                MetalBuffer(self, upstream_f32.nbytes) as grad_output, \
                MetalBuffer(self, inputs_f32.nbytes) as grad_input, \
                MetalBuffer(self, weights_f32.nbytes) as grad_weight:
            input_buffer.array(np.float32, inputs_f32.size)[:] = inputs_f32.ravel()
            weight_buffer.array(
                np.float32, weights_f32.size)[:] = weights_f32.ravel()
            grad_output.array(
                np.float32, upstream_f32.size)[:] = upstream_f32.ravel()
            command_batch = self.lib.puf_metal_command_batch_create(self.context)
            self.assertTrue(command_batch, self.last_error())
            try:
                self.assert_ok(
                    self.lib.puf_metal_command_batch_linear_backward_f32(
                        command_batch, grad_input.handle, grad_weight.handle,
                        grad_output.handle, input_buffer.handle,
                        weight_buffer.handle,
                        batch_size, input_size, output_size))
                self.assert_ok(
                    self.lib.puf_metal_command_batch_commit(command_batch))
                self.assert_ok(
                    self.lib.puf_metal_command_batch_wait(command_batch))
            finally:
                self.lib.puf_metal_command_batch_destroy(command_batch)
            np.testing.assert_allclose(
                grad_input.array(np.float32, inputs_f32.size).reshape(
                    inputs_f32.shape),
                expected_grad_input, rtol=3e-5, atol=3e-5)
            np.testing.assert_allclose(
                grad_weight.array(np.float32, weights_f32.size).reshape(
                    weights_f32.shape),
                expected_grad_weight, rtol=3e-5, atol=3e-5)

    def test_mingru_rollout_gate_matches_cpu(self):
        rng = np.random.default_rng(123456)
        for batch_size, hidden_size in ((1, 1), (3, 5), (17, 33)):
            with self.subTest(shape=(batch_size, hidden_size)):
                combined_values = rng.uniform(
                    -6, 6, (batch_size, 3 * hidden_size)).astype(np.float32)
                state_values = rng.uniform(
                    0, 2, (batch_size, hidden_size)).astype(np.float32)
                input_values = rng.uniform(
                    -2, 2, (batch_size, hidden_size)).astype(np.float32)
                expected_output, expected_state = mingru_rollout_reference(
                    combined_values, state_values, input_values)
                with MetalBuffer(self, combined_values.nbytes) as combined, \
                        MetalBuffer(self, state_values.nbytes) as state, \
                        MetalBuffer(self, input_values.nbytes) as input_buffer, \
                        MetalBuffer(self, input_values.nbytes) as output, \
                        MetalBuffer(self, state_values.nbytes) as next_state:
                    combined.array(
                        np.float32, combined_values.size)[:] = combined_values.ravel()
                    state.array(
                        np.float32, state_values.size)[:] = state_values.ravel()
                    input_buffer.array(
                        np.float32, input_values.size)[:] = input_values.ravel()
                    command_batch = self.lib.puf_metal_command_batch_create(
                        self.context)
                    self.assertTrue(command_batch, self.last_error())
                    try:
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_mingru_gate_f32(
                                command_batch, output.handle, next_state.handle,
                                combined.handle, state.handle, input_buffer.handle,
                                batch_size, hidden_size))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_commit(command_batch))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_wait(command_batch))
                    finally:
                        self.lib.puf_metal_command_batch_destroy(command_batch)
                    np.testing.assert_allclose(
                        output.array(np.float32, input_values.size).reshape(
                            input_values.shape),
                        expected_output, rtol=3e-5, atol=3e-5)
                    np.testing.assert_allclose(
                        next_state.array(np.float32, state_values.size).reshape(
                            state_values.shape),
                        expected_state, rtol=3e-5, atol=3e-5)

                    # Multi-layer rollout intentionally reuses its output as
                    # the next layer's input; validate that supported alias.
                    input_buffer.array(
                        np.float32, input_values.size)[:] = input_values.ravel()
                    command_batch = self.lib.puf_metal_command_batch_create(
                        self.context)
                    self.assertTrue(command_batch, self.last_error())
                    try:
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_mingru_gate_f32(
                                command_batch, input_buffer.handle,
                                next_state.handle, combined.handle, state.handle,
                                input_buffer.handle, batch_size, hidden_size))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_commit(command_batch))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_wait(command_batch))
                    finally:
                        self.lib.puf_metal_command_batch_destroy(command_batch)
                    np.testing.assert_allclose(
                        input_buffer.array(
                            np.float32, input_values.size).reshape(input_values.shape),
                        expected_output, rtol=3e-5, atol=3e-5)

    def test_mingru_training_scan_matches_cpu_recurrence(self):
        rng = np.random.default_rng(7890)
        sentinel = np.float32(12345.0)
        for batch_size, horizon, hidden_size in (
                (1, 1, 1), (2, 7, 5), (3, 8, 17), (2, 0, 3)):
            with self.subTest(shape=(batch_size, horizon, hidden_size)):
                combined_values = rng.uniform(
                    -5, 5, (batch_size, horizon, 3 * hidden_size)).astype(
                        np.float32)
                state_values = rng.uniform(
                    0.01, 2, (batch_size, hidden_size)).astype(np.float32)
                state_values.ravel()[0] = 0.0
                input_values = rng.uniform(
                    -2, 2, (batch_size, horizon, hidden_size)).astype(np.float32)
                expected_output, expected_state, expected_checkpoints = \
                    mingru_scan_reference(
                        combined_values, state_values, input_values, sentinel)
                sequence_elements = batch_size * horizon * hidden_size
                state_elements = batch_size * hidden_size
                combined_elements = 3 * sequence_elements
                checkpoint_elements = batch_size * (horizon + 1) * hidden_size
                with MetalBuffer(self, sequence_elements * 4) as output, \
                        MetalBuffer(self, state_elements * 4) as next_state, \
                        MetalBuffer(self, checkpoint_elements * 4) as a_star, \
                        MetalBuffer(self, checkpoint_elements * 4) as s_values, \
                        MetalBuffer(self, checkpoint_elements * 4) as log_values, \
                        MetalBuffer(self, combined_elements * 4) as combined, \
                        MetalBuffer(self, state_elements * 4) as state, \
                        MetalBuffer(self, sequence_elements * 4) as input_buffer:
                    combined.array(np.float32, combined_elements)[:] = \
                        combined_values.ravel()
                    state.array(np.float32, state_elements)[:] = state_values.ravel()
                    input_buffer.array(np.float32, sequence_elements)[:] = \
                        input_values.ravel()
                    for checkpoint in (a_star, s_values, log_values):
                        checkpoint.array(
                            np.float32, checkpoint_elements).fill(sentinel)

                    command_batch = self.lib.puf_metal_command_batch_create(
                        self.context)
                    self.assertTrue(command_batch, self.last_error())
                    try:
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_mingru_scan_forward_f32(
                                command_batch, output.handle, next_state.handle,
                                a_star.handle, s_values.handle, log_values.handle,
                                combined.handle, state.handle, input_buffer.handle,
                                batch_size, horizon, hidden_size))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_commit(command_batch))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_wait(command_batch))
                    finally:
                        self.lib.puf_metal_command_batch_destroy(command_batch)

                    np.testing.assert_allclose(
                        output.array(np.float32, sequence_elements).reshape(
                            expected_output.shape),
                        expected_output, rtol=5e-5, atol=5e-5)
                    np.testing.assert_allclose(
                        next_state.array(np.float32, state_elements).reshape(
                            expected_state.shape),
                        expected_state, rtol=5e-5, atol=5e-5)
                    checkpoint_shape = expected_checkpoints[0].shape
                    for actual_buffer, expected in zip(
                            (a_star, s_values, log_values), expected_checkpoints):
                        np.testing.assert_allclose(
                            actual_buffer.array(
                                np.float32, checkpoint_elements).reshape(
                                    checkpoint_shape),
                            expected, rtol=5e-5, atol=5e-5)

    def test_mingru_scan_backward_matches_cpu(self):
        rng = np.random.default_rng(8675309)
        for batch_size, horizon, hidden_size in (
                (1, 1, 1), (2, 7, 5), (2, 8, 3)):
            with self.subTest(shape=(batch_size, horizon, hidden_size)):
                combined_values = rng.uniform(
                    -1.5, 1.5,
                    (batch_size, horizon, 3 * hidden_size)).astype(np.float32)
                state_values = rng.uniform(
                    0.05, 1.5, (batch_size, hidden_size)).astype(np.float32)
                state_values.ravel()[0] = 0.0
                input_values = rng.uniform(
                    -1, 1, (batch_size, horizon, hidden_size)).astype(np.float32)
                grad_output_values = rng.uniform(
                    -1, 1, (batch_size, horizon, hidden_size)).astype(np.float32)
                grad_next_values = rng.uniform(
                    -1, 1, (batch_size, hidden_size)).astype(np.float32)
                expected_grads = mingru_scan_backward_reference(
                    combined_values, state_values, input_values,
                    grad_output_values, grad_next_values)

                sequence_elements = batch_size * horizon * hidden_size
                state_elements = batch_size * hidden_size
                combined_elements = 3 * sequence_elements
                checkpoint_elements = batch_size * (horizon + 1) * hidden_size
                with MetalBuffer(self, sequence_elements * 4) as output, \
                        MetalBuffer(self, state_elements * 4) as next_state, \
                        MetalBuffer(self, checkpoint_elements * 4) as a_star, \
                        MetalBuffer(self, checkpoint_elements * 4) as s_values, \
                        MetalBuffer(self, checkpoint_elements * 4) as log_values, \
                        MetalBuffer(self, combined_elements * 4) as combined, \
                        MetalBuffer(self, state_elements * 4) as state, \
                        MetalBuffer(self, sequence_elements * 4) as input_buffer, \
                        MetalBuffer(self, sequence_elements * 4) as grad_output, \
                        MetalBuffer(self, state_elements * 4) as grad_next, \
                        MetalBuffer(self, combined_elements * 4) as grad_combined, \
                        MetalBuffer(self, state_elements * 4) as grad_state, \
                        MetalBuffer(self, sequence_elements * 4) as grad_input:
                    combined.array(np.float32, combined_elements)[:] = \
                        combined_values.ravel()
                    state.array(np.float32, state_elements)[:] = state_values.ravel()
                    input_buffer.array(np.float32, sequence_elements)[:] = \
                        input_values.ravel()
                    grad_output.array(np.float32, sequence_elements)[:] = \
                        grad_output_values.ravel()
                    grad_next.array(np.float32, state_elements)[:] = \
                        grad_next_values.ravel()

                    command_batch = self.lib.puf_metal_command_batch_create(
                        self.context)
                    self.assertTrue(command_batch, self.last_error())
                    try:
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_mingru_scan_forward_f32(
                                command_batch, output.handle, next_state.handle,
                                a_star.handle, s_values.handle, log_values.handle,
                                combined.handle, state.handle, input_buffer.handle,
                                batch_size, horizon, hidden_size))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_mingru_scan_backward_f32(
                                command_batch, grad_combined.handle,
                                grad_state.handle, grad_input.handle,
                                combined.handle, a_star.handle, s_values.handle,
                                input_buffer.handle, grad_output.handle,
                                grad_next.handle,
                                batch_size, horizon, hidden_size))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_commit(command_batch))
                        self.assert_ok(
                            self.lib.puf_metal_command_batch_wait(command_batch))
                    finally:
                        self.lib.puf_metal_command_batch_destroy(command_batch)

                    actual_grads = (
                        grad_combined.array(
                            np.float32, combined_elements).reshape(
                                combined_values.shape),
                        grad_state.array(np.float32, state_elements).reshape(
                            state_values.shape),
                        grad_input.array(np.float32, sequence_elements).reshape(
                            input_values.shape),
                    )
                    for actual, expected in zip(actual_grads, expected_grads):
                        np.testing.assert_allclose(
                            actual, expected, rtol=2e-4, atol=2e-4)

    def test_mingru_cpu_backward_reference_is_a_finite_difference(self):
        rng = np.random.default_rng(2024)
        batch_size, horizon, hidden_size = 1, 4, 2
        combined = rng.uniform(
            -1, 1, (batch_size, horizon, 3 * hidden_size))
        state = rng.uniform(0.1, 1, (batch_size, hidden_size))
        inputs = rng.uniform(-1, 1, (batch_size, horizon, hidden_size))
        grad_output = rng.uniform(-1, 1, inputs.shape)
        grad_next = rng.uniform(-1, 1, state.shape)
        analytic = mingru_scan_backward_reference(
            combined, state, inputs, grad_output, grad_next)

        def loss(combined_arg, state_arg, inputs_arg):
            recurrent = state_arg.copy()
            outputs = np.empty_like(inputs_arg)
            for t in range(horizon):
                projected = combined_arg[:, t, :]
                hidden = projected[:, :hidden_size]
                gate = projected[:, hidden_size:2 * hidden_size]
                projection = projected[:, 2 * hidden_size:]
                gate_value = 1.0 / (1.0 + np.exp(-gate))
                hidden_tilde = np.where(
                    hidden >= 0, hidden + 0.5,
                    1.0 / (1.0 + np.exp(-hidden)))
                recurrent = ((1.0 - gate_value) * recurrent
                    + gate_value * hidden_tilde)
                projection_gate = 1.0 / (1.0 + np.exp(-projection))
                outputs[:, t, :] = (projection_gate * recurrent
                    + (1.0 - projection_gate) * inputs_arg[:, t, :])
            return float(np.sum(outputs * grad_output)
                + np.sum(recurrent * grad_next))

        epsilon = 1e-6
        checks = (
            (combined, analytic[0], (0, 2, 4)),
            (state, analytic[1], (0, 1)),
            (inputs, analytic[2], (0, 3, 0)),
        )
        for values, expected, index in checks:
            plus = values.copy()
            minus = values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            args_plus = [combined, state, inputs]
            args_minus = [combined, state, inputs]
            target = 0 if values is combined else 1 if values is state else 2
            args_plus[target] = plus
            args_minus[target] = minus
            numerical = ((loss(*args_plus) - loss(*args_minus))
                / (2 * epsilon))
            self.assertAlmostEqual(numerical, expected[index], places=7)

    def test_clamp_special_values_match_cuda_semantics(self):
        values = np.array([
            -np.inf, -2.0, -0.0, 0.0, 3.0, np.inf, np.nan,
        ], dtype=np.float32)
        with MetalBuffer(self, values.nbytes) as dst:
            actual = dst.array(np.float32, len(values))
            actual[:] = values
            self.assert_ok(self.lib.puf_metal_clamp_f32(
                self.context, dst.handle, -1.0, 2.0, len(values)))
            expected = np.fmin(np.fmax(values, -1.0), 2.0)
            np.testing.assert_array_equal(actual, expected)

    def test_repeated_dispatch_reuses_pipeline_safely(self):
        n = 1024
        with MetalBuffer(self, n * 4) as dst, MetalBuffer(self, n * 4) as src:
            dst.array(np.float32, n).fill(0.0)
            src.array(np.float32, n).fill(0.25)
            for _ in range(100):
                self.assert_ok(self.lib.puf_metal_add_f32(
                    self.context, dst.handle, src.handle, n))
            np.testing.assert_array_equal(
                dst.array(np.float32, n), np.full(n, 25.0, dtype=np.float32))

    def test_buffer_bounds_error_is_reported(self):
        with MetalBuffer(self, 4 * 4) as dst:
            status = self.lib.puf_metal_fill_f32(
                self.context, dst.handle, 0.0, 5)
            self.assertEqual(status, PUF_METAL_INVALID_ARGUMENT)
            self.assertIn('smaller', self.last_error())

    def test_missing_library_error_is_reported(self):
        error = ctypes.create_string_buffer(4096)
        context = self.lib.puf_metal_create(
            b'/definitely/missing/puffer.metallib', error, len(error))
        self.assertFalse(context)
        self.assertIn('Failed to load Metal library', error.value.decode())

    def test_shader_compile_error_is_reported(self):
        with tempfile.NamedTemporaryFile(suffix='.metal') as source:
            source.write(b'kernel void this_is_not_valid Metal source')
            source.flush()
            error = ctypes.create_string_buffer(4096)
            context = self.lib.puf_metal_create(
                os.fsencode(source.name), error, len(error))
        self.assertFalse(context)
        self.assertIn('Failed to load Metal library', error.value.decode())


if __name__ == '__main__':
    unittest.main()
